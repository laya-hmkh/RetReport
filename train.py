"""
Training and evaluation functions for the vision-text model.
Implements training loop, validation, and testing with comprehensive metrics.
"""

import torch
import os
import logging
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from nltk.translate.bleu_score import corpus_bleu
from evaluate import load
from tqdm import tqdm
import nltk
from PIL import Image
from torch.amp import autocast, GradScaler

# Suppress Hugging Face warnings
import transformers
transformers.logging.set_verbosity_error()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["NUMEXPR_MAX_THREADS"] = "8"

try:
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    logger.info("NLTK data dowloaded successfully")

except Exception as e:
    logger.info(f"Failed to dowbload NLTK data: {str(e)}")

def compute_medical_accuracy(generated_texts, reference_texts):
    """Compute medical terminology accuracy."""
    medical_terms = [
        "diabetic retinopathy", "microaneurysms", "hard exudates", "soft exudates",
        "hemorrhages", "neovascularization", "macular edema", "optic disc",
        "cup disc ratio", "drusen", "pigmentation", "atrophy", "proliferative",
        "non-proliferative", "background retinopathy", "cotton wool spots"
    ]
    
    term_precision = 0
    term_recall = 0
    total_refs = 0
    total_gens = 0
    
    for gen, ref in zip(generated_texts, reference_texts):
        gen_terms = set(term for term in medical_terms if term in gen.lower())
        ref_terms = set(term for term in medical_terms if term in ref.lower())
        
        if ref_terms:
            total_refs += 1
            if gen_terms:
                term_recall += len(gen_terms.intersection(ref_terms)) / len(ref_terms)
        
        if gen_terms:
            total_gens += 1
            term_precision += len(gen_terms.intersection(ref_terms)) / len(gen_terms)
    
    avg_precision = term_precision / max(total_gens, 1)
    avg_recall = term_recall / max(total_refs, 1)
    f1_medical = 2 * (avg_precision * avg_recall) / max(avg_precision + avg_recall, 1e-8)
    
    return {
        'medical_precision': avg_precision,
        'medical_recall': avg_recall,
        'medical_f1': f1_medical
    }
    
def train_model_test(model, tokenizer, train_dataset, val_dataset, test_dataset, output_dir, config, beam_search=1, early_stopping=False):
    """
    Test training pipeline with checkpointing and evaluation.
   
    Args:
        model: VisionTextModel instance.
        tokenizer: BioGpt tokenizer.
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        test_dataset: Test dataset.
        output_dir (str): Directory to save outputs and checkpoints.
        config: Configuration object with hyperparameters.
        beam_search: Number of beams for decoding (1 for greedy, >1 for beam search).
        early_stopping: Whether to use early stopping in decoding.
   
    Returns:
        tuple: Trained model, tokenizer, and last epoch completed.
    """
    try:
        logger.info("Starting train_model_test")
        device = config.DEVICE
        logger.info(f"Using device: {device}")
        model.to(device)
        
        # ADD GRADIENT CHECKPOINTING HERE:
        if hasattr(config, 'USE_GRADIENT_CHECKPOINTING') and config.USE_GRADIENT_CHECKPOINTING:
            model.text_model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
            
        scaler = GradScaler('cuda') if config.MIXED_PRECISION else None

        # Log trainable parameters if LoRA is enabled
        if config.USE_LORA:
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            logger.info(f"LoRA enabled: {trainable_params}/{total_params} parameters trainable "
                        f"({trainable_params/total_params*100:.2f}%)")

        logger.info("Initializing optimizer")
        optimizer = AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)

        logger.info("Creating DataLoaders")
        train_dataloader = DataLoader(train_dataset,
                                      batch_size=config.BATCH_SIZE, 
                                      shuffle=True, 
                                      num_workers=config.NUM_WORKERS,
                                      prefetch_factor=config.PREFETCH_FACTOR,
                                      pin_memory=True)
        val_dataloader = DataLoader(val_dataset,
                                    batch_size=config.BATCH_SIZE,
                                    shuffle=False, 
                                    num_workers=config.NUM_WORKERS,
                                    prefetch_factor=config.PREFETCH_FACTOR,
                                    pin_memory=True)
        test_dataloader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
        logger.info(f"DataLoaders created: {len(train_dataloader)} train batches, {len(val_dataloader)} val batches")

        logger.info("Initializing scheduler")
        total_steps = len(train_dataloader) * config.NUM_EPOCHS // config.ACCUM_STEPS
        warmup_steps = int(config.WARMUP_RATIO * total_steps)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

        logger.info("Loading evaluation metrics")
        rouge = load("rouge")
        meteor = load("meteor")
        bertscore = load("bertscore")
        logger.info("Evaluation metrics loaded")

        # Check for existing checkpoint
        checkpoint_dir = os.path.join(output_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        start_epoch = 0
        checkpoint_path = os.path.join(checkpoint_dir, "latest_checkpoint.pth")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            logger.info(f"Resumed training from checkpoint at epoch {start_epoch}")

        for epoch in range(start_epoch, config.NUM_EPOCHS):
            logger.info(f"Starting epoch {epoch + 1}/{config.NUM_EPOCHS}")
            model.train()
            train_loss = 0
            train_lm_loss = 0
            train_cont_loss = 0
            optimizer.zero_grad()

            for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")):
                try:
                    if batch_idx % 50 == 0:  # Reduced from 10 to 50
                        logger.info(f"Processing batch {batch_idx}/{len(train_dataloader)}")
                        
                    # Move batch to device with non_blocking for efficiency
                    pixel_values = batch["pixel_values"].to(device, non_blocking=True)
                    input_ids = batch["input_ids"].to(device, non_blocking=True)
                    attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                    
                    # Forward pass with mixed precision if enabled
                    if config.MIXED_PRECISION:
                        with autocast('cuda'):
                            outputs = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask)
                            loss = outputs['total_loss'] / config.ACCUM_STEPS
                    else:
                        outputs = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask)
                        loss = outputs['total_loss'] / config.ACCUM_STEPS

                    lm_loss = outputs['lm_loss']
                    cont_loss = outputs['cont_loss']

                    # Only log every 50 batches to reduce overhead
                    if batch_idx % 50 == 0:
                        logger.info(f"Batch {batch_idx} - LM Loss: {lm_loss.item():.4f}, "
                            f"Contrastive Loss: {cont_loss.item():.4f}, Total Loss: {loss.item() * config.ACCUM_STEPS:.4f}")

                    # Backward pass
                    if config.MIXED_PRECISION:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    # Optimizer step with gradient accumulation
                    if (batch_idx + 1) % config.ACCUM_STEPS == 0:
                        if config.MIXED_PRECISION:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()

                    train_loss += loss.item() * config.ACCUM_STEPS
                    train_lm_loss += lm_loss.item()
                    train_cont_loss += cont_loss.item()

                # ENHANCED ERROR HANDLING STARTS HERE
                except RuntimeError as e:
                    if "out of memory" in str(e) or "CUDA out of memory" in str(e):
                        logger.warning(f"CUDA OOM at batch {batch_idx}, clearing cache and skipping batch...")
                        torch.cuda.empty_cache()
                        optimizer.zero_grad()
                        # Skip this batch but continue training
                        continue
                    elif "CUDA error" in str(e):
                        logger.error(f"CUDA error at batch {batch_idx}: {str(e)}")
                        torch.cuda.empty_cache()
                        # Try to recover by skipping batch
                        optimizer.zero_grad()
                        continue
                    else:
                        logger.error(f"Runtime error processing batch {batch_idx}: {str(e)}")
                        raise
                except ValueError as e:
                    logger.error(f"Value error at batch {batch_idx}: {str(e)}")
                    # Skip corrupted batch
                    continue
                except KeyError as e:
                    logger.error(f"Missing key in batch {batch_idx}: {str(e)}")
                    # Skip batch with missing data
                    continue
                except Exception as e:
                    logger.error(f"Unexpected error processing batch {batch_idx}: {str(e)}")
                    # For any other exception, we should stop training
                    raise

            train_loss /= len(train_dataloader)
            train_lm_loss /= len(train_dataloader)
            train_cont_loss /= len(train_dataloader)
            logger.info(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}, "
                    f"Train LM Loss: {train_lm_loss:.4f}, Train Cont Loss: {train_cont_loss:.4f}")

            # Validation
            logger.info(f"Starting validation for epoch {epoch + 1}")
            model.eval()
            val_loss = 0
            val_lm_loss = 0
            val_cont_loss = 0
            generated_texts = []
            reference_texts = []
            with torch.no_grad():
                for batch_idx, batch in enumerate(tqdm(val_dataloader, desc=f"Validating Epoch {epoch + 1}")):
                    try:
                        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
                        input_ids = batch["input_ids"].to(device, non_blocking=True)
                        attention_mask = batch["attention_mask"].to(device, non_blocking=True)

                        outputs = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask)
                        val_loss += outputs['total_loss'].item()
                        val_lm_loss += outputs['lm_loss'].item()
                        val_cont_loss += outputs['cont_loss'].item()
                        
                        # Use the new generate_caption method
                        generated_ids = model.generate_caption(
                            pixel_values=pixel_values,
                            tokenizer=tokenizer,
                            max_length=config.MAX_LENGTH,
                            num_beams=beam_search,
                            early_stopping=early_stopping
                        )
                        
                        gen_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                        ref_texts = tokenizer.batch_decode(input_ids, skip_special_tokens=True)

                        generated_texts.extend(gen_texts)
                        reference_texts.extend(ref_texts)

                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            logger.warning(f"CUDA OOM during validation batch {batch_idx}, skipping...")
                            torch.cuda.empty_cache()
                            continue
                        else:
                            logger.error(f"Error during validation batch {batch_idx}: {str(e)}")
                            continue
                    except Exception as e:
                        logger.warning(f"Error processing validation batch {batch_idx}: {str(e)}")
                        continue

            val_loss /= len(val_dataloader)
            val_lm_loss /= len(val_dataloader)
            val_cont_loss /= len(val_dataloader)
            logger.info(f"Epoch {epoch + 1} - Val Loss: {val_loss:.4f}, "
                       f"Val LM Loss: {val_lm_loss:.4f}, Val Cont Loss: {val_cont_loss:.4f}")

            # Compute metrics
            logger.info(f"Computing metrics for epoch {epoch + 1}")
            bleu_score = corpus_bleu([[ref.split()] for ref in reference_texts], [gen.split() for gen in generated_texts])
            rouge_scores = rouge.compute(predictions=generated_texts, references=reference_texts)
            meteor_score = meteor.compute(predictions=generated_texts, references=reference_texts)
            bertscore_results = bertscore.compute(predictions=generated_texts, references=reference_texts, lang="en")

            # Log metrics
            logger.info(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                        f"BLEU: {bleu_score:.4f}, ROUGE-1: {rouge_scores['rouge1']:.4f}, ROUGE-L: {rouge_scores['rougeL']:.4f}, "
                        f"METEOR: {meteor_score['meteor']:.4f}, BERTScore: {sum(bertscore_results['f1'])/len(bertscore_results['f1']):.4f}")

            medical_metrics = compute_medical_accuracy(generated_texts, reference_texts)

            # Save metrics to a file
            metrics_log_path = os.path.join(output_dir, "metrics.txt")
            logger.info(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                f"BLEU: {bleu_score:.4f}, ROUGE-1: {rouge_scores['rouge1']:.4f}, ROUGE-L: {rouge_scores['rougeL']:.4f}, "
                f"METEOR: {meteor_score['meteor']:.4f}, BERTScore: {sum(bertscore_results['f1'])/len(bertscore_results['f1']):.4f}, "
                f"Medical F1: {medical_metrics['medical_f1']:.4f}")
            logger.info(f"Metrics logged to {metrics_log_path}")
            
            with open(metrics_log_path, "a") as f:
                f.write(f"Epoch {epoch + 1}\tTrain Loss: {train_loss:.4f}\tVal Loss: {val_loss:.4f}\t"
                f"BLEU: {bleu_score:.4f}\tROUGE-1: {rouge_scores['rouge1']:.4f}\tROUGE-L: {rouge_scores['rougeL']:.4f}\t"
                f"METEOR: {meteor_score['meteor']:.4f}\tBERTScore: {sum(bertscore_results['f1'])/len(bertscore_results['f1']):.4f}\t"
                f"Medical F1: {medical_metrics['medical_f1']:.4f}\n")
            
            # Save sample captions for inspection
            logger.info(f"Saving sample captions for epoch {epoch + 1}")
            captions_file = os.path.join(output_dir, f"sample_captions_epoch_{epoch + 1}.txt")
            with open(captions_file, "w", encoding="utf-8") as f:
                for i, (gen, ref) in enumerate(zip(generated_texts[:5], reference_texts[:5]), 1):
                    f.write(f"Sample {i}:\n")
                    f.write(f"Generated: {gen.strip()}\n")
                    f.write(f"Reference: {ref.strip()}\n\n")
            logger.info(f"Sample captions saved to {captions_file}")

            # Clinical relevance
            logger.info(f"Computing clinical relevance for epoch {epoch + 1}")
            clinical_terms = ["diabetic retinopathy", "microaneurysms", "hard exudates", "right eye", "left eye"]
            clinical_matches = sum(1 for gen in generated_texts if any(term in gen.lower() for term in clinical_terms))
            logger.info(f"Clinical relevance: {clinical_matches}/{len(generated_texts)} captions contain clinical terms")

            # Save checkpoint
            logger.info(f"Saving checkpoint for epoch {epoch + 1}")
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pth")  # Unique checkpoint per epoch
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss
            }
            
            torch.save(checkpoint_data, checkpoint_path)
            logger.info(f"Checkpoint saved at {checkpoint_path}")
            latest_checkpoint_path = os.path.join(checkpoint_dir, "latest_checkpoint.pth")
            torch.save(checkpoint_data, latest_checkpoint_path)
            # Log losses to a separate file
            loss_log_path = os.path.join(output_dir, "losses.txt")
            with open(loss_log_path, "a") as f:
                f.write(f"Epoch {epoch + 1}\tLLM Loss: {train_lm_loss:.4f}\tContrastive Loss: {train_cont_loss:.4f}\tTotal Loss: {train_loss:.4f}\n")
            logger.info(f"Losses logged to {loss_log_path}")

        logger.info("train_model_test completed")
        return model, tokenizer, config.NUM_EPOCHS - 1  # Return the last epoch (0-based index)

    except Exception as e:
        logger.error(f"Error in train_model_test: {str(e)}")
        raise
    
def infer_caption(model, tokenizer, image_path, transform, device, checkpoint_path=None, max_length=128, beam_search=1, early_stopping=False):
    """
    Generate a caption for a single image using the specified checkpoint.
    
    Args:
        model: VisionTextModel instance.
        tokenizer: BioGpt tokenizer.
        image_path (str): Path to the input image.
        transform: Image transformation pipeline.
        device: Device to run inference on.
        checkpoint_path (str, optional): Path to the checkpoint to load. If None, use current model weights.
        max_length (int): Maximum length for generated captions.
        beam_search (int): Number of beams for decoding.
        early_stopping (bool): Whether to use early stopping in decoding.
    
    Returns:
        str: Generated caption.
    """
    try:
        logger.info(f"Starting inference for image: {image_path}")
        
        # Load checkpoint if provided
        if checkpoint_path:
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Checkpoint loaded from {checkpoint_path}")
        
        model.to(device)
        model.eval()
        
        # Load and preprocess image
        logger.info("Loading and transforming image")
        image = Image.open(image_path).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)  # [1, C, H, W]
        
        # Generate caption
        logger.info("Generating caption")
        with torch.no_grad():
            vision_features = model.vision_model(image)  # [1, vision_dim]
            vision_embed = vision_features.reshape(1, 1, vision_features.size(-1))  # [1, 1, vision_dim]
            generated_ids = model.text_model.generate(
                inputs_embeds=vision_embed,
                attention_mask=torch.ones(1, 1, device=device),
                max_length=max_length,
                num_beams=beam_search,
                early_stopping=early_stopping
            )
            caption = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        logger.info(f"Generated caption: {caption}")
        return caption
    
    except Exception as e:
        logger.error(f"Error in inference: {str(e)}")
        raise