"""
Updated training script for MedViT + BioGPT multimodal model
Aligned with current architecture and includes comprehensive monitoring
"""
import os
import warnings

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=no INFO, 2=no WARNING, 3=no ERROR
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations

# Suppress other common warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Optional: Suppress specific transformers warnings
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Avoids tokenizer parallelism warnings

# print("🔧 Environment configured to suppress TensorFlow/warning messages")

import torch
import torch.nn as nn
import os
import json
import logging
# import wandb
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, BioGptTokenizer, BioGptForCausalLM
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

# Import your modules
from models import MedViT, VisionTextModel
from dataset import load_and_process_json, EyeNetDataset
from utils import MetricLogger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TrainingConfig:
    """Configuration class for training hyperparameters"""
    def __init__(self, config_dict=None):
        # Load from config.json if provided
        if config_dict:
            for key, value in config_dict.items():
                setattr(self, key, value)
        
        # Set device
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Ensure all required attributes exist with defaults
        if not hasattr(self, 'BATCH_SIZE'):
            self.BATCH_SIZE = 16
        if not hasattr(self, 'NUM_EPOCHS'):
            self.NUM_EPOCHS = 50
        if not hasattr(self, 'LR'):
            self.LR = 5e-5
        if not hasattr(self, 'WEIGHT_DECAY'):
            self.WEIGHT_DECAY = 0.01
        if not hasattr(self, 'WARMUP_RATIO'):
            self.WARMUP_RATIO = 0.1
        if not hasattr(self, 'MAX_LENGTH'):
            self.MAX_LENGTH = 128
        if not hasattr(self, 'MIXED_PRECISION'):
            self.MIXED_PRECISION = True
        if not hasattr(self, 'NUM_WORKERS'):
            self.NUM_WORKERS = 4
        if not hasattr(self, 'PREFETCH_FACTOR'):
            self.PREFETCH_FACTOR = 2
        if not hasattr(self, 'ACCUM_STEPS'):
            self.ACCUM_STEPS = 2
        if not hasattr(self, 'USE_GRADIENT_CHECKPOINTING'):
            self.USE_GRADIENT_CHECKPOINTING = True
        if not hasattr(self, 'BEAM_SEARCH'):
            self.BEAM_SEARCH = 1
        if not hasattr(self, 'EARLY_STOPPING'):
            self.EARLY_STOPPING = False

def setup_biogpt_tokenizer():
    """Setup BioGPT tokenizer with proper special tokens"""
    logger.info("Setting up BioGPT tokenizer...")
    
    tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
    
    # Add special tokens if missing
    special_tokens = {
        'bos_token': '<s>',
        'eos_token': '</s>',
        'unk_token': '<unk>',
        'pad_token': '<pad>'
    }
    
    added_tokens = []
    for token_type, token_value in special_tokens.items():
        if getattr(tokenizer, token_type) is None:
            added_tokens.append(token_value)
            
    if added_tokens:
        tokenizer.add_special_tokens(special_tokens)
        
    logger.info(f"✅ BioGPT tokenizer ready. Vocab size: {len(tokenizer)}")
    return tokenizer

def create_model_and_tokenizer(config):
    """Create model and tokenizer with proper initialization"""
    logger.info("Creating model and tokenizer...")
    
    # Setup tokenizer
    tokenizer = setup_biogpt_tokenizer()
    
    # Create MedViT vision encoder
    medvit = MedViT(
        stem_chs=[64, 32, 64],
        depths=[3, 4, 20, 3],
        path_dropout=0.1,
        num_classes=None  # No classification head
    )
    
    # Load pretrained MedViT weights if available
    medvit_weights_path = "MedViT_base_im1k.pth"  # Update this path
    if os.path.exists(medvit_weights_path):
        try:
            medvit.load_pretrained_weights(medvit_weights_path)
            logger.info(f"✅ Loaded MedViT pretrained weights from {medvit_weights_path}")
        except Exception as e:
            logger.warning(f"Could not load MedViT weights: {e}")
    # logger.info("⚠️ Training MedViT from scratch (no pretrained weights)")
    
    # Create BioGPT text model
    biogpt = BioGptForCausalLM.from_pretrained("microsoft/biogpt")
    
    # Resize token embeddings if we added tokens
    if len(tokenizer) != biogpt.config.vocab_size:
        biogpt.resize_token_embeddings(len(tokenizer))
        logger.info(f"Resized BioGPT embeddings to {len(tokenizer)}")
    
    # Create integrated model
    model = VisionTextModel(
        vision_model=medvit,
        text_model=biogpt,
        vision_dim=1024,  # MedViT output dimension
        text_dim=biogpt.config.hidden_size,
        config=config
    )
    
    logger.info(f"✅ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    return model, tokenizer

def create_datasets(tokenizer, config):
    """Create train, validation, and test datasets"""
    logger.info("Creating datasets...")
    
    # Define transforms
    from torchvision import transforms
    from dataset import EyeNetDataset
    
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load datasets - update these paths to your actual dataset files
    train_json = "eyenet0420\DeepEyeNet_train.json"  # Update this
    val_json = "eyenet0420\DeepEyeNet_valid.json"      # Update this  
    test_json = "eyenet0420\DeepEyeNet_test.json"    # Update this
    
    # Check if files exist
    for json_file in [train_json, val_json, test_json]:
        if not os.path.exists(json_file):
            logger.error(f"Dataset file not found: {json_file}")
            raise FileNotFoundError(f"Please update the path to {json_file}")
    
    # Load data
    train_paths, train_captions = load_and_process_json(train_json)
    val_paths, val_captions = load_and_process_json(val_json)
    test_paths, test_captions = load_and_process_json(test_json)
    
    # Create datasets
    train_dataset = EyeNetDataset(train_paths, train_captions, train_transform, tokenizer, config.MAX_LENGTH)
    val_dataset = EyeNetDataset(val_paths, val_captions, val_transform, tokenizer, config.MAX_LENGTH)
    test_dataset = EyeNetDataset(test_paths, test_captions, val_transform, tokenizer, config.MAX_LENGTH)
    
    logger.info(f"✅ Datasets created:")
    logger.info(f"   Train: {len(train_dataset)} samples")
    logger.info(f"   Validation: {len(val_dataset)} samples") 
    logger.info(f"   Test: {len(test_dataset)} samples")
    
    return train_dataset, val_dataset, test_dataset

def compute_medical_metrics(generated_texts, reference_texts):
    """Compute medical terminology accuracy and other domain-specific metrics"""
    medical_terms = [
        "diabetic retinopathy", "microaneurysms", "hard exudates", "soft exudates",
        "hemorrhages", "neovascularization", "macular edema", "optic disc",
        "cup disc ratio", "drusen", "pigmentation", "atrophy", "proliferative",
        "non-proliferative", "background retinopathy", "cotton wool spots",
        "arteriovenous nicking", "flame hemorrhages", "dot hemorrhages"
    ]
    
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    valid_samples = 0
    
    for gen_text, ref_text in zip(generated_texts, reference_texts):
        gen_terms = set(term for term in medical_terms if term in gen_text.lower())
        ref_terms = set(term for term in medical_terms if term in ref_text.lower())
        
        if len(ref_terms) > 0 or len(gen_terms) > 0:
            valid_samples += 1
            
            # Calculate precision, recall, F1 for this sample
            if len(gen_terms) > 0:
                precision = len(gen_terms.intersection(ref_terms)) / len(gen_terms)
            else:
                precision = 0.0
                
            if len(ref_terms) > 0:
                recall = len(gen_terms.intersection(ref_terms)) / len(ref_terms)
            else:
                recall = 0.0
                
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            total_precision += precision
            total_recall += recall
            total_f1 += f1
    
    if valid_samples > 0:
        avg_precision = total_precision / valid_samples
        avg_recall = total_recall / valid_samples
        avg_f1 = total_f1 / valid_samples
    else:
        avg_precision = avg_recall = avg_f1 = 0.0
    
    return {
        'medical_precision': avg_precision,
        'medical_recall': avg_recall, 
        'medical_f1': avg_f1,
        'valid_samples': valid_samples
    }

def evaluate_model(model, dataloader, tokenizer, device, config, split_name="val"):
    """Comprehensive model evaluation"""
    model.eval()
    total_loss = 0
    total_lm_loss = 0
    total_cont_loss = 0
    
    generated_texts = []
    reference_texts = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Evaluating {split_name}")):
            try:
                pixel_values = batch["pixel_values"].to(device, non_blocking=True)
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                
                # Forward pass for loss calculation
                outputs = model(pixel_values, input_ids, attention_mask)
                total_loss += outputs['total_loss'].item()
                total_lm_loss += outputs['lm_loss'].item()
                total_cont_loss += outputs['cont_loss'].item()
                
                # Generate captions
                generated_ids = model.generate_caption(
                    pixel_values,
                    tokenizer,
                    max_length=config.MAX_LENGTH,
                    num_beams=config.BEAM_SEARCH,
                    early_stopping=config.EARLY_STOPPING
                )
                
                # Decode texts
                batch_generated = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                batch_reference = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
                
                generated_texts.extend(batch_generated)
                reference_texts.extend(batch_reference)
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.warning(f"OOM during evaluation batch {batch_idx}, skipping...")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise
            except Exception as e:
                logger.warning(f"Error in evaluation batch {batch_idx}: {e}")
                continue
    
    # Calculate average losses
    avg_loss = total_loss / len(dataloader)
    avg_lm_loss = total_lm_loss / len(dataloader)
    avg_cont_loss = total_cont_loss / len(dataloader)
    
    # Calculate medical metrics
    medical_metrics = compute_medical_metrics(generated_texts, reference_texts)
    
    return {
        'avg_loss': avg_loss,
        'avg_lm_loss': avg_lm_loss,
        'avg_cont_loss': avg_cont_loss,
        'generated_texts': generated_texts[:10],  # First 10 for inspection
        'reference_texts': reference_texts[:10],
        'medical_metrics': medical_metrics
    }

def save_sample_outputs(generated_texts, reference_texts, epoch, output_dir, num_samples=10):
    """Save sample generated vs reference texts"""
    samples_file = os.path.join(output_dir, f"samples_epoch_{epoch:03d}.txt")
    
    with open(samples_file, 'w', encoding='utf-8') as f:
        f.write(f"=== EPOCH {epoch} SAMPLE OUTPUTS ===\n\n")
        
        for i in range(min(num_samples, len(generated_texts))):
            f.write(f"--- SAMPLE {i+1} ---\n")
            f.write(f"GENERATED: {generated_texts[i].strip()}\n")
            f.write(f"REFERENCE: {reference_texts[i].strip()}\n")
            f.write("\n")
    
    logger.info(f"Sample outputs saved to {samples_file}")

def plot_training_curves(metrics_history, output_dir):
    """Plot and save training curves"""
    epochs = range(1, len(metrics_history['train_loss']) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training/Validation Loss
    axes[0, 0].plot(epochs, metrics_history['train_loss'], 'b-', label='Train Loss', alpha=0.7)
    axes[0, 0].plot(epochs, metrics_history['val_loss'], 'r-', label='Val Loss', alpha=0.7)
    axes[0, 0].set_title('Training & Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Language Modeling vs Contrastive Loss
    axes[0, 1].plot(epochs, metrics_history['train_lm_loss'], 'g-', label='LM Loss', alpha=0.7)
    if 'train_cont_loss' in metrics_history and any(x > 0 for x in metrics_history['train_cont_loss']):
        axes[0, 1].plot(epochs, metrics_history['train_cont_loss'], 'm-', label='Contrastive Loss', alpha=0.7)
    axes[0, 1].set_title('Loss Components')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Medical Metrics
    if 'medical_f1' in metrics_history:
        axes[1, 0].plot(epochs, metrics_history['medical_f1'], 'purple', label='Medical F1', alpha=0.7)
        axes[1, 0].plot(epochs, metrics_history['medical_precision'], 'orange', label='Medical Precision', alpha=0.7)
        axes[1, 0].plot(epochs, metrics_history['medical_recall'], 'brown', label='Medical Recall', alpha=0.7)
        axes[1, 0].set_title('Medical Terminology Metrics')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Learning Rate (if available)
    if 'learning_rate' in metrics_history:
        axes[1, 1].plot(epochs, metrics_history['learning_rate'], 'cyan', label='Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()

def train_model(config, output_dir, use_tensorboard=True):
    """Main training function with TensorBoard logging instead of wandb"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize TensorBoard instead of wandb
    writer = None
    if use_tensorboard:
        tensorboard_dir = os.path.join(output_dir, 'tensorboard_logs')
        writer = SummaryWriter(tensorboard_dir)
        logger.info(f"📊 TensorBoard logging to: {tensorboard_dir}")
        logger.info(f"🌐 Start TensorBoard with: tensorboard --logdir {tensorboard_dir}")
    
    
    logger.info(f"Starting training with output directory: {output_dir}")
    logger.info(f"Configuration: {vars(config)}")
    
    # Create model and datasets
    model, tokenizer = create_model_and_tokenizer(config)
    train_dataset, val_dataset, test_dataset = create_datasets(tokenizer, config)
    
    # Move model to device
    model.to(config.DEVICE)
    
    # Enable gradient checkpointing if specified
    if config.USE_GRADIENT_CHECKPOINTING:
        if hasattr(model.text_model, 'gradient_checkpointing_enable'):
            model.text_model.gradient_checkpointing_enable()
            logger.info("✅ Gradient checkpointing enabled")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        prefetch_factor=config.PREFETCH_FACTOR,
        pin_memory=True,
        drop_last=True  # For stable batch sizes
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        prefetch_factor=config.PREFETCH_FACTOR,
        pin_memory=True
    )
    
    # Setup optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=config.LR,
        weight_decay=config.WEIGHT_DECAY,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    total_steps = len(train_loader) * config.NUM_EPOCHS // config.ACCUM_STEPS
    warmup_steps = int(config.WARMUP_RATIO * total_steps)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Mixed precision scaler
    scaler = GradScaler('cuda') if config.MIXED_PRECISION and torch.cuda.is_available() else None
    
    logger.info(f"Training setup complete:")
    logger.info(f"  Total steps: {total_steps}")
    logger.info(f"  Warmup steps: {warmup_steps}")
    logger.info(f"  Mixed precision: {config.MIXED_PRECISION}")
    
    # Training metrics tracking
    metrics_history = {
        'train_loss': [],
        'val_loss': [],
        'train_lm_loss': [],
        'val_lm_loss': [],
        'train_cont_loss': [],
        'val_cont_loss': [],
        'medical_f1': [],
        'medical_precision': [],
        'medical_recall': [],
        'learning_rate': []
    }
    
    # Best model tracking
    best_val_loss = float('inf')
    best_medical_f1 = 0.0
    patience_counter = 0
    max_patience = 5
    
    # Training loop
    for epoch in range(config.NUM_EPOCHS):
        logger.info(f"\n{'='*60}")
        logger.info(f"EPOCH {epoch + 1}/{config.NUM_EPOCHS}")
        logger.info(f"{'='*60}")
        
        # Training phase
        model.train()
        train_metrics = {'total_loss': 0, 'lm_loss': 0, 'cont_loss': 0, 'count': 0}
        optimizer.zero_grad()
        
        train_pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(train_pbar):
            try:
                pixel_values = batch["pixel_values"].to(config.DEVICE, non_blocking=True)
                input_ids = batch["input_ids"].to(config.DEVICE, non_blocking=True)
                attention_mask = batch["attention_mask"].to(config.DEVICE, non_blocking=True)
                
                # Forward pass with mixed precision
                if config.MIXED_PRECISION and scaler:
                    with autocast('cuda'):
                        outputs = model(pixel_values, input_ids, attention_mask)
                        loss = outputs['total_loss'] / config.ACCUM_STEPS
                else:
                    outputs = model(pixel_values, input_ids, attention_mask)
                    loss = outputs['total_loss'] / config.ACCUM_STEPS
                
                # Backward pass
                if config.MIXED_PRECISION and scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # Optimizer step with gradient accumulation
                if (batch_idx + 1) % config.ACCUM_STEPS == 0:
                    if config.MIXED_PRECISION and scaler:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                    
                    scheduler.step()
                    optimizer.zero_grad()
                
                # Update metrics
                train_metrics['total_loss'] += outputs['total_loss'].item()
                train_metrics['lm_loss'] += outputs['lm_loss'].item()
                train_metrics['cont_loss'] += outputs['cont_loss'].item()
                train_metrics['count'] += 1
                
                # Update progress bar
                if batch_idx % 10 == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    train_pbar.set_postfix({
                        'Loss': f"{outputs['total_loss'].item():.4f}",
                        'LM': f"{outputs['lm_loss'].item():.4f}",
                        'Cont': f"{outputs['cont_loss'].item():.4f}",
                        'LR': f"{current_lr:.2e}"
                    })
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.warning(f"OOM at batch {batch_idx}, clearing cache...")
                    torch.cuda.empty_cache()
                    optimizer.zero_grad()
                    continue
                else:
                    raise
        
        # Calculate average training metrics
        avg_train_loss = train_metrics['total_loss'] / train_metrics['count']
        avg_train_lm_loss = train_metrics['lm_loss'] / train_metrics['count']
        avg_train_cont_loss = train_metrics['cont_loss'] / train_metrics['count']
        current_lr = scheduler.get_last_lr()[0]
        
        # Validation phase
        logger.info("Running validation...")
        val_results = evaluate_model(model, val_loader, tokenizer, config.DEVICE, config, "validation")
        
        # Log metrics
        logger.info(f"Epoch {epoch + 1} Results:")
        logger.info(f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {val_results['avg_loss']:.4f}")
        logger.info(f"  Train LM Loss: {avg_train_lm_loss:.4f} | Val LM Loss: {val_results['avg_lm_loss']:.4f}")
        logger.info(f"  Train Cont Loss: {avg_train_cont_loss:.4f} | Val Cont Loss: {val_results['avg_cont_loss']:.4f}")
        logger.info(f"  Medical F1: {val_results['medical_metrics']['medical_f1']:.4f}")
        logger.info(f"  Learning Rate: {current_lr:.2e}")
        
        # Update metrics history
        metrics_history['train_loss'].append(avg_train_loss)
        metrics_history['val_loss'].append(val_results['avg_loss'])
        metrics_history['train_lm_loss'].append(avg_train_lm_loss)
        metrics_history['val_lm_loss'].append(val_results['avg_lm_loss'])
        metrics_history['train_cont_loss'].append(avg_train_cont_loss)
        metrics_history['val_cont_loss'].append(val_results['avg_cont_loss'])
        metrics_history['medical_f1'].append(val_results['medical_metrics']['medical_f1'])
        metrics_history['medical_precision'].append(val_results['medical_metrics']['medical_precision'])
        metrics_history['medical_recall'].append(val_results['medical_metrics']['medical_recall'])
        metrics_history['learning_rate'].append(current_lr)
        
        # Save sample outputs
        save_sample_outputs(
            val_results['generated_texts'],
            val_results['reference_texts'],
            epoch + 1,
            output_dir
        )
        
        # Log to wandb if enabled
        if writer:
            writer.add_scalar('train/loss', avg_train_loss, epoch)
            writer.add_scalar('train/lm_loss', avg_train_lm_loss, epoch)
            writer.add_scalar('train/cont_loss', avg_train_cont_loss, epoch)
            writer.add_scalar('val/loss', val_results['avg_loss'], epoch)
            writer.add_scalar('val/lm_loss', val_results['avg_lm_loss'], epoch)
            writer.add_scalar('val/cont_loss', val_results['avg_cont_loss'], epoch)
            writer.add_scalar('medical/f1', val_results['medical_metrics']['medical_f1'], epoch)
            writer.add_scalar('medical/precision', val_results['medical_metrics']['medical_precision'], epoch)
            writer.add_scalar('medical/recall', val_results['medical_metrics']['medical_recall'], epoch)
            writer.add_scalar('learning_rate', current_lr, epoch)
            if epoch % 10 == 0:
                sample_images = pixel_values[:4]  # First 4 images
                writer.add_images('training_samples', sample_images, epoch)
    
        
        # Save checkpoints
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': val_results['avg_loss'],
            'medical_f1': val_results['medical_metrics']['medical_f1'],
            'config': vars(config)
        }
        
        # Save regular checkpoint
        torch.save(checkpoint, os.path.join(output_dir, f"checkpoint_epoch_{epoch+1:03d}.pth"))
        
        # Save best model based on medical F1 score
        if val_results['medical_metrics']['medical_f1'] > best_medical_f1:
            best_medical_f1 = val_results['medical_metrics']['medical_f1']
            torch.save(checkpoint, os.path.join(output_dir, "best_medical_f1.pth"))
            logger.info(f"🏆 New best medical F1: {best_medical_f1:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Save best model based on validation loss
        if val_results['avg_loss'] < best_val_loss:
            best_val_loss = val_results['avg_loss']
            torch.save(checkpoint, os.path.join(output_dir, "best_val_loss.pth"))
            logger.info(f"🏆 New best val loss: {best_val_loss:.4f}")
        
        # Plot training curves
        if (epoch + 1) % 5 == 0:  # Every 5 epochs
            plot_training_curves(metrics_history, output_dir)
        
        # Early stopping check
        if patience_counter >= max_patience:
            logger.info(f"Early stopping triggered after {patience_counter} epochs without improvement")
            break
        
        # Save metrics to JSON
        with open(os.path.join(output_dir, 'metrics_history.json'), 'w') as f:
            json.dump(metrics_history, f, indent=2)
    
    # Final evaluation on test set
    logger.info("Running final test evaluation...")
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2)
    
    # Load best model for test evaluation
    best_model_path = os.path.join(output_dir, "best_medical_f1.pth")
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded best model for test evaluation (Medical F1: {checkpoint['medical_f1']:.4f})")
    
    test_results = evaluate_model(model, test_loader, tokenizer, config.DEVICE, config, "test")
    
    logger.info("=== FINAL TEST RESULTS ===")
    logger.info(f"Test Loss: {test_results['avg_loss']:.4f}")
    logger.info(f"Test LM Loss: {test_results['avg_lm_loss']:.4f}")
    logger.info(f"Test Contrastive Loss: {test_results['avg_cont_loss']:.4f}")
    logger.info(f"Test Medical F1: {test_results['medical_metrics']['medical_f1']:.4f}")
    logger.info(f"Test Medical Precision: {test_results['medical_metrics']['medical_precision']:.4f}")
    logger.info(f"Test Medical Recall: {test_results['medical_metrics']['medical_recall']:.4f}")
    
    # Save final test results
    final_results = {
        'test_loss': test_results['avg_loss'],
        'test_lm_loss': test_results['avg_lm_loss'],
        'test_cont_loss': test_results['avg_cont_loss'],
        'test_medical_metrics': test_results['medical_metrics'],
        'best_val_loss': best_val_loss,
        'best_medical_f1': best_medical_f1,
        'total_epochs': epoch + 1,
        'config': vars(config)
    }
    
    with open(os.path.join(output_dir, 'final_results.json'), 'w') as f:
        json.dump(final_results, f, indent=2)
    
    # Save test samples
    save_sample_outputs(
        test_results['generated_texts'],
        test_results['reference_texts'],
        "FINAL_TEST",
        output_dir,
        num_samples=20
    )
    
    # Final training curves
    plot_training_curves(metrics_history, output_dir)
    
    if writer:
        writer.close()
    
    logger.info("🎉 Training completed successfully!")
    return model, tokenizer, final_results

def inference_single_image(model_path, image_path, tokenizer, transform=None, device=None):
    """
    Generate caption for a single image using trained model
    
    Args:
        model_path (str): Path to trained model checkpoint
        image_path (str): Path to input image
        tokenizer: BioGPT tokenizer
        transform: Image transformation pipeline
        device: Device to run inference on
    
    Returns:
        str: Generated caption
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if transform is None:
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    logger.info(f"Loading model from {model_path}")
    
    # Create model architecture (you'll need to adjust this based on your config)
    config = TrainingConfig()  # Default config
    model, _ = create_model_and_tokenizer(config)
    
    # Load trained weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    pixel_values = transform(image).unsqueeze(0).to(device)
    
    logger.info("Generating caption...")
    with torch.no_grad():
        generated_ids = model.generate_caption(
            pixel_values,
            tokenizer,
            max_length=128,
            num_beams=3,
            early_stopping=True
        )
        caption = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    logger.info(f"Generated caption: {caption}")
    return caption

def analyze_training_performance(metrics_history_path):
    """
    Analyze training performance from saved metrics
    
    Args:
        metrics_history_path (str): Path to metrics_history.json file
    """
    with open(metrics_history_path, 'r') as f:
        metrics = json.load(f)
    
    print("=== TRAINING PERFORMANCE ANALYSIS ===")
    print(f"Total epochs: {len(metrics['train_loss'])}")
    print(f"Best validation loss: {min(metrics['val_loss']):.4f} (epoch {metrics['val_loss'].index(min(metrics['val_loss'])) + 1})")
    print(f"Best medical F1: {max(metrics['medical_f1']):.4f} (epoch {metrics['medical_f1'].index(max(metrics['medical_f1'])) + 1})")
    
    # Check for overfitting
    final_train_loss = metrics['train_loss'][-5:]  # Last 5 epochs
    final_val_loss = metrics['val_loss'][-5:]
    
    avg_final_train = sum(final_train_loss) / len(final_train_loss)
    avg_final_val = sum(final_val_loss) / len(final_val_loss)
    
    if avg_final_val > avg_final_train * 1.2:
        print("⚠️  Warning: Model may be overfitting (val loss >> train loss)")
    else:
        print("✅ Model seems to be learning well (no severe overfitting)")
    
    # Learning rate analysis
    if 'learning_rate' in metrics:
        print(f"Initial learning rate: {metrics['learning_rate'][0]:.2e}")
        print(f"Final learning rate: {metrics['learning_rate'][-1]:.2e}")
    
    # Medical metrics trend
    if len(metrics['medical_f1']) > 10:
        early_f1 = sum(metrics['medical_f1'][:5]) / 5
        late_f1 = sum(metrics['medical_f1'][-5:]) / 5
        improvement = ((late_f1 - early_f1) / early_f1) * 100
        print(f"Medical F1 improvement: {improvement:.1f}% (from {early_f1:.3f} to {late_f1:.3f})")

def main():
    """Main training script"""
    
    # Load configuration from config.json
    config_path = "config.json"
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        config = TrainingConfig(config_dict)
        logger.info(f"✅ Loaded configuration from {config_path}")
    else:
        config = TrainingConfig()
        logger.info("⚠️  Using default configuration (config.json not found)")
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"experiments/medvit_biogpt_{timestamp}"
    try:
        # Uncomment this line when you've updated the dataset paths
        model, tokenizer, results = train_model(config, output_dir, use_tensorboard=True)
        
        logger.info("✅ Training script is ready!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()