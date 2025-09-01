
import os 
os.environ["WANDB_API_KEY"]="cf51efadb9fbcd28c7cc7da919cd44ea99a858cf"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import wandb
wandb.login()

import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from transformers import AdamW, get_linear_schedule_with_warmup
import os
from pathlib import Path
from datetime import datetime
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm, trange
# Evaluation imports
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import re
from collections import Counter
import nltk
# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
# Configuration
CONFIG = {
    'data_dir': 'mm_retinal_dataset/processed',
    'images_dir': 'mm_retinal_dataset/images',
    'model_name': 'blip2-opt-2.7b',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'batch_size': 8,
    'num_epochs': 10,
    'learning_rate': 2e-5,
    'output_dir': 'blip2_base1_checkpoints',
    'max_length': 256,
    'use_lora': True,
    'lora_r': 16,          # LoRA rank
    'lora_alpha': 32,      # LoRA scaling parameter
    'lora_dropout': 0.1,   # LoRA dropout
}
# Medical terms dictionary for clinical accuracy evaluation
MEDICAL_TERMS = {
    # Retinal pathologies
    'diabetic_retinopathy': ['diabetic retinopathy', 'dr', 'proliferative diabetic retinopathy', 'pdr', 'non-proliferative diabetic retinopathy', 'npdr'],
    'macular_degeneration': ['macular degeneration', 'amd', 'age-related macular degeneration', 'wet amd', 'dry amd'],
    'glaucoma': ['glaucoma', 'optic nerve damage', 'increased intraocular pressure', 'cup-to-disc ratio'],
    'hypertensive_retinopathy': ['hypertensive retinopathy', 'arteriovenous nicking', 'cotton wool spots'],
    'retinal_detachment': ['retinal detachment', 'rhegmatogenous', 'tractional', 'exudative'],
    
    # Retinal features
    'microaneurysms': ['microaneurysms', 'micro-aneurysms', 'small red dots'],
    'hard_exudates': ['hard exudates', 'lipid deposits', 'yellow deposits'],
    'soft_exudates': ['soft exudates', 'cotton wool spots', 'nerve fiber layer infarcts'],
    'hemorrhages': ['hemorrhages', 'bleeding', 'dot and blot hemorrhages', 'flame-shaped hemorrhages'],
    'neovascularization': ['neovascularization', 'new vessel formation', 'abnormal blood vessels'],
    'drusen': ['drusen', 'yellow deposits', 'hard drusen', 'soft drusen'],
    'pigment_epithelium': ['pigment epithelium', 'rpe', 'retinal pigment epithelium'],
    
    # Anatomical structures
    'optic_disc': ['optic disc', 'optic nerve head', 'disc'],
    'macula': ['macula', 'macular', 'fovea', 'foveal'],
    'blood_vessels': ['blood vessels', 'arteries', 'veins', 'vasculature'],
    'retina': ['retina', 'retinal'],
    
    # Severity indicators
    'mild': ['mild', 'early', 'minimal'],
    'moderate': ['moderate', 'intermediate'],
    'severe': ['severe', 'advanced', 'proliferative'],
    'normal': ['normal', 'healthy', 'no abnormalities', 'unremarkable']
}

class MetricsEvaluator:
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        self.smoothing_function = SmoothingFunction().method1
        
    def preprocess_text(self, text):
        """Preprocess text for evaluation"""
        # Convert to lowercase and remove extra whitespace
        text = text.lower().strip()
        # Remove punctuation for some metrics
        text = re.sub(r'[^\w\s]', '', text)
        return text
    
    def tokenize_text(self, text):
        """Tokenize text into words"""
        return self.preprocess_text(text).split()
    
    def compute_bleu_scores(self, reference, candidate):
        """Compute BLEU-1, BLEU-2, BLEU-3, BLEU-4 scores"""
        ref_tokens = self.tokenize_text(reference)
        cand_tokens = self.tokenize_text(candidate)
        
        # BLEU scores with different n-gram weights
        bleu_scores = {}
        for n in range(1, 5):
            weights = [1.0/n] * n + [0.0] * (4-n)
            try:
                score = sentence_bleu(
                    [ref_tokens], 
                    cand_tokens, 
                    weights=weights,
                    smoothing_function=self.smoothing_function
                )
                bleu_scores[f'bleu_{n}'] = score
            except:
                bleu_scores[f'bleu_{n}'] = 0.0
                
        return bleu_scores
    
    def compute_rouge_l(self, reference, candidate):
        """Compute ROUGE-L score"""
        try:
            scores = self.rouge_scorer.score(reference, candidate)
            return scores['rougeL'].fmeasure
        except:
            return 0.0
    
    def compute_meteor(self, reference, candidate):
        """Compute METEOR score"""
        try:
            ref_tokens = self.tokenize_text(reference)
            cand_tokens = self.tokenize_text(candidate)
            return meteor_score([ref_tokens], cand_tokens)
        except:
            return 0.0
    
    def compute_cider(self, references, candidates):
        """
        Simplified CIDEr implementation
        Note: For full CIDEr, you might want to use the official implementation
        This is a simplified version based on TF-IDF weighted n-gram matching
        """
        def get_ngrams(tokens, n):
            return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
        
        def compute_tf_idf(ref_corpus, cand_tokens, n):
            # Compute term frequency for candidate
            cand_ngrams = get_ngrams(cand_tokens, n)
            cand_tf = Counter(cand_ngrams)
            
            # Compute document frequency across reference corpus
            all_ngrams = []
            for ref in ref_corpus:
                ref_tokens = self.tokenize_text(ref)
                all_ngrams.extend(get_ngrams(ref_tokens, n))
            
            corpus_df = Counter(all_ngrams)
            corpus_size = len(ref_corpus)
            
            # Compute TF-IDF weighted score
            score = 0.0
            for ngram, tf in cand_tf.items():
                if ngram in corpus_df:
                    idf = np.log(corpus_size / corpus_df[ngram])
                    score += tf * idf
                    
            return score
        
        cider_scores = []
        for ref, cand in zip(references, candidates):
            ref_tokens = self.tokenize_text(ref)
            cand_tokens = self.tokenize_text(cand)
            
            # Compute CIDEr for n-grams 1 to 4
            scores = []
            for n in range(1, 5):
                score = compute_tf_idf([ref], cand_tokens, n)
                scores.append(score)
            
            # Average the scores (simplified CIDEr)
            cider_score = np.mean(scores) if scores else 0.0
            cider_scores.append(cider_score)
        
        return np.mean(cider_scores)
    
    def extract_medical_terms(self, text):
        """Extract medical terms from text"""
        text_lower = text.lower()
        found_terms = []
        
        for category, terms in MEDICAL_TERMS.items():
            for term in terms:
                if term in text_lower:
                    found_terms.append((category, term))
        
        return found_terms
    
    def compute_clinical_accuracy(self, references, candidates):
        """Compute clinical term accuracy"""
        correct_matches = 0
        total_medical_terms = 0
        
        for ref, cand in zip(references, candidates):
            ref_terms = set([term[0] for term in self.extract_medical_terms(ref)])
            cand_terms = set([term[0] for term in self.extract_medical_terms(cand)])
            
            # Count correct matches
            matches = ref_terms.intersection(cand_terms)
            correct_matches += len(matches)
            
            # Count total medical terms in reference
            total_medical_terms += len(ref_terms)
        
        if total_medical_terms == 0:
            return 0.0
        
        return correct_matches / total_medical_terms
    
    def evaluate_batch(self, references, candidates):
        """Evaluate a batch of predictions"""
        metrics = {}
        
        # Individual metrics
        bleu_scores = {'bleu_1': [], 'bleu_2': [], 'bleu_3': [], 'bleu_4': []}
        rouge_scores = []
        meteor_scores = []
        
        for ref, cand in zip(references, candidates):
            # BLEU scores
            bleu = self.compute_bleu_scores(ref, cand)
            for key, value in bleu.items():
                bleu_scores[key].append(value)
            
            # ROUGE-L
            rouge_scores.append(self.compute_rouge_l(ref, cand))
            
            # METEOR
            meteor_scores.append(self.compute_meteor(ref, cand))
        
        # Average individual metrics
        for key, scores in bleu_scores.items():
            metrics[key] = np.mean(scores)
        
        metrics['rouge_l'] = np.mean(rouge_scores)
        metrics['meteor'] = np.mean(meteor_scores)
        
        # Corpus-level metrics
        metrics['cider'] = self.compute_cider(references, candidates)
        metrics['clinical_accuracy'] = self.compute_clinical_accuracy(references, candidates)
        
        return metrics
class RetinalImageDataset(Dataset):
    def __init__(self, csv_file, images_dir, processor, split='train'):
        """
        Dataset for MM-Retinal images and captions
        """
        self.df = pd.read_csv(csv_file)
        self.images_dir = Path(images_dir)
        self.processor = processor
        self.split = split
        
        # Common image extensions to search for
        self.image_extensions = ['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp']
       
        # Image preprocessing transforms
        self.transform = transforms.Compose([
            transforms.Resize((384, 384)),  # BLIP-2 standard size
            transforms.ToTensor(),
        ])
       
        print(f"Loaded {split} dataset: {len(self.df)} samples")
        print(f"Modality distribution:")
        print(self.df['modality'].value_counts())
       
    def __len__(self):
        return len(self.df)
    
    def _find_image_with_extension(self, base_path):
        """
        Find image file with any common extension
        """
        for ext in self.image_extensions:
            full_path = base_path.with_suffix(ext)
            if full_path.exists():
                return full_path
        return None
   
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
       
        # Get image path - check multiple possible locations
        image_id = row['Image_ID']
        modality = row['modality']
       
        # Remove any existing extension from image_id
        image_id_base = Path(image_id).stem
       
        # Try different path structures with extension search
        possible_base_paths = [
            self.images_dir / modality / image_id_base,
            self.images_dir / image_id_base,
            self.images_dir / modality.lower() / image_id_base,
            self.images_dir / f"{modality.lower()}_images" / image_id_base
        ]
       
        image_path = None
        for base_path in possible_base_paths:
            image_path = self._find_image_with_extension(base_path)
            if image_path is not None:
                break
               
        if image_path is None:
            # Create list of all attempted paths for error message
            attempted_paths = []
            for base_path in possible_base_paths:
                for ext in self.image_extensions:
                    attempted_paths.append(str(base_path.with_suffix(ext)))
            
            raise FileNotFoundError(f"Image not found: {image_id} in any of the following locations:\n" + 
                                  "\n".join(attempted_paths[:10]) + 
                                  (f"\n... and {len(attempted_paths)-10} more" if len(attempted_paths) > 10 else ""))
       
        # Load and preprocess image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (384, 384), color='black')
       
        # Get caption - use en_caption_final if available, otherwise en_caption
        caption = row.get('en_caption_final', row.get('en_caption', ''))
        if pd.isna(caption) or caption == '':
            caption = f"{modality} image shows retinal findings."  # Fallback caption
           
        # Process with BLIP-2 processor
        processed = self.processor(
            images=image,
            text=caption,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=CONFIG['max_length']
        )
       
        return {
            'pixel_values': processed['pixel_values'].squeeze(0),
            'input_ids': processed['input_ids'].squeeze(0),
            'attention_mask': processed['attention_mask'].squeeze(0),
            'caption': caption,
            'image_id': image_id,
            'modality': modality
        }
    
def custom_collate_fn(batch):
    """Custom collate function to handle variable length sequences"""
    
    # Extract components
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    captions = [item['caption'] for item in batch]
    image_ids = [item['image_id'] for item in batch]
    modalities = [item['modality'] for item in batch]
    
    # Find max length in this batch
    input_ids_list = [item['input_ids'] for item in batch]
    attention_masks_list = [item['attention_mask'] for item in batch]
    
    max_len = max(len(ids) for ids in input_ids_list)
    
    # Pad to max length in batch
    padded_input_ids = []
    padded_attention_masks = []
    
    for input_ids, attention_mask in zip(input_ids_list, attention_masks_list):
        # Pad input_ids
        padding_length = max_len - len(input_ids)
        padded_ids = torch.cat([
            input_ids,
            torch.zeros(padding_length, dtype=input_ids.dtype)
        ])
        
        # Pad attention_mask
        padded_mask = torch.cat([
            attention_mask,
            torch.zeros(padding_length, dtype=attention_mask.dtype)
        ])
        
        padded_input_ids.append(padded_ids)
        padded_attention_masks.append(padded_mask)
    
    return {
        'pixel_values': pixel_values,
        'input_ids': torch.stack(padded_input_ids),
        'attention_mask': torch.stack(padded_attention_masks),
        'caption': captions,
        'image_id': image_ids,
        'modality': modalities
    }

def create_dataloaders(processor, batch_size=8):
    """Create train, validation, and test dataloaders"""
    
    data_dir = Path(CONFIG['data_dir'])
    images_dir = Path(CONFIG['images_dir'])
    
    # Create datasets
    train_dataset = RetinalImageDataset(
        csv_file=data_dir / 'train.csv',
        images_dir=images_dir,
        processor=processor,
        split='train'
    )
    
    val_dataset = RetinalImageDataset(
        csv_file=data_dir / 'val.csv', 
        images_dir=images_dir,
        processor=processor,
        split='val'
    )
    
    test_dataset = RetinalImageDataset(
        csv_file=data_dir / 'test.csv',
        images_dir=images_dir, 
        processor=processor,
        split='test'
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True, 
        collate_fn=custom_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
        collate_fn=custom_collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
        collate_fn=custom_collate_fn
    )
    
    print(f"\nDataLoaders created:")
    print(f"Train: {len(train_loader)} batches ({len(train_dataset)} samples)")
    print(f"Val: {len(val_loader)} batches ({len(val_dataset)} samples)")
    print(f"Test: {len(test_loader)} batches ({len(test_dataset)} samples)")
    
    return train_loader, val_loader, test_loader

def setup_blip2_for_finetuning(model_name, device="cuda"):
    """Setup BLIP-2 model for fine-tuning with Q-Former and LLM unfrozen"""
    processor = Blip2Processor.from_pretrained(model_name)
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device
    )
    
    if CONFIG['use_lora']:
        # Keep everything frozen initially
        for param in model.parameters():
            param.requires_grad = False
        
        # Apply LoRA configuration
        lora_config = LoraConfig(
        r=CONFIG['lora_r'],
        lora_alpha=CONFIG['lora_alpha'],
        # target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        # target_modules="all-linear",
        target_modules=['q_proj', 'k_proj', 'v_proj', 'query', 'key', 'value', 'dense', 'fc1', 'fc2'],
        lora_dropout=CONFIG['lora_dropout'],
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
        )
        
        # Apply LoRA to Q-Former and Language Model
        model.qformer = get_peft_model(model.qformer, lora_config)
        model.language_model = get_peft_model(model.language_model, lora_config)
    
    else: 
        # Freeze the Vision Transformer (ViT)
        for param in model.vision_model.parameters():
            param.requires_grad = False
        
        # UNFREEZE Q-former for domain adaptation
        for param in model.qformer.parameters():
            param.requires_grad = True
        
        # UNFREEZE language model for medical caption generation
        for param in model.language_model.parameters():
            param.requires_grad = True
    
    # Print parameter statistics
    frozen_params = 0
    total_params = 0
    trainable_params = 0
    
    print("Parameter breakdown:")
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
        else:
            frozen_params += param.numel()
            
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Frozen parameters: {frozen_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable percentage: {100 * trainable_params / total_params:.2f}%")
    
    return model, processor

def configure_for_generation_task(model):
    """Configure model for text generation tasks"""
    # Set pad token if not exists
    if model.language_model.config.pad_token_id is None:
        model.language_model.config.pad_token_id = model.language_model.config.eos_token_id
    
    return model

class BLIP2Trainer:
    def __init__(self, model, train_loader, val_loader, device="cuda", lr=2e-5, num_epochs=10):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        
        # Initialize metrics evaluator
        self.metrics_evaluator = MetricsEvaluator()
        
        # Move model to device
        self.model.to(device)
        
        if CONFIG['use_lora']:
            # Setup optimizer - for LoRA parameters only
            trainable_params = []
            for name, module in self.model.named_modules():
                if hasattr(module, 'lora_A') or hasattr(module, 'lora_B'):
                    trainable_params.extend([p for p in module.parameters() if p.requires_grad])
        else:
            # Setup optimizer - for trainable parameters
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        self.optimizer = AdamW(trainable_params, lr=lr, weight_decay=0.01)
        
        # Calculate total steps for scheduler
        self.total_steps = len(train_loader) * num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(0.1 * self.total_steps),
            num_training_steps=self.total_steps
        )
        
        # Loss tracking
        self.train_losses = []
        self.val_losses = []
        self.val_metrics_history = []
    
    def compute_loss(self, batch):
        """Compute loss for a batch"""
        pixel_values = batch['pixel_values'].to(self.device)
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        
        # Forward pass with labels for language modeling loss
        outputs = self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids  # Auto-regressive language modeling
        )
        
        return outputs.loss
    
    def train_epoch(self, epoch_num):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch_num} [Train]",
            unit="batch",
            dynamic_ncols=True,
            leave=False
        )
        
        for batch_idx, batch in enumerate(pbar):
            self.optimizer.zero_grad()
            
            # Compute loss
            loss = self.compute_loss(batch)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update parameters
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)
            current_lr = self.scheduler.get_last_lr()[0]

            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg': f'{avg_loss:.4f}',
                'LR': f'{current_lr:.2e}'
            })
                
            # Log to wandb if initialized
            if wandb.run is not None:
                wandb.log({
                    "batch_train_loss": loss.item(),
                    "learning_rate": current_lr,
                    "step": len(self.train_losses) * len(self.train_loader) + batch_idx
                })
        
        pbar.close()
        avg_loss = total_loss / len(self.train_loader)
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def validate_with_metrics(self, epoch_num):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        
        all_references = []
        all_candidates = []
        all_modalities = []

        # Create progress bar for validation
        pbar = tqdm(
            self.val_loader,
            desc=f"Epoch {epoch_num} [Val]",
            unit="batch",
            dynamic_ncols=True,
            leave=False
        )

        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
                loss = self.compute_loss(batch)
                total_loss += loss.item()
                avg_loss = total_loss / (batch_idx + 1)

                # Update progress bar
                pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Avg': f'{avg_loss:.4f}'})
                
        
                # Generate captions for metrics
                pixel_values = batch['pixel_values'].to(self.device)
                
                # Generate captions
                generated_ids = self.model.generate(
                    pixel_values=pixel_values,
                    max_length=150,
                    num_beams=3,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.model.language_model.config.eos_token_id
                )
                
                # Decode generated text
                generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
                
                # Collect references and candidates
                all_references.extend(batch['caption'])
                all_candidates.extend(generated_texts)
                all_modalities.extend(batch['modality'])
        
        pbar.close()
        
        # Compute loss        
        avg_loss = total_loss / len(self.val_loader)
        self.val_losses.append(avg_loss)
        
        # Compute comprehensive metrics
        metrics = self.metrics_evaluator.evaluate_batch(all_references, all_candidates)
        metrics['val_loss'] = avg_loss
        
        # Compute modality-specific metrics
        modality_metrics = {}
        unique_modalities = list(set(all_modalities))
        
        for modality in tqdm(unique_modalities, desc="Modality metrics", leave=False):
            modality_indices = [i for i, m in enumerate(all_modalities) if m == modality]
            modality_refs = [all_references[i] for i in modality_indices]
            modality_cands = [all_candidates[i] for i in modality_indices]
            
            if modality_refs:  # Only compute if we have samples for this modality
                modality_metrics[modality] = self.metrics_evaluator.evaluate_batch(
                    modality_refs, modality_cands
                )
        
        # Store metrics history
        metrics_with_modality = {**metrics, 'modality_metrics': modality_metrics}
        self.val_metrics_history.append(metrics_with_modality)
        
        return metrics_with_modality, list(zip(all_references[:5], all_candidates[:5], all_modalities[:5]))
    
    def validate(self, epoch_num=1):
        """Validate model (original method for compatibility)"""
        metrics, _ = self.validate_with_metrics(epoch_num)
        return metrics['val_loss']
    
    def generate_sample_captions(self, num_samples=3):
        """Generate sample captions for monitoring progress"""
        self.model.eval()
        samples = []
        
        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                if i >= num_samples:
                    break
                    
                pixel_values = batch['pixel_values'][:1].to(self.device)  # Take first sample
                
                # Generate caption
                generated_ids = self.model.generate(
                    pixel_values=pixel_values,
                    max_length=150,
                    num_beams=3,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.model.language_model.config.eos_token_id
                )
                
                # Decode generated text
                generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                original_caption = batch['caption'][0]
                image_id = batch['image_id'][0]
                modality = batch['modality'][0]
                
                samples.append({
                    'image_id': image_id,
                    'modality': modality,
                    'original': original_caption,
                    'generated': generated_text
                })
        
        return samples
    
    def save_checkpoint(self, path, epoch, val_loss):
        """Save model checkpoint"""
        if CONFIG['use_lora']:
            # Save LoRA adapters
            checkpoint = {
                'epoch': epoch,
                'val_loss': val_loss,
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'val_metrics_history': self.val_metrics_history
            }
            torch.save(checkpoint, path)
            
            # Save LoRA adapters separately
            lora_path = path.replace('.pt', '_lora_adapters')
            os.makedirs(lora_path, exist_ok=True)
            self.model.qformer.save_pretrained(lora_path + '/qformer')
            self.model.language_model.save_pretrained(lora_path + '/language_model')
        else:
            # Traditional full model saving
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'val_metrics_history': self.val_metrics_history
            }
            torch.save(checkpoint, path)
        
        print(f"✅ Checkpoint saved to {path}")
def comprehensive_evaluation(model, processor, test_loader, device="cuda", save_results=True):
    """
    Comprehensive evaluation on test set with all metrics
    """
    print("\n" + "="*50)
    print("COMPREHENSIVE EVALUATION")
    print("="*50)
    
    model.eval()
    metrics_evaluator = MetricsEvaluator()
    
    all_references = []
    all_candidates = []
    all_modalities = []
    all_image_ids = []
    
    print("Generating captions for test set...")


    # Create progress bar for evaluation
    pbar = tqdm(
        test_loader,
        desc="Generating captions",
        unit="batch",
        dynamic_ncols=True
    )

    
    with torch.no_grad():
        for batch in pbar:
            pixel_values = batch['pixel_values'].to(device)
            
            # Generate captions with multiple decoding strategies
            generated_ids = model.generate(
                pixel_values=pixel_values,
                max_length=150,
                num_beams=5,
                early_stopping=True,
                temperature=0.8,
                do_sample=True,
                pad_token_id=model.language_model.config.eos_token_id
            )
            
            # Decode generated text
            generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
            
            # Collect data
            all_references.extend(batch['caption'])
            all_candidates.extend(generated_texts)
            all_modalities.extend(batch['modality'])
            all_image_ids.extend(batch['image_id'])
    
    pbar.close()
    print(f"\nEvaluating {len(all_references)} samples...")
    
    # Compute overall metrics
    overall_metrics = metrics_evaluator.evaluate_batch(all_references, all_candidates)
    
    # Compute modality-specific metrics
    modality_metrics = {}
    unique_modalities = list(set(all_modalities))
    
    for modality in tqdm(unique_modalities, desc="Computing modality metrics", leave=False):
        modality_indices = [i for i, m in enumerate(all_modalities) if m == modality]
        modality_refs = [all_references[i] for i in modality_indices]
        modality_cands = [all_candidates[i] for i in modality_indices]
        
        if modality_refs:
            modality_metrics[modality] = metrics_evaluator.evaluate_batch(
                modality_refs, modality_cands
            )
            modality_metrics[modality]['sample_count'] = len(modality_refs)
    
    # Print results
    print("\n" + "="*50)
    print("OVERALL METRICS")
    print("="*50)
    print(f"BLEU-1: {overall_metrics['bleu_1']:.4f}")
    print(f"BLEU-2: {overall_metrics['bleu_2']:.4f}")
    print(f"BLEU-3: {overall_metrics['bleu_3']:.4f}")
    print(f"BLEU-4: {overall_metrics['bleu_4']:.4f}")
    print(f"ROUGE-L: {overall_metrics['rouge_l']:.4f}")
    print(f"METEOR: {overall_metrics['meteor']:.4f}")
    print(f"CIDEr: {overall_metrics['cider']:.4f}")
    print(f"Clinical Accuracy: {overall_metrics['clinical_accuracy']:.4f}")
    
    # Print modality-specific results
    print("\n" + "="*50)
    print("MODALITY-SPECIFIC METRICS")
    print("="*50)
    for modality, metrics in modality_metrics.items():
        print(f"\n{modality.upper()} ({metrics['sample_count']} samples):")
        print(f"  BLEU-1: {metrics['bleu_1']:.4f}")
        print(f"  BLEU-2: {metrics['bleu_2']:.4f}")
        print(f"  BLEU-3: {metrics['bleu_3']:.4f}")
        print(f"  BLEU-4: {metrics['bleu_4']:.4f}")
        print(f"  ROUGE-L: {metrics['rouge_l']:.4f}")
        print(f"  METEOR: {metrics['meteor']:.4f}")
        print(f"  CIDEr: {metrics['cider']:.4f}")
        print(f"  Clinical Accuracy: {metrics['clinical_accuracy']:.4f}")
    
    # Show some example predictions
    print("\n" + "="*50)
    print("SAMPLE PREDICTIONS")
    print("="*50)
    for i in range(min(5, len(all_references))):
        print(f"\nSample {i+1} ({all_modalities[i]}):")
        print(f"Image ID: {all_image_ids[i]}")
        print(f"Reference: {all_references[i]}")
        print(f"Generated: {all_candidates[i]}")
        
        # Show medical terms found
        ref_terms = metrics_evaluator.extract_medical_terms(all_references[i])
        gen_terms = metrics_evaluator.extract_medical_terms(all_candidates[i])
        print(f"Reference medical terms: {[term[1] for term in ref_terms]}")
        print(f"Generated medical terms: {[term[1] for term in gen_terms]}")
        print("-" * 40)
    
    # Save detailed results if requested
    if save_results:
        results_df = pd.DataFrame({
            'image_id': all_image_ids,
            'modality': all_modalities,
            'reference_caption': all_references,
            'generated_caption': all_candidates
        })
        
        # Add individual metrics for each sample
        individual_metrics = []
        for ref, cand in tqdm(zip(all_references, all_candidates), 
                            desc="Computing individual metrics", 
                            total=len(all_references),
                            leave=False):
            sample_metrics = {}
            
            # BLEU scores
            bleu = metrics_evaluator.compute_bleu_scores(ref, cand)
            sample_metrics.update(bleu)
            
            # Other metrics
            sample_metrics['rouge_l'] = metrics_evaluator.compute_rouge_l(ref, cand)
            sample_metrics['meteor'] = metrics_evaluator.compute_meteor(ref, cand)
            
            # Clinical accuracy for this sample
            ref_terms = set([term[0] for term in metrics_evaluator.extract_medical_terms(ref)])
            cand_terms = set([term[0] for term in metrics_evaluator.extract_medical_terms(cand)])
            matches = len(ref_terms.intersection(cand_terms))
            total_ref_terms = len(ref_terms)
            sample_metrics['clinical_accuracy'] = matches / total_ref_terms if total_ref_terms > 0 else 0.0
            
            individual_metrics.append(sample_metrics)
        
        # Add individual metrics to dataframe
        metrics_df = pd.DataFrame(individual_metrics)
        results_df = pd.concat([results_df, metrics_df], axis=1)
        
        # Save results
        results_path = os.path.join(CONFIG['output_dir'], 'evaluation_results.csv')
        results_df.to_csv(results_path, index=False)
        print(f"\n📊 Detailed results saved to {results_path}")
        
        # Save summary metrics
        summary_metrics = {
            'overall_metrics': overall_metrics,
            'modality_metrics': modality_metrics,
            'evaluation_date': datetime.now().isoformat(),
            'model_config': CONFIG
        }
        
        summary_path = os.path.join(CONFIG['output_dir'], 'evaluation_summary.json')
        import json
        with open(summary_path, 'w') as f:
            json.dump(summary_metrics, f, indent=2)
        print(f"📈 Summary metrics saved to {summary_path}")
    
    # Log to wandb
    if wandb.run is not None:
        wandb_metrics = {"test_" + k: v for k, v in overall_metrics.items()}
        
        # Add modality-specific metrics to wandb
        for modality, metrics in modality_metrics.items():
            for metric_name, value in metrics.items():
                if metric_name != 'sample_count':
                    wandb_metrics[f"test_{modality}_{metric_name}"] = value
        
        wandb.log(wandb_metrics)
    
    return overall_metrics, modality_metrics, results_df

def main_training_loop():
    """Main training function with integrated evaluation"""
    
    # Initialize wandb
    wandb.init(
        project="blip2-base1-finetuning",
        name=f"blip2-mm-retinal-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config=CONFIG
    )
    
    print("Setting up BLIP-2 for retinal image captioning...")
    
    # Setup model and processor
    model, processor = setup_blip2_for_finetuning(
        model_name=CONFIG['model_name'],
        device=CONFIG['device']
    )
    model = configure_for_generation_task(model)
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        processor=processor,
        batch_size=CONFIG['batch_size']
    )
    
    print(f"\n🚀 Starting BLIP-2 Fine-tuning")
    print(f"Device: {CONFIG['device']}")
    print(f"Model: {CONFIG['model_name']}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Initialize trainer
    trainer = BLIP2Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=CONFIG['device'],
        lr=CONFIG['learning_rate'],
        num_epochs=CONFIG['num_epochs']
    )
    
    # Create output directory
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    # Training loop with metrics evaluation
    print("\n" + "="*30)
    print("STARTING TRAINING")
    print("="*30)
    
    best_val_loss = float('inf')
    best_bleu_4 = 0.0
    
    for epoch in range(CONFIG['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{CONFIG['num_epochs']}")
        print("-" * 50)
        
        # Training phase
        print("Training...")
        train_loss = trainer.train_epoch(epoch+1)
        
        # Validation phase with comprehensive metrics
        print("Validating and computing metrics...")
        val_metrics, sample_predictions = trainer.validate_with_metrics()
        
        val_loss = val_metrics['val_loss']
        
        # Print results
        print(f"\nEpoch {epoch + 1} Results:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  BLEU-4: {val_metrics['bleu_4']:.4f}")
        print(f"  ROUGE-L: {val_metrics['rouge_l']:.4f}")
        print(f"  METEOR: {val_metrics['meteor']:.4f}")
        print(f"  CIDEr: {val_metrics['cider']:.4f}")
        print(f"  Clinical Accuracy: {val_metrics['clinical_accuracy']:.4f}")
        
        # Print sample generations
        print(f"\nSample Generations:")
        for i, (ref, gen, modality) in enumerate(sample_predictions[:2]):
            print(f"  Sample {i+1} ({modality}):")
            print(f"    Reference: {ref[:100]}...")
            print(f"    Generated: {gen[:100]}...")
            print()
        
        # Save checkpoint based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(CONFIG['output_dir'], "best_model_loss.pt")
            trainer.save_checkpoint(checkpoint_path, epoch, val_loss)
            
            # Save in HuggingFace format
            model_save_path = os.path.join(CONFIG['output_dir'], "best_model_loss_hf")
            if CONFIG['use_lora']:
                # Save base model and LoRA adapters separately
                os.makedirs(model_save_path, exist_ok=True)
                model.qformer.save_pretrained(model_save_path + '/qformer_lora')
                model.language_model.save_pretrained(model_save_path + '/language_model_lora')
                processor.save_pretrained(model_save_path)
            else:
                # Traditional full model saving
                model.save_pretrained(model_save_path)
                processor.save_pretrained(model_save_path)
            
            print(f"🏆 Best model (by loss) saved to {model_save_path}")
        
        # Also save checkpoint based on BLEU-4 score
        if val_metrics['bleu_4'] > best_bleu_4:
            best_bleu_4 = val_metrics['bleu_4']
            checkpoint_path = os.path.join(CONFIG['output_dir'], "best_model_bleu4.pt")
            trainer.save_checkpoint(checkpoint_path, epoch, val_loss)
            
            # Save in HuggingFace format
            model_save_path = os.path.join(CONFIG['output_dir'], "best_model_bleu4_hf")
            if CONFIG['use_lora']:
                os.makedirs(model_save_path, exist_ok=True)
                model.qformer.save_pretrained(model_save_path + '/qformer_lora')
                model.language_model.save_pretrained(model_save_path + '/language_model_lora')
                processor.save_pretrained(model_save_path)
            else:
                model.save_pretrained(model_save_path)
                processor.save_pretrained(model_save_path)
            
            print(f"🎯 Best model (by BLEU-4) saved to {model_save_path}")
        
        # Log epoch metrics to wandb
        if wandb.run is not None:
            wandb_log = {
                "epoch": epoch + 1,
                "epoch_train_loss": train_loss,
                "epoch_val_loss": val_loss,
                "best_val_loss": best_val_loss,
                "best_bleu_4": best_bleu_4
            }
            
            # Add all validation metrics
            for metric_name, value in val_metrics.items():
                if metric_name != 'modality_metrics':
                    wandb_log[f"val_{metric_name}"] = value
            
            # Add modality-specific metrics
            if 'modality_metrics' in val_metrics:
                for modality, metrics in val_metrics['modality_metrics'].items():
                    for metric_name, value in metrics.items():
                        if metric_name != 'sample_count':
                            wandb_log[f"val_{modality}_{metric_name}"] = value
            
            wandb.log(wandb_log)
    
    print(f"\n🎉 Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best BLEU-4 score: {best_bleu_4:.4f}")

    # Save final model
    final_path = os.path.join(CONFIG['output_dir'], "final_model")
    if CONFIG['use_lora']:
        os.makedirs(final_path, exist_ok=True)
        model.qformer.save_pretrained(final_path + '/qformer_lora')
        model.language_model.save_pretrained(final_path + '/language_model_lora')
        processor.save_pretrained(final_path)
    else:
        model.save_pretrained(final_path)
        processor.save_pretrained(final_path)

    print(f"Final model saved to {final_path}")
    
    # Comprehensive test set evaluation
    print("\n🔍 Running comprehensive test set evaluation...")
    overall_test_metrics, modality_test_metrics, test_results_df = comprehensive_evaluation(
        model=model,
        processor=processor,
        test_loader=test_loader,
        device=CONFIG['device'],
        save_results=True
    )
    
    # Final summary
    print("\n" + "="*50)
    print("FINAL TEST SET RESULTS")
    print("="*50)
    print(f"BLEU-1: {overall_test_metrics['bleu_1']:.4f}")
    print(f"BLEU-2: {overall_test_metrics['bleu_2']:.4f}")
    print(f"BLEU-3: {overall_test_metrics['bleu_3']:.4f}")
    print(f"BLEU-4: {overall_test_metrics['bleu_4']:.4f}")
    print(f"ROUGE-L: {overall_test_metrics['rouge_l']:.4f}")
    print(f"METEOR: {overall_test_metrics['meteor']:.4f}")
    print(f"CIDEr: {overall_test_metrics['cider']:.4f}")
    print(f"Clinical Accuracy: {overall_test_metrics['clinical_accuracy']:.4f}")
    
    wandb.finish()
    
    return model, processor, test_loader, overall_test_metrics, modality_test_metrics

def evaluate_pretrained_model(model_path, test_loader, processor=None, device="cuda"):
    """
    Evaluate a pre-trained model on test set
    """
    print(f"Loading model from {model_path}")
    
    if processor is None:
        processor = Blip2Processor.from_pretrained(CONFIG['model_name'])
    
    if CONFIG['use_lora']:
        # Load base model
        model = Blip2ForConditionalGeneration.from_pretrained(
            CONFIG['model_name'],
            torch_dtype=torch.float16,
            device_map=device
        )
        
        # Load LoRA adapters
        from peft import PeftModel
        model.qformer = PeftModel.from_pretrained(model.qformer, model_path + '/qformer_lora')
        model.language_model = PeftModel.from_pretrained(model.language_model, model_path + '/language_model_lora')
    else:
        # Load full fine-tuned model
        model = Blip2ForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map=device
        )
    
    model = configure_for_generation_task(model)
    
    # Run comprehensive evaluation
    overall_metrics, modality_metrics, results_df = comprehensive_evaluation(
        model=model,
        processor=processor,
        test_loader=test_loader,
        device=device,
        save_results=True
    )
    
    return overall_metrics, modality_metrics, results_df
def test_data_loading():
    """Test function to verify data loading works correctly"""
    print("Testing data loading...")
    
    # Setup processor
    processor = Blip2Processor.from_pretrained(CONFIG['model_name'])
    
    # Create a small test dataset
    test_df = pd.read_csv(Path(CONFIG['data_dir']) / 'train.csv').head(5)
    
    dataset = RetinalImageDataset(
        csv_file=Path(CONFIG['data_dir']) / 'train.csv',
        images_dir=Path(CONFIG['images_dir']),
        processor=processor,
        split='test'
    )
    
    # Test loading first sample
    try:
        sample = dataset[0]
        print("Data loading successful!")
        print(f"  Image shape: {sample['pixel_values'].shape}")
        print(f"  Caption length: {len(sample['caption'])}")
        print(f"  Input IDs shape: {sample['input_ids'].shape}")
        print(f"  Modality: {sample['modality']}")
        return True
    except Exception as e:
        print(f"Data loading failed: {e}")
        return False
def run_metrics_only_evaluation(model_path=None):
    """
    Run evaluation with metrics only (without training)
    """
    print("MM-Retinal BLIP-2 Evaluation Only")
    print("="*50)
    
    # Setup processor
    processor = Blip2Processor.from_pretrained(CONFIG['model_name'])
    
    # Create test dataloader
    _, _, test_loader = create_dataloaders(
        processor=processor,
        batch_size=CONFIG['batch_size']
    )
    
    if model_path:
        # Evaluate existing model
        overall_metrics, modality_metrics, results_df = evaluate_pretrained_model(
            model_path=model_path,
            test_loader=test_loader,
            processor=processor,
            device=CONFIG['device']
        )
    else:
        # Evaluate base model (no fine-tuning)
        print("Evaluating base BLIP-2 model (no fine-tuning)...")
        model, _ = setup_blip2_for_finetuning(
            model_name=CONFIG['model_name'],
            device=CONFIG['device']
        )
        model = configure_for_generation_task(model)
        
        overall_metrics, modality_metrics, results_df = comprehensive_evaluation(
            model=model,
            processor=processor,
            test_loader=test_loader,
            device=CONFIG['device'],
            save_results=True
        )
    
    return overall_metrics, modality_metrics, results_df

print("MM-Retinal BLIP-2 Fine-tuning Pipeline with Comprehensive Evaluation")
print("="*70)

# Test data loading first
if test_data_loading():
    print("\nData loading test passed.")
    
    # Choose execution mode
    print("\nExecution options:")
    print("1. Full training with evaluation")
    print("2. Evaluation only (base model)")
    print("3. Evaluation only (existing fine-tuned model)")
    
    # For automated execution, run full training
    print("\nStarting full training with evaluation...")
    
    # Run main training with integrated evaluation
    model, processor, test_loader, overall_metrics, modality_metrics = main_training_loop()
    
    print("\nTraining and evaluation completed successfully!")
    print("Check the output directory for detailed results and saved models.")
else:
    print("\nData loading test failed. Please check your data paths and files.")

# Additional utility functions for post-training analysis

def analyze_medical_term_coverage():
    """Analyze coverage of medical terms in the dataset with progress tracking"""
    print("\nAnalyzing medical term coverage in dataset...")
    
    data_dir = Path(CONFIG['data_dir'])
    
    for split in ['train', 'val', 'test']:
        df = pd.read_csv(data_dir / f'{split}.csv')
        print(f"\n{split.upper()} set analysis:")
        
        term_counts = {}
        total_samples = len(df)
        
        # Progress bar for analyzing medical terms
        for category, terms in tqdm(MEDICAL_TERMS.items(), 
                                  desc=f"Analyzing {split} medical terms", 
                                  leave=False):
            count = 0
            for _, row in df.iterrows():
                caption = str(row.get('en_caption_final', row.get('en_caption', ''))).lower()
                if any(term in caption for term in terms):
                    count += 1
            term_counts[category] = count
            print(f"  {category}: {count}/{total_samples} ({100*count/total_samples:.1f}%)")
    
    return term_counts

def compare_models_metrics(model_paths, test_loader, processor):
    """Compare multiple models on the same test set with progress tracking"""
    print("\nComparing multiple models...")
    
    results = {}
    
    # Progress bar for model comparison
    for model_name in tqdm(model_paths.keys(), desc="Evaluating models", unit="model"):
        model_path = model_paths[model_name]
        overall_metrics, modality_metrics, _ = evaluate_pretrained_model(
            model_path=model_path,
            test_loader=test_loader,
            processor=processor,
            device=CONFIG['device']
        )
        results[model_name] = overall_metrics
    
    # Create comparison table
    comparison_df = pd.DataFrame(results).T
    print("\nModel Comparison:")
    print(comparison_df.round(4))
    
    # Save comparison
    comparison_path = os.path.join(CONFIG['output_dir'], 'model_comparison.csv')
    comparison_df.to_csv(comparison_path)
    print(f"Comparison saved to {comparison_path}")
    
    return comparison_df

