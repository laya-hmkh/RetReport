"""
Main execution script for training a vision-text model on the DeepEyeNet dataset.
This script coordinates the pipeline, defines global configurations, and integrates
the MedViT vision model with BioGpt for multimodal medical image captioning.
We are using MedViT-S in this code.
"""

import os
import logging
import torch
import time
import torchvision.transforms as transforms
from models import MedViT, VisionTextModel
from dataset import EyeNetDataset, load_and_process_json
from train import train_model_test
from transformers import BioGptForCausalLM, BioGptTokenizer
from peft import LoraConfig, get_peft_model
import argparse
from train import infer_caption
import json

# Configure logging for detailed debugging and monitoring
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

os.environ["NUMEXPR_MAX_THREADS"] = "8" # Adjust to CPU core count if needed | 16 | 20 is available
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Clear CUDA cache and optimize memory usage
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    
class Config:
    """Configuration class holding global hyperparameters and paths."""
    
    # Load from JSON config file
    def __init__(self, config_path='config.json'):
        with open(config_path, 'r') as f:
            cfg = json.load(f)
        
        # Make all keys attributes of the class
        for key, value in cfg.items():
            setattr(self, key.upper(), value)
        
        # Fixed values that don't change per experiment
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.VISION_DIM = 1024
        self.TEXT_DIM = 1024
        self.STEM_CHS = [64, 32, 64]
        self.DEPTHS = [3, 4, 20, 3]
        self.CONT_LOSS_WEIGHT = 0.5

        self.LORA_RANK = 8
        self.LORA_ALPHA = 16
        self.LORA_DROPOUT = 0.1

        self.PRETRAINED_PATH = 'MedViT_base_im1k.pth'
        self.OUTPUT_DIR = 'MedBio'

        self.JSON_PATHS = {
            'train': 'eyenet0420/DeepEyeNet_train.json',
            'val': 'eyenet0420/DeepEyeNet_valid.json',
            'test': 'eyenet0420/DeepEyeNet_test.json'
        }

        self.CUDA_ENV = {
            "CUDA_LAUNCH_BLOCKING": "1",
            "TORCH_USE_CUDA_DSA": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "garbage_collection_threshold:0.6, max_split_size_mb:32",
            "TF_ENABLE_ONEDNN_OPTS": "0"
        }


def setup_environment(config):
    """Set up environment variables for CUDA debugging and memory optimization."""
    for key, value in config.CUDA_ENV.items():  # Changed from Config.CUDA_ENV
        os.environ[key] = value
    if config.DEVICE.type == "cpu":  # Changed from Config.DEVICE
        logger.warning("CUDA not available, falling back to CPU. Training may be slow.")
    logger.info(f"Using device: {config.DEVICE}")  # Changed from Config.DEVICE

def define_transforms():
    """Define data augmentation and normalization transforms for training and evaluation."""
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    eval_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transforms, eval_transforms

def validate_config(config):
    """Validate configuration parameters."""
    if config.BATCH_SIZE < 1:
        raise ValueError("BATCH_SIZE must be >= 1")
    if config.LR <= 0:
        raise ValueError("Learning rate must be > 0")
    if not os.path.exists(config.PRETRAINED_PATH):
        raise FileNotFoundError(f"Pretrained weights not found: {config.PRETRAINED_PATH}")
    
    for split, path in config.JSON_PATHS.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"{split} JSON file not found: {path}")

def main():
    """Main pipeline for training or inference of the vision-text model."""
    parser = argparse.ArgumentParser(description="Train or infer with VisionTextModel")
    parser.add_argument("--mode", choices=["train", "infer"], default="train", help="Mode: train or infer")
    parser.add_argument("--image_path", type=str, help="Path to image for inference")
    parser.add_argument("--checkpoint_path", type=str, help="Path to checkpoint for inference")
    args = parser.parse_args()
    config = Config('config.json')
    validate_config(config)

    try:
        start_time = time.time()
        logger.info(f"Starting pipeline in {args.mode} mode")

        # Set up environment and random seed
        setup_environment(config)
        logger.info("Setting random seed for reproducibility")
        torch.manual_seed(42)

        # Initialize vision model (MedViT)
        logger.info("Initializing MedViT model")
        vision_model = MedViT(stem_chs=config.STEM_CHS, depths=config.DEPTHS, num_classes=None)
        logger.info(f"Loading pretrained weights from {config.PRETRAINED_PATH}")
        if not os.path.exists(config.PRETRAINED_PATH):
            logger.error(f"Pretrained weights file not found: {config.PRETRAINED_PATH}")
            raise FileNotFoundError(f"Pretrained weights file not found: {config.PRETRAINED_PATH}")
        vision_model.load_pretrained_weights(config.PRETRAINED_PATH)

        # Initialize text model and tokenizer (BioGpt) with optional LoRA
        logger.info("Initializing BioGPT model and tokenizer")
        text_model = BioGptForCausalLM.from_pretrained("microsoft/biogpt")
        if config.USE_LORA:
            lora_config = LoraConfig(
                r=config.LORA_RANK,
                lora_alpha=config.LORA_ALPHA,
                target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
                lora_dropout=config.LORA_DROPOUT,
                task_type="CAUSAL_LM"
            )
            text_model = get_peft_model(text_model, lora_config)
            logger.info(f"Applied LoRA to BioGpt with rank={config.LORA_RANK}, alpha={config.LORA_ALPHA}")
        tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")

        # Initialize combined vision-text model
        logger.info("Initializing VisionTextModel")
        model = VisionTextModel(
            vision_model=vision_model,
            text_model=text_model,
            vision_dim=config.VISION_DIM,
            text_dim=config.TEXT_DIM,
            config=config
        )

        # Define transforms
        logger.info("Defining transforms")
        train_transforms, eval_transforms = define_transforms()

        if args.mode == "train":
            # Load datasets
            logger.info("Loading datasets")
            for split, path in config.JSON_PATHS.items():
                if not os.path.exists(path):
                    logger.error(f"JSON file not found: {path}")
                    raise FileNotFoundError(f"JSON file not found: {path}")

            logger.info("Processing train dataset")
            train_paths, train_captions = load_and_process_json(config.JSON_PATHS['train'])
            logger.info("Processing validation dataset")
            val_paths, val_captions = load_and_process_json(config.JSON_PATHS['val'])
            logger.info("Processing test dataset")
            test_paths, test_captions = load_and_process_json(config.JSON_PATHS['test'])
            logger.info(f"Loaded {len(train_paths)} train, {len(val_paths)} val, {len(test_paths)} test samples")

            # Verify datasets are not empty
            if not train_paths or not val_paths or not test_paths:
                logger.error("One or more datasets are empty")
                raise ValueError("One or more datasets are empty")

            # Check sample image
            logger.info(f"Checking sample image: {train_paths[0]}")
            if not os.path.exists(train_paths[0]):
                logger.error(f"Sample image not found: {train_paths[0]}")
                raise FileNotFoundError(f"Sample image not found: {train_paths[0]}")

            # Initialize datasets
            logger.info("Initializing datasets")
            train_dataset = EyeNetDataset(train_paths, train_captions, train_transforms, tokenizer, config.MAX_LENGTH)
            val_dataset = EyeNetDataset(val_paths, val_captions, eval_transforms, tokenizer, config.MAX_LENGTH)
            test_dataset = EyeNetDataset(test_paths, test_captions, eval_transforms, tokenizer, config.MAX_LENGTH)
            logger.info("Datasets initialized")

            # Create output directory
            logger.info(f"Creating output directory: {config.OUTPUT_DIR}")
            os.makedirs(config.OUTPUT_DIR, exist_ok=True)

            # Run training
            logger.info("Running train_model_test")
            model, tokenizer, last_epoch = train_model_test(
                model=model,
                tokenizer=tokenizer,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                test_dataset=test_dataset,
                output_dir=config.OUTPUT_DIR,
                config=config,
                beam_search=config.BEAM_SEARCH,
                early_stopping=config.EARLY_STOPPING
            )
            logger.info(f"Training pipeline completed after epoch {last_epoch + 1}")

        elif args.mode == "infer":
            if not args.image_path or not os.path.exists(args.image_path):
                logger.error(f"Image path not provided or invalid: {args.image_path}")
                raise FileNotFoundError(f"Image path not provided or invalid: {args.image_path}")
            if args.checkpoint_path and not os.path.exists(args.checkpoint_path):
                logger.error(f"Checkpoint path not found: {args.checkpoint_path}")
                raise FileNotFoundError(f"Checkpoint path not found: {args.checkpoint_path}")

            # Run inference
            logger.info(f"Running inference on image: {args.image_path}")
            caption = infer_caption(
                model=model,
                tokenizer=tokenizer,
                image_path=args.image_path,
                transform=eval_transforms,
                device=config.DEVICE,
                checkpoint_path=args.checkpoint_path,
                max_length=config.MAX_LENGTH,
                beam_search=config.BEAM_SEARCH,
                early_stopping=config.EARLY_STOPPING
            )
            print(f"Generated Caption: {caption}")

        end_time = time.time()
        total_time = end_time - start_time
        hours, rem = divmod(total_time, 3600)
        minutes, seconds = divmod(rem, 60)
        logger.info(f"Total execution time: {int(hours)} h {int(minutes)} m {seconds:.2f} s")

    except Exception as e:
        logger.error(f"Error in pipeline: {str(e)}")
        raise


    
if __name__ == '__main__':
    # config = Config('config.json')  # You can change to another config file
    # print("Training mode:", config.EXP_MODE)
    # print("Batch size:", config.BATCH_SIZE)
    # print("Learning rate:", config.LR)
    main()
