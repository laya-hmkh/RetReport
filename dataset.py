"""
Dataset utilities for the DeepEyeNet dataset.
Handles image loading, caption preprocessing, and dataset creation.
"""

import os
import re
import json
import logging
from PIL import Image
from torch.utils.data import Dataset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def preprocess_caption(keywords, clinical_desc):  
    """
    Preprocess captions by normalizing text and expanding medical abbreviations.
    Enhanced version that handles redundancy and improves medical report structure.
    
    Args:
        keywords (str): Keywords associated with the clinical description.
        clinical_desc (str): Clinical description of the medical image.
    
    Returns:
        str: Processed caption combining keywords and description in medical report format.
    """
    # Clean and normalize inputs
    clinical_desc = clinical_desc.strip() if clinical_desc else ""
    keywords = keywords.strip() if keywords else ""
    
    # Remove trailing commas and extra spaces
    clinical_desc = re.sub(r',\s*$', '', clinical_desc)
    keywords = re.sub(r',\s*$', '', keywords)
    
    # Normalize slashes and measurements first
    clinical_desc = re.sub(r'(\d+)\\(\d+)', r'\1/\2', clinical_desc)
    keywords = re.sub(r'(\d+)\\(\d+)', r'\1/\2', keywords)
    keywords = re.sub(r'\s*\\\s*', ', ', keywords)
    
    # Normalize spacing but preserve case for medical terms initially
    clinical_desc = re.sub(r'\s+', ' ', clinical_desc)
    keywords = re.sub(r'\s+', ' ', keywords)
    
    # Enhanced abbreviation map
    abbreviation_map = {
        'fa': 'fluorescein angiography',
        'oct': 'optical coherence tomography',
        'bdr': 'background diabetic retinopathy',
        'srnv': 'subretinal neovascularization',
        're': 'right eye',
        'le': 'left eye',
        'ou': 'both eyes',
        'pdr': 'proliferative diabetic retinopathy',
        'npdr': 'non-proliferative diabetic retinopathy',
        'dme': 'diabetic macular edema',
        'cme': 'cystoid macular edema',
        'cnv': 'choroidal neovascularization',
        'rpe': 'retinal pigment epithelium',
        'rnfl': 'retinal nerve fiber layer',
        'epiretinal': 'epiretinal membrane',
        'vh': 'vitreous hemorrhage',
        'ppa': 'peripapillary atrophy',
        'od': 'optic disc',
        'cup/disc': 'cup to disc ratio',
        'c/d': 'cup to disc ratio'
    }
    
    # Apply abbreviation expansion to both (case-insensitive)
    for abbr, full in abbreviation_map.items():
        clinical_desc = re.sub(r'\b' + abbr + r'\b', full, clinical_desc, flags=re.IGNORECASE)
        keywords = re.sub(r'\b' + abbr + r'\b', full, keywords, flags=re.IGNORECASE)
    
    # Now convert to lowercase after abbreviation expansion
    clinical_desc = clinical_desc.lower()
    keywords = keywords.lower()
    
    # Handle redundancy - remove keywords that are already in clinical description
    if keywords and clinical_desc:
        # Split keywords and check for redundancy
        keyword_list = [kw.strip() for kw in keywords.split(',')]
        unique_keywords = []
        
        for kw in keyword_list:
            # Only include keyword if it's not already substantially covered in clinical_desc
            if kw not in clinical_desc and not any(word in clinical_desc for word in kw.split() if len(word) > 3):
                unique_keywords.append(kw)
        
        # Reconstruct keywords without redundancy
        keywords = ', '.join(unique_keywords) if unique_keywords else ""
    
    # Extract patient demographics if present
    age_match = re.search(r'(\d+)-year-old', clinical_desc)
    gender_match = re.search(r'\b(male|female|man|woman)\b', clinical_desc)
    
    # Create enhanced medical report structure
    if keywords and clinical_desc:
        # Both keywords and description available
        if age_match or gender_match:
            # Has demographic info
            demographics = []
            if age_match:
                demographics.append(f"{age_match.group(1)}-year-old")
            if gender_match:
                demographics.append(gender_match.group(1))
            demo_str = " ".join(demographics)
            
            # Remove demographics from clinical_desc to avoid repetition
            clean_desc = re.sub(r'\d+-year-old\s*', '', clinical_desc)
            clean_desc = re.sub(r'\b(male|female|man|woman)\b,?\s*', '', clean_desc)
            clean_desc = re.sub(r'^,\s*', '', clean_desc)  # Remove leading comma
            clean_desc = re.sub(r'\s+', ' ', clean_desc).strip()
            
            if clean_desc and clean_desc != keywords:
                combined = f"patient: {demo_str}. findings: {keywords}. description: {clean_desc}"
            else:
                combined = f"patient: {demo_str}. findings: {keywords}"
        else:
            # No demographics
            combined = f"findings: {keywords}. description: {clinical_desc}"
    
    elif keywords:
        # Only keywords available
        combined = f"findings: {keywords}"
    
    elif clinical_desc:
        # Only clinical description available
        if age_match or gender_match:
            demographics = []
            if age_match:
                demographics.append(f"{age_match.group(1)}-year-old")
            if gender_match:
                demographics.append(gender_match.group(1))
            demo_str = " ".join(demographics)
            
            clean_desc = re.sub(r'\d+-year-old\s*', '', clinical_desc)
            clean_desc = re.sub(r'\b(male|female|man|woman)\b,?\s*', '', clean_desc)
            clean_desc = re.sub(r'^,\s*', '', clean_desc)
            clean_desc = re.sub(r'\s+', ' ', clean_desc).strip()
            
            if clean_desc:
                combined = f"patient: {demo_str}. description: {clean_desc}"
            else:
                combined = f"patient: {demo_str}"
        else:
            combined = f"description: {clinical_desc}"
    else:
        # Fallback
        combined = "no clinical information available"
    
    # Final cleanup
    combined = re.sub(r'\s+', ' ', combined).strip()
    combined = re.sub(r'\.+', '.', combined)  # Remove multiple dots
    combined = re.sub(r'\.$', '', combined) + '.'  # Ensure single final dot
    
    return combined

def load_and_process_json(json_file):
    """
    Load and process JSON file containing image paths and captions.
    
    Args:
        json_file (str): Path to the JSON file.
    
    Returns:
        tuple: Lists of valid image paths and corresponding captions.
    """
    # Check if JSON file exists
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"JSON file not found: {json_file}")
    # Load JSON file
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    # Handle JSON Parsing Errors
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in {json_file}: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Error reading {json_file}: {str(e)}")
    # Initialize lists for image paths and their captions
    image_paths, captions = [], []
    # Process JSON data
    for item in data:
        # Validate item format
        if not isinstance(item, dict) or len(item) != 1:
            logger.warning(f"Invalid item format: {item}")
            continue
        # Extract image path and value
        image_path, value = next(iter(item.items()))
        # Validate value format
        if not isinstance(value, dict) or "keywords" not in value or "clinical-description" not in value:
            logger.warning(f"Invalid value format for {image_path}: {value}")
            continue
        # Normalize and Validate Image path
        full_path = os.path.normpath(os.path.join(image_path))
        # Check if image file exists
        if os.path.exists(full_path):
            image_paths.append(full_path)
            caption = preprocess_caption(value["keywords"], value["clinical-description"])
            captions.append(caption)
        # else:
        #     logger.warning(f"Image not found at {full_path}")
    return image_paths, captions

class EyeNetDataset(Dataset):
    """
    Custom dataset for the DeepEyeNet dataset, pairing medical images with captions.
    """
    def __init__(self, image_paths, captions, transform, tokenizer, max_length):
        """
        Initialize the dataset.
        
        Args:
            image_paths (list): List of image file paths.
            captions (list): List of corresponding captions.
            transform: Image transformation pipeline.
            tokenizer: Text tokenizer for captions.
            max_length (int): Maximum length for tokenized captions.
        """
        self.image_paths = image_paths
        self.captions = captions
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, index):
        try:
            if index < 5 or index % 1000 == 0: 
                logger.info(f"Loading image{index}")
            image = Image.open(self.image_paths[index]).convert("RGB")
            image = self.transform(image)
            caption = self.captions[index]
            inputs = self.tokenizer(
                caption,
                return_tensors="pt",
                padding="max_length",
                max_length=self.max_length,
                truncation=True
            )
            
            if index < 5:  # Print only first few for sanity
                print(f"\n--- Sample {index} ---")
                print(f"Raw caption: {caption}")
                print(f"Tokenized input_ids: {inputs['input_ids'].squeeze().tolist()}")
                print(f"Decoded: {self.tokenizer.decode(inputs['input_ids'].squeeze())}")
                print(f"Attention mask: {inputs['attention_mask'].squeeze().tolist()}")
                print(f"Image tensor shape: {image.shape}")
            
            return {
                "pixel_values": image,
                "input_ids": inputs["input_ids"].squeeze(),
                "attention_mask": inputs["attention_mask"].squeeze()
            }
        except Exception as e:
            logger.error(f"Error in EyeNetDataset.__getitem__ for index {index}: {str(e)}")
            raise