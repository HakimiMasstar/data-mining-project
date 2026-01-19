import os
import random
import shutil
import glob
import numpy as np
import tqdm

# Configuration
SOURCE_DATASET = 'Datasets'
TARGET_DATASET = 'Global_Supply_Chain_Simulation'

# Month mapping
MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]

COUNTRIES = ["USA", "Brazil", "India"]

# Rot Probability Logic (0.05 = Very Fresh, 0.95 = Very Rotten)
SEASONALITY = {
    "Apple": {
        # USA: Harvest Aug-Nov. Good Q4. Bad Summer (aging storage).
        "USA":    [0.1, 0.1, 0.1, 0.2, 0.3, 0.5, 0.6, 0.1, 0.05, 0.05, 0.05, 0.1],
        # Brazil: Harvest Jan-May. Good Q1-Q2.
        "Brazil": [0.1, 0.05, 0.05, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.5, 0.3],
        # India: Harvest Jul-Oct. Good Q3.
        "India":  [0.3, 0.4, 0.5, 0.6, 0.7, 0.6, 0.2, 0.1, 0.05, 0.05, 0.2, 0.3],
    },
    "Orange": {
        # USA: Harvest Nov-May. Good Winter/Spring.
        "USA":    [0.05, 0.05, 0.05, 0.1, 0.1, 0.3, 0.5, 0.6, 0.5, 0.3, 0.1, 0.05],
        # Brazil: Harvest Jun-Sept. Good Summer.
        "Brazil": [0.6, 0.5, 0.4, 0.3, 0.2, 0.05, 0.05, 0.05, 0.1, 0.3, 0.5, 0.7],
        # India: Harvest Sep-Dec & Jan-Feb. Good Q1 & Q4.
        "India":  [0.1, 0.1, 0.3, 0.4, 0.5, 0.6, 0.5, 0.3, 0.1, 0.1, 0.1, 0.1],
    },
    "Banana": {
        # USA: Imported. Stable. Slight heat issues in July/Aug.
        "USA":    [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1],
        # Brazil: Year-round.
        "Brazil": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        # India: Year-round. **STRIKE IN MAY** (High Rot).
        "India":  [0.1, 0.1, 0.1, 0.1, 0.9, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    }
}

def get_all_images(dataset_root):
    """
    Recursively finds all images in the source dataset.
    Returns dictionary: {'freshapples': [paths], 'rottenapples': [paths], ...}
    """
    images = {}
    classes = ['freshapples', 'rottenapples', 'freshbanana', 'rottenbanana', 'freshoranges', 'rottenoranges']
    
    print("Scanning source dataset...")
    for cls in classes:
        # Search recursively in Datasets/train/cls and Datasets/test/cls and Datasets/cls
        pattern = os.path.join(dataset_root, '**', cls, '*')
        files = glob.glob(pattern, recursive=True)
        # Filter for image extensions
        img_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        images[cls] = img_files
        print(f"  - {cls}: {len(img_files)} images")
        
    return images

def get_destination_probabilities(fruit, is_fresh):
    """
    Returns a list of bins (Country, Month) and their associated probability weights.
    Fresh images -> Weighted towards Low Rot Probability.
    Rotten images -> Weighted towards High Rot Probability.
    """
    bins = []
    weights = []
    
    for country in COUNTRIES:
        probs = SEASONALITY[fruit][country]
        for month_idx, rot_prob in enumerate(probs):
            month = MONTHS[month_idx]
            bins.append((country, month))
            
            if is_fresh:
                # Probability of being FRESH = 1 - Rot_Prob
                # Add epsilon to ensure non-zero chance
                weights.append((1.0 - rot_prob) + 0.01)
            else:
                # Probability of being ROTTEN = Rot_Prob
                weights.append(rot_prob + 0.01)
                
    # Normalize weights
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]
    
    return bins, normalized_weights

def main():
    if os.path.exists(TARGET_DATASET):
        print(f"Removing existing {TARGET_DATASET}...")
        shutil.rmtree(TARGET_DATASET)
        
    all_images = get_all_images(SOURCE_DATASET)
    
    fruit_map = {
        "freshapples": "Apple", "rottenapples": "Apple",
        "freshbanana": "Banana", "rottenbanana": "Banana",
        "freshoranges": "Orange", "rottenoranges": "Orange"
    }
    
    print(f"\nGenerating Global Supply Chain Simulation in '{TARGET_DATASET}'...")
    
    count = 0
    
    for cls, img_list in all_images.items():
        if not img_list: continue
        
        fruit = fruit_map[cls]
        is_fresh = "fresh" in cls
        
        bins, weights = get_destination_probabilities(fruit, is_fresh)
        
        # Vectorized choice for speed
        # We assign a bin index to every image in this class at once
        chosen_bin_indices = np.random.choice(len(bins), size=len(img_list), p=weights)
        
        for i, img_path in enumerate(img_list):
            bin_idx = chosen_bin_indices[i]
            country, month = bins[bin_idx]
            
            # Structure: Global_Supply_Chain_Simulation/Country/Month/Fruit/Image.png
            target_dir = os.path.join(TARGET_DATASET, country, month, fruit)
            os.makedirs(target_dir, exist_ok=True)
            
            # Unique filename
            filename = f"{fruit}_{'Fresh' if is_fresh else 'Rotten'}_{count:06d}{os.path.splitext(img_path)[1]}"
            shutil.copy2(img_path, os.path.join(target_dir, filename))
            count += 1
            
            if count % 1000 == 0:
                print(f"  - Simulated {count} shipments...")

    print(f"\nSimulation Complete. {count} total shipments generated across {len(COUNTRIES)} countries.")

if __name__ == "__main__":
    main()
