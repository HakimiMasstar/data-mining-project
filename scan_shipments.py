import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import pandas as pd
import json
import os
import glob
from tqdm import tqdm

# Config
SIMULATION_DIR = 'Global_Supply_Chain_Simulation'
MODEL_PATH = 'quality_inspector.pth'
INDICES_PATH = 'class_indices.json'
OUTPUT_CSV = 'shipment_audit_log.csv'
IMG_SIZE = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class InspectorCNN(nn.Module):
    def __init__(self, num_classes):
        super(InspectorCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * (IMG_SIZE // 8) * (IMG_SIZE // 8), 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def load_resources():
    with open(INDICES_PATH, 'r') as f:
        class_indices = json.load(f)
    idx_to_class = {v: k for k, v in class_indices.items()}
    
    model = InspectorCNN(len(class_indices)).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model, idx_to_class

def scan():
    if not os.path.exists(SIMULATION_DIR):
        print("Simulation not found. Run generate_supply_chain.py first.")
        return

    print("Loading Quality Inspector...")
    model, idx_to_class = load_resources()
    
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    
    print("Scanning Global Supply Chain...")
    records = []
    
    # Structure: Global_Supply_Chain_Simulation/Country/Month/Fruit/Image.png
    # Use glob for easier traversal
    search_path = os.path.join(SIMULATION_DIR, '*', '*', '*', '*')
    files = glob.glob(search_path)
    img_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"Found {len(img_files)} shipments. Processing...")
    
    batch_size = 32
    # Process in batches for speed
    for i in tqdm(range(0, len(img_files), batch_size)):
        batch_files = img_files[i : i+batch_size]
        batch_tensors = []
        valid_files = []
        
        for f in batch_files:
            try:
                img = Image.open(f).convert('RGB')
                batch_tensors.append(transform(img))
                valid_files.append(f)
            except:
                pass
                
        if not batch_tensors: continue
        
        batch_stack = torch.stack(batch_tensors).to(device)
        
        with torch.no_grad():
            outputs = model(batch_stack)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidences, preds = torch.max(probs, 1)
            
        for idx, fpath in enumerate(valid_files):
            pred_class = idx_to_class[preds[idx].item()]
            confidence = confidences[idx].item()
            
            # Parse Path
            # .../Country/Month/Fruit/Image.png
            parts = fpath.split(os.sep)
            country = parts[-4]
            month = parts[-3]
            fruit = parts[-2]
            
            status = 'Rotten' if 'rotten' in pred_class.lower() else 'Fresh'
            
            records.append({
                'Shipment_ID': os.path.basename(fpath),
                'Country': country,
                'Month': month,
                'Fruit': fruit,
                'Predicted_Status': status,
                'Raw_Class': pred_class,
                'Confidence': confidence,
                'File_Path': fpath
            })
            
    print(f"Saving Audit Log to {OUTPUT_CSV}...")
    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_CSV, index=False)
    print("Scan Complete.")

if __name__ == "__main__":
    scan()
