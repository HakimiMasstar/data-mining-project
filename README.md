# 🍎 Global Fruit Supply Chain Intelligence (Data Mining Project)

This project implements an end-to-end Data Mining pipeline to analyze fruit quality (Apples, Bananas, Oranges) across a simulated global supply chain involving **USA, Brazil, and India**.

## 🏗️ The 4-Stage Pipeline

### 1. Data Engineering (`generate_supply_chain.py`)
Transforms 12,000+ raw images into a structured global simulation.
- **Logic:** Distributes every image to a specific **Country** and **Month** using a weighted probability model based on real-world harvest seasons.
- **Output:** `Global_Supply_Chain_Simulation/` folder structure.

### 2. Machine Learning (`train_quality_inspector.py`)
Trains a Convolutional Neural Network (CNN) using PyTorch.
- **Task:** Computer Vision classification (Fresh vs. Rotten).
- **Output:** `quality_inspector.pth` (The automated inspector).

### 3. Data Mining (`scan_shipments.py`)
Performs a massive automated audit of the global supply chain.
- **Process:** The AI "scans" every image in the simulation and records findings.
- **Output:** `shipment_audit_log.csv` (A detailed database containing Shipment ID, Country, Month, Fruit, AI Prediction, and Confidence).

### 4. Business Intelligence (`supply_chain_dashboard.py`)
Generates 7 interactive visualizations to identify risks and trends.
- **Output:** `Global_Supply_Chain_Dashboard.html`.

---

## 🌍 Agricultural Logic (Seasonality)
The simulation is grounded in real-world facts:
- **USA:** Apple harvest in Autumn (Aug-Nov); quality dips in Summer due to storage aging.
- **Brazil:** Counter-seasonal supply; fresh apples in Q1-Q2 when Northern Hemisphere stock is low.
- **India:** Two distinct Orange harvests (Ambiya & Mrig). 
- **The "Anomaly":** A simulated **Logistics Strike in India (May)** causes a massive spike in Banana rot rates, testing the AI's ability to detect supply chain failures.

---

## 🚀 Usage Instructions

1. **Install Dependencies:**
   ```bash
   pip install torch torchvision pandas numpy plotly tqdm pillow
   ```

2. **Run Pipeline:**
   ```bash
   # Step 1: Forge the Supply Chain
   python generate_supply_chain.py

   # Step 2: Train the AI
   python train_quality_inspector.py

   # Step 3: Mine the Data
   python scan_shipments.py

   # Step 4: Visualize Results
   python supply_chain_dashboard.py
   ```

3. **View Results:**
   Open `Global_Supply_Chain_Dashboard.html` in any web browser to see interactive charts and supply chain insights.
