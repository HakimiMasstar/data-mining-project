# Presentation: Global Fruit Supply Chain Intelligence
## 10-Slide Summary

---

### **Slide 1: Title Slide**
- **Project Name:** Global Fruit Supply Chain Intelligence
- **Objective:** Optimizing Quality and Logistics through Data Mining
- **Date:** February 16, 2026

---

### **Slide 2: The Problem Statement**
- Global fruit supply chains face significant waste due to spoilage.
- Lack of real-time visibility into fruit quality at scale.
- Need for automated auditing to identify regional and seasonal risks.

---

### **Slide 3: Project Architecture (The Pipeline)**
1. **Simulation:** Generating 13k shipments across 3 countries.
2. **Computer Vision:** Training a CNN for quality detection.
3. **Data Mining:** Scanning shipments to build an audit database.
4. **BI Analytics:** Visualizing spoilage and seasonality.

---

### **Slide 4: Data Preparation (CRISP-DM)**
- **Source:** 12,000+ labeled images (Apple, Banana, Orange).
- **Process:** Distributed images into a folder structure representing "USA," "Brazil," and "India" over 12 months.
- **Goal:** Create a "ground truth" simulation for auditing.

---

### **Slide 5: The AI Inspector (CNN Model)**
- **Architecture:** 3-layer CNN with ReLU activation and MaxPool.
- **Input:** 128x128 RGB images.
- **Outcome:** Trained to distinguish between "Fresh" and "Rotten" fruit.

---

### **Slide 6: Model Evaluation Results**
- **Final Accuracy:** 96.41%
- **Loss:** 0.1037
- **Inference:** High confidence in identifying rot, even in varied lighting and fruit types.

---

### **Slide 7: Global Spoilage Overview**
- **India:** 60.7% Spoilage Rate
- **Brazil:** 56.3% Spoilage Rate
- **USA:** 49.2% Spoilage Rate
- *Insight:* Developing nations show higher baseline spoilage due to logistics infrastructure gaps.

---

### **Slide 8: Seasonality & Anomalies**
- **India Peak:** May spoilage reached 91% (Logistics Strike).
- **USA Peak:** July spoilage (Heatwave/Summer demand).
- **Brazil Peak:** November–December (Southern Summer).

---

### **Slide 9: Business Impact**
- **Waste Reduction:** Automated detection allows for faster redirection of at-risk shipments.
- **Cost Savings:** Identified specific months and regions where cold-chain investment is critical.
- **Scalability:** The model can be applied to other produce categories.

---

### **Slide 10: Final Recommendations**
- Implement AI-driven auditing at the port of entry.
- Dynamic routing: Divert shipments during identified high-risk seasonal windows.
- Infrastructure: Target "May in India" for immediate cold-chain upgrades.

---
*End of Presentation*
