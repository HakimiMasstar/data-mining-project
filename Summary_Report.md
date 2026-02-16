# Executive Summary: Global Fruit Supply Chain Intelligence

## 1. Introduction
This report details the implementation of an end-to-end data mining pipeline designed to optimize fruit quality inspection and supply chain logistics for a global operation spanning the USA, Brazil, and India. By leveraging Computer Vision (CV) and Business Intelligence (BI), the project identifies spoilage risks and seasonal anomalies to reduce waste.

## 2. Methodology Overview
The project followed the CRISP-DM framework:
- **Data Engineering (ETL):** Simulated a global supply chain by distributing 12,000+ images across geographic and temporal dimensions.
- **Modeling:** Developed a Convolutional Neural Network (CNN) "Quality Inspector" using PyTorch to classify fruit as "Fresh" or "Rotten."
- **Data Mining:** Audited 13,000+ simulated shipments to extract quality metrics and confidence scores.
- **Analysis:** Aggregated data to visualize spoilage trends and identify high-risk periods.

## 3. Key Findings
- **Model Performance:** The CNN achieved a high classification accuracy of **~96.4%**, ensuring reliable automated auditing.
- **Regional Spoilage Rates:**
    - **India:** Highest average spoilage (~60%), heavily influenced by extreme seasonal heat and logistics disruptions.
    - **Brazil:** ~56% spoilage, showing significant increases during the southern hemisphere's summer months (Oct–Dec).
    - **USA:** Lowest average spoilage (~49%), though risks peak during the domestic summer (July).
- **Anomalies Detected:** A major logistics strike was identified in India during May, resulting in a spike in rotten shipments (>90% spoilage).

## 4. Recommendations
1. **Cold Chain Investment:** Prioritize the enhancement of refrigerated transport in the Indian corridor, specifically targeting the pre-monsoon heat in April–May.
2. **Seasonal Sourcing:** Shift sourcing volumes to the USA during the October–December window to offset the high spoilage rates observed in Brazil during that period.
3. **Automated Auditing:** Deploy the "Quality Inspector" model at regional distribution centers to provide real-time visibility into vendor performance and fruit quality.

---
*End of Report*
