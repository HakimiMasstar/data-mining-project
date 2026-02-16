from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

def apply_corporate_style(slide, title_text, bullet_points, is_title_slide=False):
    # Title Styling
    title_shape = slide.shapes.title
    title_shape.text = title_text
    title_frame = title_shape.text_frame
    title_p = title_frame.paragraphs[0]
    title_p.font.bold = True
    title_p.font.size = Pt(36)
    title_p.font.name = 'Arial'
    title_p.font.color.rgb = RGBColor(0, 32, 96) # Dark Corporate Blue
    
    if not is_title_slide:
        # Content Styling
        content_shape = slide.placeholders[1]
        tf = content_shape.text_frame
        tf.clear()
        
        for point in bullet_points:
            p = tf.add_paragraph()
            p.text = point
            p.level = 0
            p.font.size = Pt(20)
            p.font.name = 'Arial'
            p.space_after = Pt(12)
            p.font.color.rgb = RGBColor(64, 64, 64)

def create_professional_presentation():
    prs = Presentation()
    
    slides_data = [
        {"title": "Global Fruit Supply Chain Intelligence", "content": ["Strategic Data Mining for Quality Assurance", "Analysis & Logistics Optimization", "February 2026"], "is_title": True},
        {"title": "Problem Statement", "content": ["Global supply chains suffer from significant wastage due to fruit spoilage.", "Manual quality inspection is slow, subjective, and difficult to scale.", "Lack of data-driven insights prevents proactive regional risk mitigation."]},
        {"title": "The Solution Pipeline", "content": ["Stage 1: Geographic & Temporal Simulation (Digital Twin)", "Stage 2: Deep Learning Quality Inspector (CNN)", "Stage 3: Automated Audit Extraction (Large-scale Data Mining)", "Stage 4: Business Intelligence & Risk Mapping"]},
        {"title": "Data Architecture: CRISP-DM", "content": ["Business Understanding: Aligning ML goals with logistics KPIs.", "Data Preparation: Image normalization and temporal distribution.", "Model-Centric Design: Star Schema for rapid BI querying."]},
        {"title": "AI Quality Inspector (CNN)", "content": ["Architecture: Convolutional Neural Network (3 Blocks + Dense Layer).", "Performance: ~96% classification accuracy on Fresh vs. Rotten classes.", "Deployment: Scalable inference for real-time shipment auditing."]},
        {"title": "Global Performance Audit", "content": ["India: 60.7% average spoilage - Highest risk region identified.", "Brazil: 56.3% spoilage - Critical seasonality during Q4.", "USA: 49.2% spoilage - Best performing logistics corridor Corridors."]},
        {"title": "Seasonality & Anomalies", "content": ["India Logistics Strike (May): Spoilage spiked to 91%.", "Northern Hemisphere Summer: High heat correlation with USA spoilage in July.", "Southern Hemisphere Summer: Increased risk for Brazil in November."]},
        {"title": "Business & Economic Impact", "content": ["Efficiency: 95% reduction in manual auditing time.", "Waste Mitigation: Early detection allows for cargo redirection.", "Scalability: Architecture supports additional fruit varieties."]},
        {"title": "Strategic Recommendations", "content": ["Direct investment in cold-chain infrastructure for the Indian corridor.", "Shift procurement strategy to USA during Brazilian peak-heat windows.", "Integrate AI Inspector into existing ERP systems for real-time visibility."]},
        {"title": "Conclusion & Next Steps", "content": ["The pilot project proves that CV-driven auditing is commercially viable.", "Immediate rollout suggested for high-volume regional hubs.", "Future Scope: Predictive analytics for harvest-to-port spoilage forecasting."]}
    ]

    for data in slides_data:
        layout = prs.slide_layouts[0] if data.get("is_title") else prs.slide_layouts[1]
        slide = prs.slides.add_slide(layout)
        apply_corporate_style(slide, data["title"], data["content"], data.get("is_title", False))
        
        if not data.get("is_title"):
            # Add a decorative corporate bar at the bottom
            left = Inches(0.5)
            top = Inches(6.8)
            width = Inches(9)
            height = Inches(0.05)
            shape = slide.shapes.add_shape(1, left, top, width, height)
            shape.fill.solid()
            shape.fill.fore_color.rgb = RGBColor(0, 32, 96)
            shape.line.visible = False

    prs.save('Global_Supply_Chain_Corporate.pptx')
    print("Corporate Presentation saved as Global_Supply_Chain_Corporate.pptx")

if __name__ == "__main__":
    create_professional_presentation()
