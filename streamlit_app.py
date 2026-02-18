import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import json
import os

# Try to import gdown, install if not available
try:
    import gdown
except ImportError:
    st.error("⚠️ Installing required dependency: gdown...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
    import gdown
    st.success("✓ gdown installed successfully!")

import base64
from io import BytesIO

# Page configuration
st.set_page_config(
    page_title='Fruit Quality Inspector',
    page_icon='🍎',
    layout='centered',
    initial_sidebar_state='expanded'
)

# Custom CSS styling
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
    }
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    .result-fresh {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 10px 0;
    }
    .result-rotten {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #dc3545;
        margin: 10px 0;
    }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    .example-img {
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.2s;
        cursor: pointer;
    }
    .example-img:hover {
        transform: scale(1.05);
    }
    div[data-testid="stHorizontalBlock"] > div:first-child {
        cursor: pointer;
    }
    </style>
""", unsafe_allow_html=True)

# Model architecture (must match training)
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
            nn.Linear(128 * 16 * 16, 512),
            nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

@st.cache_resource
def download_and_load_model():
    """Download model from Google Drive and load it."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Google Drive file IDs
    MODEL_URL = "https://drive.google.com/uc?id=1G9-9wrNSwq7djgkAX5Od13hSv1IDUCss"
    INDICES_URL = "https://drive.google.com/uc?id=1OfpMY-S0zr8mpISRb7Mwmeaw8GKNzY1R"
    
    model_path = "quality_inspector.pth"
    indices_path = "class_indices.json"
    
    # Download files if they don't exist
    if not os.path.exists(model_path):
        with st.spinner('📥 Downloading model... This may take a minute.'):
            try:
                gdown.download(MODEL_URL, model_path, quiet=False)
            except:
                st.error("❌ Failed to download model. Please check the file ID.")
                return None, None
    
    if not os.path.exists(indices_path):
        with st.spinner('📥 Downloading class indices...'):
            try:
                gdown.download(INDICES_URL, indices_path, quiet=False)
            except:
                st.error("❌ Failed to download class indices. Please check the file ID.")
                return None, None
    
    # Load class indices
    with open(indices_path, 'r') as f:
        class_indices = json.load(f)
    
    # Initialize model
    model = InspectorCNN(len(class_indices)).to(device)
    
    # Load trained weights
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None
    
    return model, class_indices

def predict_image(image, model, class_indices):
    """Predict the quality of a fruit image."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    
    # Prepare image
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    # Prediction
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    # Get class name
    idx_to_class = {v: k for k, v in class_indices.items()}
    pred_class = idx_to_class[predicted.item()]
    
    # Determine if fresh or rotten
    is_rotten = 'rotten' in pred_class.lower()
    
    return {
        'class': pred_class,
        'is_rotten': is_rotten,
        'confidence': confidence.item() * 100,
        'fruit_type': pred_class.replace('fresh', '').replace('rotten', '').strip()
    }

def load_example_image(image_name):
    """Load an example image from the test_images folder."""
    test_images_dir = "test_images"
    image_path = os.path.join(test_images_dir, image_name)
    
    if os.path.exists(image_path):
        return Image.open(image_path).convert('RGB')
    return None

def display_results(image, result):
    """Display prediction results."""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader('📸 Image')
        st.image(image, use_container_width=True)
    
    with col2:
        st.subheader('🤖 AI Prediction')
        
        # Display result with styling
        if result['is_rotten']:
            st.markdown(f"""
                <div class='result-rotten'>
                    <h3 style='color: #dc3545; margin: 0;'>⚠️ ROTTEN</h3>
                    <p style='font-size: 1.2em; margin: 10px 0;'>
                        <strong>Fruit:</strong> {result['fruit_type'].title()}<br>
                        <strong>Status:</strong> Not suitable for sale
                    </p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class='result-fresh'>
                    <h3 style='color: #28a745; margin: 0;'>✅ FRESH</h3>
                    <p style='font-size: 1.2em; margin: 10px 0;'>
                        <strong>Fruit:</strong> {result['fruit_type'].title()}<br>
                        <strong>Status:</strong> Quality approved
                    </p>
                </div>
            """, unsafe_allow_html=True)
        
        # Confidence score
        st.subheader('📊 Confidence Score')
        st.progress(result['confidence'] / 100)
        st.markdown(f"""
            <div class='metric-card'>
                <h2 style='color: #667eea; margin: 0;'>{result['confidence']:.1f}%</h2>
                <p style='color: #666;'>Model Confidence</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Interpretation
        if result['confidence'] >= 90:
            st.success('✓ High confidence prediction - very reliable')
        elif result['confidence'] >= 70:
            st.info('ℹ️ Good confidence - suitable for automatic processing')
        else:
            st.warning('⚠️ Low confidence - consider manual review')

def main():
    # Header
    st.title('🍎 Fruit Quality Inspector')
    st.subheader('AI-Powered Freshness Detection')
    st.markdown('---')
    
    # Sidebar info
    with st.sidebar:
        st.header('📊 Model Information')
        st.info('\n\n'.join([
            '**Architecture:** CNN (InspectorCNN)',
            '**Accuracy:** 99.65%',
            '**Classes:** 6 (Fresh/Rotten)',
            '**Fruits:** Apples, Bananas, Oranges',
            '',
            '**Supported Formats:**',
            '- JPG/JPEG',
            '- PNG'
        ]))
        
        st.header('🎯 How to Use')
        st.markdown("""
        1. Upload a fruit image below
        2. AI will analyze the quality
        3. View prediction and confidence
        4. Use results for QC decisions
        """)
        
        st.header('💡 Tips for Best Results')
        st.markdown("""
        - Use clear, well-lit images
        - Focus on a single fruit
        - Avoid blurry or dark photos
        - Center the fruit in the frame
        """)
    
    # Load model
    model, class_indices = download_and_load_model()
    
    if model is None:
        st.error("⚠️ Model could not be loaded. Please refresh the page.")
        st.stop()
    
    # Initialize session state for selected example
    if 'selected_example' not in st.session_state:
        st.session_state.selected_example = None
    
    # File upload
    st.header('📤 Upload Image')
    uploaded_file = st.file_uploader(
        'Choose a fruit image (Apple, Banana, or Orange)...',
        type=['jpg', 'jpeg', 'png'],
        help='Upload a clear image of a single fruit for best results'
    )
    
    # Example images section
    st.markdown("### 🖼️ Or click an example image to test:")
    
    # Define example images with labels
    examples = [
        ("fresh_apple.png", "🍎 Fresh Apple", "green"),
        ("fresh_banana.png", "🍌 Fresh Banana", "green"),
        ("fresh_orange.png", "🍊 Fresh Orange", "green"),
        ("rotten_apple.png", "🍎 Rotten Apple", "red"),
        ("rotten_banana.png", "🍌 Rotten Banana", "red"),
        ("rotten_orange.png", "🍊 Rotten Orange", "red")
    ]
    
    # Display example buttons in a grid
    st.markdown("<p style='text-align: center; color: #666; margin-bottom: 20px;'>Click any button to test with that example</p>", unsafe_allow_html=True)
    cols = st.columns(3)
    for idx, (img_name, label, color) in enumerate(examples):
        with cols[idx % 3]:
            example_img = load_example_image(img_name)
            if example_img:
                # Show only the button (no image)
                btn_type = "primary" if color == "green" else "secondary"
                if st.button(label, key=f'btn_{idx}', type=btn_type, use_container_width=True):
                    st.session_state.selected_example = example_img
                    st.session_state.selected_label = label
                    st.rerun()
    
    # Process uploaded file
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        with st.spinner('🔍 Analyzing image with AI...'):
            result = predict_image(image, model, class_indices)
        display_results(image, result)
    
    # Process selected example
    elif st.session_state.selected_example is not None:
        st.markdown(f"<h4 style='text-align: center; color: #667eea;'>Testing with: {st.session_state.selected_label}</h4>", unsafe_allow_html=True)
        with st.spinner('🔍 Analyzing example image with AI...'):
            result = predict_image(st.session_state.selected_example, model, class_indices)
        display_results(st.session_state.selected_example, result)
        
        # Clear selection button
        if st.button('🔄 Test Another Image'):
            st.session_state.selected_example = None
            st.rerun()
    
    else:
        # Display placeholder when no image uploaded
        st.info("👆 Upload an image above or click an example to get started!")
        
        # Show model capabilities
        st.markdown("---")
        st.subheader("🎯 What This App Can Detect")
        
        cap_col1, cap_col2 = st.columns(2)
        with cap_col1:
            st.markdown("""
            **Fresh Fruits:**
            - ✅ Fresh Apples
            - ✅ Fresh Bananas  
            - ✅ Fresh Oranges
            """)
        with cap_col2:
            st.markdown("""
            **Rotten/Spoiled:**
            - ❌ Rotten Apples
            - ❌ Rotten Bananas
            - ❌ Rotten Oranges
            """)
    
    # Footer
    st.markdown('---')
    st.markdown("""
        <div style='text-align: center; color: #666; padding: 20px;'>
            <p><strong>Global Fruit Supply Chain Intelligence</strong></p>
            <p>Powered by Deep Learning | 99.65% Accuracy | Built with Streamlit</p>
            <p style='font-size: 0.9em; color: #999;'>Upload your fruit images and get instant quality assessment</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
