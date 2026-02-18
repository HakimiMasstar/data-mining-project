# Global Fruit Quality Inspector - Streamlit App

An interactive web application for testing the trained fruit quality classification model.

## 🚀 Live Demo

**Coming Soon:** Deployed on Streamlit Cloud

## 📋 Prerequisites

Before running locally, ensure you have:
- Python 3.8+
- Trained model file: `quality_inspector.pth`
- Class indices file: `class_indices.json`

## 🛠️ Installation

### Option 1: Run Locally

```bash
# Clone or navigate to the project directory
cd data-mining-project

# Install dependencies
pip install -r requirements_streamlit.txt

# Run the app
streamlit run streamlit_app.py
```

### Option 2: Deploy to Streamlit Cloud

1. **Upload Model Files to Google Drive**
   - Upload `quality_inspector.pth` to Google Drive
   - Upload `class_indices.json` to Google Drive
   - Get shareable links and extract file IDs
   - Update the `MODEL_URL` and `INDICES_URL` in `streamlit_app.py`

2. **Create GitHub Repository**
   ```bash
   git init
   git add streamlit_app.py requirements_streamlit.txt
   git commit -m "Initial Streamlit app"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/fruit-quality-inspector.git
   git push -u origin main
   ```

3. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Set main file path: `streamlit_app.py`
   - Click "Deploy"

## 📁 File Structure

```
data-mining-project/
├── streamlit_app.py           # Main Streamlit application
├── requirements_streamlit.txt # Python dependencies
├── quality_inspector.pth      # Trained model (for local use)
├── class_indices.json         # Class mappings (for local use)
└── README_STREAMLIT.md        # This file
```

## 🎯 Features

- 📤 **Drag & Drop Upload**: Easy image upload interface
- 🔍 **AI Prediction**: Real-time classification with 99.65% accuracy
- 📊 **Confidence Score**: Visual progress bar showing model certainty
- 🎨 **Color-coded Results**: Green for Fresh, Red for Rotten
- ℹ️ **Smart Interpretation**: Guidance based on confidence levels
- 📱 **Responsive Design**: Works on desktop and mobile
- 🖼️ **Multiple Formats**: Supports JPG, JPEG, and PNG

## 🧪 How to Use

1. **Upload an Image**: Click the upload area or drag & drop a fruit image
2. **Wait for Analysis**: AI processes the image automatically
3. **View Results**: See prediction, fruit type, and confidence score
4. **Interpret Confidence**:
   - 🟢 **≥90%**: High confidence - very reliable
   - 🟡 **70-89%**: Good confidence - suitable for automation
   - 🔴 **<70%**: Low confidence - consider manual review

## 🍎 Supported Fruits

- **Apples** (Fresh & Rotten)
- **Bananas** (Fresh & Rotten)
- **Oranges** (Fresh & Rotten)

## 🔧 Troubleshooting

### Model Download Issues
If the model fails to download from Google Drive:
1. Check that file IDs are correct in `streamlit_app.py`
2. Ensure files are publicly accessible
3. For large models, consider using Git LFS or direct hosting

### Local Deployment
If running locally without Google Drive:
```python
# In streamlit_app.py, replace download_and_load_model() with:
@st.cache_resource
def load_model_local():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    with open('class_indices.json', 'r') as f:
        class_indices = json.load(f)
    
    model = InspectorCNN(len(class_indices)).to(device)
    model.load_state_dict(torch.load('quality_inspector.pth', map_location=device))
    model.eval()
    
    return model, class_indices
```

## 📊 Model Performance

- **Architecture**: Convolutional Neural Network (CNN)
- **Accuracy**: 99.65% on validation data
- **Training Data**: 10,901 labeled images
- **Classes**: 6 (Fresh Apples, Fresh Bananas, Fresh Oranges, Rotten Apples, Rotten Bananas, Rotten Oranges)

## 📝 Notes

- The model was trained on images with specific characteristics
- Best results with clear, well-lit photos of single fruits
- Performance may vary with unusual angles, lighting, or multiple fruits

## 🤝 Support

For issues or questions:
- Check the main project documentation
- Review the training notebook for model details
- Ensure all dependencies are correctly installed

## 📄 License

This project is part of the Global Fruit Supply Chain Intelligence initiative.

---

**Built with ❤️ using Streamlit and PyTorch**
