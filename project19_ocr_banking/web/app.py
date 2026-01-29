"""
Streamlit Web Interface for OCR Banking
"""
import streamlit as st
import cv2
import numpy as np
from pathlib import Path
import sys
from PIL import Image
import io
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ocr_pipeline import OCRPipeline

# Page config
st.set_page_config(
    page_title="OCR Banking - MB Bank",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #424242;
        margin-top: 2rem;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1E88E5;
    }
    .success-box {
        background-color: #E8F5E9;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #4CAF50;
    }
    .field-label {
        font-weight: bold;
        color: #1565C0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_pipeline():
    """Load OCR pipeline (cached)"""
    return OCRPipeline(
        detector_type='craft',
        recognizer_type='vietocr',
        device='cpu',
        use_postprocessing=True
    )


def process_image(image, pipeline):
    """Process image with OCR pipeline"""
    # Convert PIL to numpy
    image_np = np.array(image)
    
    # Convert RGB to BGR (OpenCV format)
    if len(image_np.shape) == 3:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    # Process
    result = pipeline.process_image_array(image_np, source="uploaded")
    
    return result


def display_results(result):
    """Display OCR results"""
    # Status
    if result['status'] == 'success':
        st.markdown('<div class="success-box">✅ OCR Processing Successful</div>', unsafe_allow_html=True)
    else:
        st.error(f"❌ Processing failed: {result.get('error', 'Unknown error')}")
        return
    
    # Document Type
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<p class="field-label">Document Type</p>', unsafe_allow_html=True)
        doc_type_display = {
            'cccd': '🪪 Căn cước công dân (CCCD)',
            'cmnd': '🪪 Chứng minh nhân dân (CMND)',
            'bank_statement': '🏦 Sao kê ngân hàng',
            'loan_document': '📄 Hợp đồng vay',
            'unknown': '❓ Không xác định'
        }
        st.info(doc_type_display.get(result['document_type'], result['document_type']))
    
    with col2:
        st.markdown('<p class="field-label">Confidence Score</p>', unsafe_allow_html=True)
        confidence = result['confidence']
        color = "green" if confidence > 0.8 else "orange" if confidence > 0.5 else "red"
        st.markdown(f"<h3 style='color: {color};'>{confidence:.1%}</h3>", unsafe_allow_html=True)
    
    # Raw Text
    st.markdown('<p class="sub-header">📝 Recognized Text</p>', unsafe_allow_html=True)
    with st.expander("View raw text", expanded=False):
        st.text(result['raw_text'])
    
    # Extracted Fields
    st.markdown('<p class="sub-header">📋 Extracted Information</p>', unsafe_allow_html=True)
    
    if result['extracted_fields']:
        # Display in a nice table format
        fields_data = []
        
        for field, value in result['extracted_fields'].items():
            field_display = field.replace('_', ' ').title()
            
            if isinstance(value, list):
                value_display = ', '.join(str(v) for v in value) if value else '-'
            elif isinstance(value, dict):
                value_display = json.dumps(value, indent=2)
            else:
                value_display = str(value) if value else '-'
            
            fields_data.append({
                'Field': field_display,
                'Value': value_display
            })
        
        st.table(fields_data)
    else:
        st.info("No structured fields extracted")
    
    # Text Regions
    st.markdown('<p class="sub-header">🔍 Text Regions</p>', unsafe_allow_html=True)
    
    with st.expander(f"View {len(result['text_regions'])} text regions", expanded=False):
        for i, region in enumerate(result['text_regions'][:10]):  # Show first 10
            st.markdown(f"**Region {i+1}:** {region['text']}")


def main():
    # Header
    st.markdown('<h1 class="main-header">🏦 OCR Banking System</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #757575;">MB Bank - eKYC Document Recognition</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        
        st.markdown("#### Supported Documents")
        st.markdown("""
        - 🪪 CCCD (Căn cước công dân)
        - 🪪 CMND (Chứng minh nhân dân)
        - 🏦 Bank Statements
        - 📄 Loan Documents
        """)
        
        st.markdown("---")
        
        st.markdown("#### 📖 Instructions")
        st.markdown("""
        1. Upload document image
        2. Wait for processing
        3. Review extracted information
        4. Export results if needed
        """)
        
        st.markdown("---")
        
        st.markdown("#### 💡 Tips")
        st.markdown("""
        - Use high-quality images
        - Ensure good lighting
        - Avoid shadows or glare
        - Keep document flat
        """)
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📤 Upload", "📊 Batch Process", "ℹ️ About"])
    
    with tab1:
        st.markdown('<div class="info-box">Upload a document image for OCR processing</div>', unsafe_allow_html=True)
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
            help="Supported formats: JPG, JPEG, PNG, BMP, TIFF"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("#### 📷 Uploaded Image")
                image = Image.open(uploaded_file)
                st.image(image, use_container_width=True)
            
            with col2:
                st.markdown("#### 🔄 Processing...")
                
                with st.spinner("Loading OCR pipeline..."):
                    pipeline = load_pipeline()
                
                with st.spinner("Detecting and recognizing text..."):
                    result = process_image(image, pipeline)
                
                st.success("✅ Processing complete!")
            
            # Display results
            st.markdown("---")
            display_results(result)
            
            # Download results
            st.markdown("---")
            st.markdown("#### 💾 Export Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # JSON export
                json_str = json.dumps(result, indent=2, ensure_ascii=False)
                st.download_button(
                    label="📥 Download JSON",
                    data=json_str,
                    file_name=f"ocr_result_{uploaded_file.name}.json",
                    mime="application/json"
                )
            
            with col2:
                # Text export
                text_export = f"""OCR Results
====================
Document Type: {result['document_type']}
Confidence: {result['confidence']:.1%}

Raw Text:
{result['raw_text']}

Extracted Fields:
{json.dumps(result['extracted_fields'], indent=2, ensure_ascii=False)}
"""
                st.download_button(
                    label="📥 Download TXT",
                    data=text_export,
                    file_name=f"ocr_result_{uploaded_file.name}.txt",
                    mime="text/plain"
                )
    
    with tab2:
        st.markdown('<div class="info-box">Process multiple documents at once</div>', unsafe_allow_html=True)
        
        uploaded_files = st.file_uploader(
            "Choose multiple image files",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
            accept_multiple_files=True,
            help="Upload up to 10 images"
        )
        
        if uploaded_files:
            if len(uploaded_files) > 10:
                st.error("⚠️ Maximum 10 files allowed per batch")
            else:
                if st.button("🚀 Process All", type="primary"):
                    pipeline = load_pipeline()
                    
                    progress_bar = st.progress(0)
                    results_container = st.container()
                    
                    all_results = []
                    
                    for i, file in enumerate(uploaded_files):
                        st.markdown(f"**Processing: {file.name}**")
                        
                        image = Image.open(file)
                        result = process_image(image, pipeline)
                        all_results.append(result)
                        
                        progress_bar.progress((i + 1) / len(uploaded_files))
                    
                    st.success(f"✅ Processed {len(uploaded_files)} documents")
                    
                    # Display summary
                    with results_container:
                        st.markdown("### 📊 Batch Results Summary")
                        
                        summary_data = []
                        for i, (file, result) in enumerate(zip(uploaded_files, all_results)):
                            summary_data.append({
                                'File': file.name,
                                'Type': result['document_type'],
                                'Status': result['status'],
                                'Confidence': f"{result['confidence']:.1%}"
                            })
                        
                        st.table(summary_data)
                        
                        # Download all results
                        all_json = json.dumps(all_results, indent=2, ensure_ascii=False)
                        st.download_button(
                            label="📥 Download All Results (JSON)",
                            data=all_json,
                            file_name="batch_ocr_results.json",
                            mime="application/json"
                        )
    
    with tab3:
        st.markdown("### 📚 About OCR Banking System")
        
        st.markdown("""
        This system provides automated document recognition and information extraction 
        for banking documents, specifically designed for MB Bank's eKYC process.
        
        #### 🎯 Features
        - **Text Detection**: CRAFT & DBNet algorithms
        - **Text Recognition**: VietOCR (Vietnamese-optimized)
        - **Information Extraction**: Regex-based field extraction
        - **Post-processing**: Fuzzy matching and validation
        
        #### 📄 Supported Documents
        - **CCCD**: Căn cước công dân (12-digit ID)
        - **CMND**: Chứng minh nhân dân (9-digit ID)
        - **Bank Statements**: Transaction history and balances
        - **Loan Documents**: Contract information
        
        #### 🔧 Technology Stack
        - **Detection**: CRAFT, DBNet, OpenCV
        - **Recognition**: VietOCR, PaddleOCR, EasyOCR
        - **Backend**: FastAPI, Python
        - **Frontend**: Streamlit
        - **Deployment**: Docker
        
        #### 📞 Contact
        For issues or questions, contact the development team.
        """)
        
        st.markdown("---")
        st.markdown("**Version:** 1.0.0 | **Last Updated:** 2025")


if __name__ == "__main__":
    main()
