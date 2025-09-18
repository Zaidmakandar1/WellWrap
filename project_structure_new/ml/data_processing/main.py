"""
AI-Powered Medical Report Simplifier
Main Streamlit application for processing medical reports and generating patient-friendly explanations.
"""

import streamlit as st
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from report_processor import ReportProcessor
from model_handler import ModelHandler
from simplifier import MedicalSimplifier

# Page configuration
st.set_page_config(
    page_title="Medical Report Simplifier",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.simplification-box {
    background-color: #f0f8ff;
    padding: 1rem;
    border-radius: 10px;
    border-left: 5px solid #1f77b4;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

def initialize_components():
    """Initialize the ML components if not already in session state"""
    if 'model_handler' not in st.session_state:
        with st.spinner("Loading AI models... This may take a moment."):
            st.session_state.model_handler = ModelHandler()
    
    if 'report_processor' not in st.session_state:
        st.session_state.report_processor = ReportProcessor(model_handler=st.session_state.model_handler)
    
    if 'simplifier' not in st.session_state:
        st.session_state.simplifier = MedicalSimplifier(st.session_state.model_handler)

def main():
    """Main application function"""
    
    # Header
    st.markdown('<h1 class="main-header">🏥 AI-Powered Medical Report Simplifier</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    **Transform complex medical reports into patient-friendly explanations**
    
    Upload your medical test report (PDF or text) and get easy-to-understand explanations 
    of your results, including what they mean for your health.
    """)
    
    # Initialize components
    initialize_components()
    
    # Sidebar
    with st.sidebar:
        st.header("📋 How it works")
        st.markdown("""
        1. **Upload** your medical report (PDF or text)
        2. **Process** with AI to extract key information
        3. **Get** simple explanations in plain English
        4. **Understand** what your results mean
        """)
        
        st.header("📊 Supported Reports")
        st.markdown("""
        - Complete Blood Count (CBC)
        - Lipid Profile
        - Basic Metabolic Panel
        - Liver Function Tests
        - Thyroid Function Tests
        """)
        
        st.header("⚠️ Important Note")
        st.warning("This tool is for educational purposes only. Always consult with your healthcare provider for medical advice.")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Report")
        
        # Input method selection
        input_method = st.radio(
            "Choose input method:",
            ["Upload PDF file", "Paste text directly"]
        )
        
        extracted_text = ""
        
        if input_method == "Upload PDF file":
            uploaded_file = st.file_uploader(
                "Choose a PDF file", 
                type="pdf",
                help="Upload your medical report in PDF format"
            )
            
            if uploaded_file is not None:
                with st.spinner("Extracting text from PDF..."):
                    extracted_text = st.session_state.report_processor.extract_text_from_pdf(uploaded_file)
                
                if extracted_text:
                    st.success("✅ Text extracted successfully!")
                    with st.expander("Preview extracted text"):
                        st.text_area("Extracted content:", extracted_text, height=200)
                else:
                    st.error("❌ Failed to extract text from PDF")
        
        else:
            extracted_text = st.text_area(
                "Paste your medical report text here:",
                height=300,
                placeholder="Paste the content of your medical report here..."
            )
    
    with col2:
        st.header("🔍 Simplified Explanation")
        
        if extracted_text:
            if st.button("🚀 Simplify Report", type="primary"):
                with st.spinner("Analyzing report and generating explanations..."):
                    try:
                        # Process the report
                        processed_data = st.session_state.report_processor.parse_medical_data(extracted_text)
                        
                        # Generate simplifications
                        simplifications = st.session_state.simplifier.simplify_report(processed_data)
                        
                        # Display results
                        if simplifications:
                            st.success("✅ Report analysis complete!")
                            
                            for item in simplifications:
                                st.markdown(f"""
                                <div class="simplification-box">
                                <h4>🔬 {item['test_name']}</h4>
                                <p><strong>Your Result:</strong> {item['value']} {item.get('unit', '')}</p>
                                <p><strong>Normal Range:</strong> {item.get('normal_range', 'Not available')}</p>
                                <p><strong>Simple Explanation:</strong> {item['explanation']}</p>
                                {f"<p><strong>Health Insight:</strong> {item['health_insight']}</p>" if item.get('health_insight') else ""}
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.warning("⚠️ Could not extract recognizable medical data from the report.")
                            
                    except Exception as e:
                        st.error(f"❌ Error processing report: {str(e)}")
                        st.info("This might happen if the report format is not recognized or if there are technical issues.")
        
        else:
            st.info("👆 Please upload a PDF file or paste text to begin analysis")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
    Built with ❤️ for better health understanding | 
    <strong>Always consult your doctor for medical decisions</strong>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
