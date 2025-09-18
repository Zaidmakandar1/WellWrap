"""
AI-Powered Medical Report Simplifier
Main Streamlit application for processing medical reports and generating patient-friendly explanations.
"""

import streamlit as st
import sys
import os
from pathlib import Path

# Add ML data processing to path for imports
ml_data_path = Path(__file__).parent.parent / "data_processing"
sys.path.append(str(ml_data_path))

# Use lightweight medical analysis instead of heavy ML dependencies
ML_AVAILABLE = False

# Import the lightweight analyzer from the backend
backend_path = Path(__file__).parent.parent.parent / "backend"
sys.path.append(str(backend_path))

try:
    from advanced_medical_analyzer import AdvancedMedicalAnalyzer
    analyzer = AdvancedMedicalAnalyzer()
    ANALYZER_AVAILABLE = True
    st.success("✅ Medical analyzer loaded successfully!")
except ImportError as e:
    st.error(f"❌ Could not load medical analyzer: {e}")
    ANALYZER_AVAILABLE = False
    analyzer = None

# Page configuration
st.set_page_config(
    page_title="WellWrap",
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
    """Initialize the medical analyzer if not already in session state"""
    if 'analyzer' not in st.session_state:
        if ANALYZER_AVAILABLE:
            with st.spinner("Loading medical analyzer..."):
                st.session_state.analyzer = analyzer
                st.success("✅ Medical analyzer ready!")
        else:
            st.session_state.analyzer = None
            st.error("❌ Medical analyzer not available")

def main():
    """Main application function"""
    
    # Header
    st.markdown('<h1 class="main-header">🏥 WellWrap - AI-Powered Health Analysis</h1>', 
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
                    if st.session_state.analyzer:
                        extracted_text = st.session_state.analyzer.extract_text_from_pdf(uploaded_file)
                    else:
                        extracted_text = "Medical analyzer not available"
                
                if extracted_text and extracted_text != "Medical analyzer not available":
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
                        if st.session_state.analyzer:
                            # Extract medical data
                            test_results = st.session_state.analyzer.extract_medical_data(extracted_text)
                            
                            # Detect diseases
                            disease_risks = st.session_state.analyzer.detect_diseases(test_results)
                            
                            # Generate health summary
                            health_summary = st.session_state.analyzer.generate_health_summary(test_results, disease_risks)
                            
                            # Display results
                            if test_results or disease_risks:
                                st.success("✅ Report analysis complete!")
                                
                                # Health Score
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    score = health_summary.get('health_score', 0)
                                    st.metric("Health Score", f"{score}/100")
                                with col2:
                                    st.metric("Tests Analyzed", len(test_results))
                                with col3:
                                    st.metric("Risk Factors", len(disease_risks))
                                
                                # Test Results
                                if test_results:
                                    st.subheader("🔬 Test Results")
                                    for test in test_results:
                                        status_color = "🟢" if test.status == "normal" else "🔴" if test.status == "high" else "🟡"
                                        st.markdown(f"""
                                        <div class="simplification-box">
                                        <h4>{status_color} {test.test_name}</h4>
                                        <p><strong>Your Result:</strong> {test.value} {test.unit or ''}</p>
                                        <p><strong>Normal Range:</strong> {test.normal_range or 'Not available'}</p>
                                        <p><strong>Status:</strong> {test.status.title()}</p>
                                        </div>
                                        """, unsafe_allow_html=True)
                                
                                # Disease Risks
                                if disease_risks:
                                    st.subheader("⚠️ Health Risks")
                                    for risk in disease_risks:
                                        risk_color = "🔴" if risk.risk_level == "high" else "🟡" if risk.risk_level == "medium" else "🟢"
                                        st.markdown(f"""
                                        <div class="simplification-box">
                                        <h4>{risk_color} {risk.disease_name}</h4>
                                        <p><strong>Risk Level:</strong> {risk.risk_level.title()}</p>
                                        <p><strong>Description:</strong> {risk.description}</p>
                                        <p><strong>Recommendations:</strong></p>
                                        <ul>
                                        {"".join([f"<li>{rec}</li>" for rec in risk.recommendations[:3]])}
                                        </ul>
                                        </div>
                                        """, unsafe_allow_html=True)
                                
                                # Summary
                                st.subheader("📋 Summary")
                                st.info(health_summary.get('summary', 'Analysis completed'))
                                
                            else:
                                st.warning("⚠️ Could not extract recognizable medical data from the report.")
                        else:
                            st.error("❌ Medical analyzer not available")
                            
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
