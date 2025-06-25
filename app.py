"""
Streamlit Web Application for Automated Metadata Generation
Main application file that integrates all modules into a user-friendly interface
"""

import streamlit as st
import pandas as pd
import json
import io
import time
import tempfile
import shutil
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Import your actual processing modules from the src folder
import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.append(str(src_path))

from document_processor import DocumentProcessor
from nlp_analyzer import NLPAnalyzer  
from metadata_generator import MetadataGenerator

# Note: Your modules are located in the src/ folder:
# - src/document_processor.py (contains DocumentProcessor class)
# - src/nlp_analyzer.py (contains NLPAnalyzer class)  
# - src/metadata_generator.py (contains MetadataGenerator class)

# Initialize components
@st.cache_resource
def initialize_components():
    """Initialize all processing components"""
    processor = DocumentProcessor()    # Your real DocumentProcessor class
    analyzer = NLPAnalyzer()          # Your real NLPAnalyzer class  
    generator = MetadataGenerator()    # Your real MetadataGenerator class
    return processor, analyzer, generator

def create_analytics_dashboard(metadata_list):
    """Create analytics dashboard for processed documents"""
    if not metadata_list:
        return
    
    st.subheader("📊 Document Analytics Dashboard")
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Documents", len(metadata_list))
    
    with col2:
        avg_words = sum(m.word_count for m in metadata_list) / len(metadata_list)
        st.metric("Avg Word Count", f"{int(avg_words):,}")
    
    with col3:
        avg_confidence = sum(m.confidence_score for m in metadata_list) / len(metadata_list)
        st.metric("Avg Confidence", f"{avg_confidence:.2f}")
    
    with col4:
        total_reading_time = sum(m.reading_time_minutes for m in metadata_list)
        st.metric("Total Reading Time", f"{total_reading_time} min")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Document types pie chart
        doc_types = [m.document_type for m in metadata_list]
        type_counts = pd.Series(doc_types).value_counts()
        
        fig = px.pie(values=type_counts.values, names=type_counts.index, 
                    title="Document Types Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Sentiment distribution
        sentiments = [m.sentiment for m in metadata_list]
        sentiment_counts = pd.Series(sentiments).value_counts()
        
        fig = px.bar(x=sentiment_counts.index, y=sentiment_counts.values,
                    title="Sentiment Distribution", 
                    color=sentiment_counts.values,
                    color_continuous_scale="RdYlBu")
        st.plotly_chart(fig, use_container_width=True)

def display_metadata_card(metadata, index):
    """Display a metadata card with key information"""
    with st.expander(f"📄 {metadata.filename} - {metadata.title}", expanded=False):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.write("**Summary:**", metadata.summary)
            st.write("**Topics:**", ", ".join(metadata.primary_topics))
            st.write("**Keywords:**", ", ".join(metadata.keywords[:5]))
            
            if hasattr(metadata, 'entities') and metadata.entities:
                st.write("**Entities:**")
                for entity_type, entities in metadata.entities.items():
                    if entities:
                        st.write(f"  - {entity_type}: {', '.join(entities[:3])}")
        
        with col2:
            st.metric("Word Count", f"{metadata.word_count:,}")
            st.metric("Confidence", f"{metadata.confidence_score:.2f}")
            st.metric("Reading Time", f"{metadata.reading_time_minutes} min")
            st.metric("Document Type", metadata.document_type)

def main():
    """Main Streamlit application"""
    
    # Page configuration
    st.set_page_config(
        page_title="Automated Metadata Generator",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #2E86AB;
        padding: 1rem 0;
        border-bottom: 2px solid #f0f2f6;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .info-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2E86AB;
        margin: 1rem 0;
    }
    
    .success-box {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 Automated Metadata Generation System</h1>
        <p>Upload documents and get intelligent metadata automatically generated using advanced NLP analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize components
    processor, analyzer, generator = initialize_components()
    
    # Initialize session state
    if 'processed_documents' not in st.session_state:
        st.session_state.processed_documents = []
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/300x150/2E86AB/FFFFFF?text=Metadata+Generator", 
                caption="Intelligent Document Analysis")
        
        st.markdown("### 🚀 Features")
        st.markdown("""
        - **Multi-format Support**: PDF, DOCX, TXT
        - **OCR Capability**: Scanned documents
        - **Smart Analysis**: Topics, entities, sentiment
        - **Structured Output**: JSON, XML, YAML
        - **Batch Processing**: Multiple files
        """)
        
        st.markdown("### 📊 Analytics")
        if st.session_state.processed_documents:
            total_docs = len(st.session_state.processed_documents)
            total_words = sum(doc.word_count for doc in st.session_state.processed_documents)
            st.metric("Processed Documents", total_docs)
            st.metric("Total Words Analyzed", f"{total_words:,}")
        else:
            st.info("Upload documents to see analytics")
        
        # Settings
        st.markdown("### ⚙️ Settings")
        export_format = st.selectbox("Export Format", ["JSON", "XML", "YAML"])
        include_confidence = st.checkbox("Include Confidence Scores", value=True)
        batch_mode = st.checkbox("Batch Processing Mode", value=False)
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload & Process", "📊 Analytics", "📋 Results", "⚙️ Settings"])
    
    with tab1:
        st.header("Document Upload and Processing")
        
        # File upload section
        st.subheader("1. Upload Documents")
        uploaded_files = st.file_uploader(
            "Choose files to process",
            type=['pdf', 'docx', 'txt', 'doc'],
            accept_multiple_files=batch_mode,
            help="Supported formats: PDF, DOCX, TXT, DOC"
        )
        
        # Custom tags input
        st.subheader("2. Custom Configuration")
        col1, col2 = st.columns(2)
        
        with col1:
            custom_tags = st.text_input(
                "Custom Tags (comma-separated)",
                placeholder="e.g., important, quarterly-report, 2024"
            )
        
        with col2:
            processing_options = st.multiselect(
                "Processing Options",
                ["Extract Images", "Deep Topic Analysis", "Entity Linking", "OCR Processing"],
                default=["Deep Topic Analysis"]
            )
        
        # Process button
        if st.button("🚀 Process Documents", type="primary", use_container_width=True):
            if uploaded_files:
                files_to_process = uploaded_files if isinstance(uploaded_files, list) else [uploaded_files]
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, uploaded_file in enumerate(files_to_process):
                    status_text.text(f"Processing {uploaded_file.name}...")
                    
                    try:
                        # Save uploaded file temporarily
                        temp_dir = Path("temp_uploads")
                        temp_dir.mkdir(exist_ok=True)
                        
                        temp_file_path = temp_dir / uploaded_file.name
                        with open(temp_file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # Process the saved file
                        doc_result = processor.process_document(str(temp_file_path))
                        nlp_result = analyzer.analyze_document(doc_result['processed_text'], uploaded_file.name)
                        
                        # Clean up temporary file
                        temp_file_path.unlink()
                        
                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                        continue
                    
                    # Generate metadata
                    tags_list = [tag.strip() for tag in custom_tags.split(',')] if custom_tags else None
                    metadata = generator.generate_metadata(doc_result, nlp_result, tags_list)
                    
                    # Store in session state
                    st.session_state.processed_documents.append(metadata)
                    
                    # Update progress
                    progress_bar.progress((i + 1) / len(files_to_process))
                
                status_text.text("✅ Processing complete!")
                st.success(f"Successfully processed {len(files_to_process)} document(s)!")
                
                # Auto-switch to results tab
                st.rerun()
            else:
                st.warning("Please upload at least one document to process.")
    
    with tab2:
        st.header("Analytics Dashboard")
        if st.session_state.processed_documents:
            create_analytics_dashboard(st.session_state.processed_documents)
        else:
            st.info("No documents processed yet. Upload and process documents to see analytics.")
    
    with tab3:
        st.header("Processing Results")
        
        if st.session_state.processed_documents:
            # Display options
            col1, col2, col3 = st.columns(3)
            with col1:
                view_mode = st.selectbox("View Mode", ["Cards", "Table", "JSON"])
            with col2:
                sort_by = st.selectbox("Sort By", ["Filename", "Confidence", "Word Count", "Processing Date"])
            with col3:
                if st.button("🗑️ Clear All Results"):
                    st.session_state.processed_documents = []
                    st.rerun()
            
            # Display results based on view mode
            if view_mode == "Cards":
                for i, metadata in enumerate(st.session_state.processed_documents):
                    display_metadata_card(metadata, i)
            
            elif view_mode == "Table":
                # Create DataFrame for table view
                table_data = []
                for metadata in st.session_state.processed_documents:
                    table_data.append({
                        'Filename': metadata.filename,
                        'Title': metadata.title,
                        'Document Type': metadata.document_type,
                        'Word Count': metadata.word_count,
                        'Confidence': metadata.confidence_score,
                        'Sentiment': metadata.sentiment,
                        'Topics': ', '.join(metadata.primary_topics[:3])
                    })
                
                df = pd.DataFrame(table_data)
                st.dataframe(df, use_container_width=True)
            
            elif view_mode == "JSON":
                for i, metadata in enumerate(st.session_state.processed_documents):
                    with st.expander(f"JSON - {metadata.filename}"):
                        json_output = generator.export_metadata(metadata, 'json')
                        st.code(json_output, language='json')
            
            # Bulk export
            st.subheader("📥 Export Results")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📄 Export All as JSON"):
                    all_metadata = []
                    for metadata in st.session_state.processed_documents:
                        json_data = json.loads(generator.export_metadata(metadata, 'json'))
                        all_metadata.append(json_data)
                    
                    output = json.dumps(all_metadata, indent=2)
                    st.download_button(
                        "Download JSON",
                        output,
                        file_name="metadata_export.json",
                        mime="application/json"
                    )
            
            with col2:
                if st.button("📊 Export as CSV"):
                    table_data = []
                    for metadata in st.session_state.processed_documents:
                        table_data.append({
                            'filename': metadata.filename,
                            'title': metadata.title,
                            'summary': metadata.summary,
                            'document_type': metadata.document_type,
                            'word_count': metadata.word_count,
                            'confidence_score': metadata.confidence_score,
                            'sentiment': metadata.sentiment,
                            'topics': '|'.join(metadata.primary_topics),
                            'keywords': '|'.join(metadata.keywords)
                        })
                    
                    df = pd.DataFrame(table_data)
                    csv = df.to_csv(index=False)
                    st.download_button(
                        "Download CSV",
                        csv,
                        file_name="metadata_export.csv",
                        mime="text/csv"
                    )
        
        else:
            st.info("No results to display. Process some documents first!")
    
    with tab4:
        st.header("Advanced Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔧 Processing Configuration")
            
            confidence_threshold = st.slider(
                "Confidence Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.1,
                help="Minimum confidence score for topic classification"
            )
            
            max_keywords = st.number_input(
                "Maximum Keywords",
                min_value=5,
                max_value=50,
                value=20,
                help="Maximum number of keywords to extract"
            )
            
            enable_ocr = st.checkbox("Enable OCR for Scanned Documents", value=True)
            enable_entity_linking = st.checkbox("Enable Entity Linking", value=False)
        
        with col2:
            st.subheader("📊 Output Configuration")
            
            include_metadata = st.multiselect(
                "Include in Output",
                ["File Statistics", "Language Features", "Document Structure", "Processing Logs"],
                default=["File Statistics", "Document Structure"]
            )
            
            output_language = st.selectbox(
                "Summary Language",
                ["Auto-detect", "English", "Spanish", "French", "German"],
                index=0
            )
            
            st.subheader("💾 Data Management")
            if st.button("🧹 Clear Processing Cache"):
                st.cache_resource.clear()
                st.success("Cache cleared successfully!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>🤖 Automated Metadata Generation System v1.0</p>
        <p>Built with Streamlit • Powered by Advanced NLP</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()