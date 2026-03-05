import streamlit as st
import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from app import HandwrittenOCRSystem
import time
import base64

# --- Page Configuration ---
st.set_page_config(
    page_title="HandScript OCR - AI Note Converter",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styles ---
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #2e7d32; color: white; }
    .header-style { font-size: 40px; font-weight: bold; color: #1b5e20; text-align: center; margin-bottom: 20px; }
</style>
""", unsafe_allow_html=True)

# --- Initialization ---
@st.cache_resource
def load_ocr_system():
    return HandwrittenOCRSystem()

ocr_system = load_ocr_system()

# --- Sidebar ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2666/2666505.png", width=100)
    st.title("HandScript AI")
    st.markdown("---")
    st.subheader("🛠️ Accuracy Settings")
    dpi_value = st.slider("Conversion Res (DPI)", 100, 600, 150, help="Higher DPI improves accuracy but increases processing time.")
    auto_correct_toggle = st.toggle("Enable AI Auto-Correction (Spell Check)", value=False)
    scan_enhancement = st.toggle("Enable Scan Enhancement (Adaptive Thresholding)", value=False, help="Improves readability of faded or low-contrast handwritten notes.")
    st.info("System is using **CLAHE + Sharpening + Bilateral Filtering** for superior handwriting recovery.")
    st.markdown("---")
    if st.button("Clear Cache"):
        st.cache_resource.clear()
        st.rerun()

# --- Main App ---
st.markdown('<div class="header-style">📝 HandScript AI Converter</div>', unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Directly convert handwritten PDFs into clean, searchable Text.</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload your handwritten PDF here", type=["pdf"])

if uploaded_file is not None:
    # Save uploaded file to temp location
    temp_path = os.path.join("temp_upload.pdf")
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    col1, col2 = st.columns([1, 1])

    with col1:
        st.write("### 📄 Document Analysis")
        try:
            # Low DPI for fast preview
            pages_preview = ocr_system.convert_pdf_to_images(temp_path, dpi=100)
            st.image(pages_preview[0], caption="Page 1 Preview", use_column_width=True)
            st.success(f"File loaded: {len(pages_preview)} pages detected.")
        except Exception as e:
            st.error(f"Error reading PDF: {str(e)}")

    with col2:
        st.write("### ⚙️ Action Center")
        if st.button("🚀 Convert PDF to Clean Text"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            start_time = time.time()
            try:
                # Use flexible DPI and auto-correction for processing
                results = ocr_system.process_pdf(
                    temp_path, 
                    dpi=dpi_value, 
                    auto_correct=auto_correct_toggle,
                    use_thresholding=scan_enhancement
                )
                
                all_text_results = []
                for i, res in enumerate(results):
                    all_text_results.append(res)
                    progress_bar.progress((i + 1) / len(results))
                    
                end_time = time.time()
                st.balloons()
                st.success(f"Successfully converted to text in {end_time - start_time:.2f}s")
                st.session_state['ocr_results'] = all_text_results
                
            except Exception as e:
                st.error(f"Conversion failed: {str(e)}")

    # --- Results Section ---
    if 'ocr_results' in st.session_state:
        st.markdown("---")
        
        # Tabs for analysis
        tab1, tab2 = st.tabs(["🖋️ Extracted Text", "💬 Ask Your Notes"])
        
        with tab1:
            full_text = "\n\n".join([f"--- Page {r['page']} ---\n{r['text']}" for r in st.session_state['ocr_results']])
            st.text_area(label="Consolidated Text Output", value=full_text, height=400)
            
            # Download Buttons
            st.download_button(
                label="Download as Text (.txt)",
                data=full_text,
                file_name="extracted_notes.txt",
                mime="text/plain"
            )

        with tab2:
            st.subheader("🤖 Note Query Engine")
            st.write("Ask a question about your handwritten notes. The AI will look for relevant sections.")
            
            detail_level = st.select_slider(
                "Detail Level",
                options=["Quick Answer", "Detailed Context"],
                value="Detailed Context"
            )
            
            user_query = st.text_input("What would you like to find?", placeholder="Example: What are the action items for Saran?")
            
            if user_query:
                # Use our retriever from the OCR system
                with st.spinner("Analyzing notes for details..."):
                    # Get more results for synthesis
                    top_k = 12 if detail_level == "Detailed Context" else 4
                    search_results = ocr_system.retriever.search(user_query, top_k=top_k)
                    
                if isinstance(search_results, str):
                    st.warning(search_results)
                else:
                    st.success(f"Found {len(search_results)} highly relevant sections for your query.")
                    
                    # Synthesized Answer at Top
                    if detail_level == "Detailed Context":
                        st.markdown("### 📝 Detailed Context & Synthesis")
                        # Group by page for better flow
                        pages_found = sorted(list(set([r['page'] for r in search_results])))
                        full_synthesis = ""
                        for p in pages_found:
                            page_chunks = [r['answer'] for r in search_results if r['page'] == p]
                            # Clean up and join (blocks already have bullets from app.py)
                            page_text = "\n".join(page_chunks)
                            full_synthesis += f"**Page {p}:**\n{page_text}\n\n"
                        
                        # Move formatting outside the f-string to avoid syntax error
                        formatted_synthesis = full_synthesis.replace('\n', '<br>')
                        
                        st.markdown(f"""
                            <div style="background-color: #e8f5e9; padding: 20px; border-radius: 10px; border-left: 10px solid #2e7d32; margin-bottom: 20px;">
                                <div style="font-size: 1.15rem; color: #1b5e20; line-height: 1.7;">
                                    {formatted_synthesis}
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                        st.markdown("---")
                        st.subheader("🔍 Individual Matches Found")
                    
                    for i, res in enumerate(search_results):
                        with st.expander(f"Match {i+1} (Page {res['page']}) - Confidence: {res['confidence']:.2f}", expanded=(i==0)):
                            # High contrast display for handwritten extraction
                            st.markdown(f"""
                                <div style="background-color: #ffffff; padding: 15px; border-left: 5px solid #2e7d32; border-radius: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                                    <p style="color: #1a1a1a; font-size: 1.1rem; line-height: 1.6; font-weight: 500; margin: 0;">
                                        {res['answer']}
                                    </p>
                                </div>
                            """, unsafe_allow_html=True)
                    
                    if len(search_results) > 1 and detail_level == "Quick Answer":
                        st.markdown("---")
                        st.markdown("### 📝 Quick Summary")
                        full_context = " ".join([r['answer'] for r in search_results])
                        st.write(full_context)

else:
    st.info("Please upload a handwritten PDF to begin. Once processed, you'll be able to ask questions about your notes.")
