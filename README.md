A robust standalone pipeline to convert handwritten PDF notes into clean structured text using **PaddleOCRv4** and **pypdfium2**.

## 🚀 Why this system?
- **🎯 Handwritten Excellence:** Specifically tuned for messy/cursive notes.
- **🛡️ 100% Local:** No cloud APIs. Your data is yours.
- **⚡ Poppler-Free:** Unlike other OCR systems, this requires NO external system tools. It runs entirely via Python.

## 🛠️ Setup Instructions

### 1. Install Python Packages
```bash
pip install -r requirements.txt
```
*(Model weights will download automatically on the first run)*

*Note: If you have an NVIDIA GPU, install `paddlepaddle-gpu` instead of `paddlepaddle` for faster processing.*

### 3. Run the System
Choose your preferred interface:

**A. Modern Web UI (Recommended)**
```bash
streamlit run streamlit_app.py
```

**B. Desktop GUI**
```bash
python ui_app.py
```

**C. Command Line**
```bash
python app.py "your_notes.pdf"
```

## 🛠️ Features
- **PDF Ingestion:** Converts multi-page PDFs to 300 DPI images.
- **Preprocessing:** Adaptive thresholding and denoising via OpenCV.
- **Handwriting OCR:** Uses PaddleOCR's deep learning model (excellent for handwriting).
- **Persistence:** Automatically saves extracted text, timestamps, and confidence scores to a local SQLite database (`output/extracted_text.db`).
- **Text Cleaning:** Basic rule-based cleaning to remove OCR noise.

## 📁 Output Structure
- `output/`: Contains the SQLite database and (optionally) cropped image sections.
- `output/extracted_text.db`: Your structured data for later retrieval.

## 🎯 Hackathon Tips
- **Accuracy:** If the text is messy, try increasing the `dpi` in `app.py` to 450 or 600.
- **Tables:** PaddleOCR identifies table structures. You can extend the `process_pdf` method to handle `PP-Structure` for complex layouts.
- **Offline:** The first time you run the script, it will download the OCR models (~100MB). After that, it works completely offline.
