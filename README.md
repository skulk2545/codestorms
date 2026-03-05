# 📝 HandScript AI - Advanced Handwritten OCR & Note Converter

Turn your messy handwritten notes into clean, searchable, and structured digital text instantly. **HandScript AI** is an optimized OCR system designed for speed and accuracy, specifically tailored for handwritten PDF documents.

## 🚀 Speed & Accuracy Updates
Our latest version is **10x faster** than standard OCR implementations:
- **Optimized DPI**: Intelligent resolution scaling (150 DPI) for lightning-fast processing.
- **Advanced Pre-processing**: Uses CLAHE + Sharpening + Median Filtering to recover faded or low-contrast strokes.
- **Skip Logic**: Automatic skipping of compute-intensive classification when not needed.

## ✨ Key Features
- **PDF to Text**: Convert multi-page handwritten PDFs into text area outputs or downloadable `.txt` files.
- **🤖 Note Query Engine**: Ask questions like "What are the action items?" and the AI will scan your notes and synthesize a detailed answer.
- **Intelligent Structure**: Automated line-joining logic that reconstructs full sentences from broken OCR lines.
- **Scan Enhancement**: Adaptive thresholding for low-contrast or poorly lit scans.
- **Local SQLite Storage**: Saves all extractions with confidence scores for future reference.

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/skulk2545/codestorms.git
   cd codestorms
   ```

2. **Set up the environment:**
   ```bash
   # Create a virtual environment
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate

   # Install dependencies
   pip install -r requirements.txt
   ```

## 💻 Usage

### Launch the UI (Streamlit)
```bash
streamlit run streamlit_app.py
```
Upload your handwritten PDF and start asking questions about your notes!

### Command Line Interface
```bash
python app.py your_notes.pdf --dpi 150 --out extracted.txt
```

## 🧠 Technology Stack
- **OCR Engine**: [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) (Optimized for handwriting)
- **Semantic Search**: `Sentence-Transformers` (all-MiniLM-L6-v2)
- **Frontend**: Streamlit
- **Image Processing**: OpenCV, PIL
- **Backend**: Python, SQLite

## 📝 Project Structure
- `app.py`: Core OCR logic, image preprocessing, and semantic retriever.
- `streamlit_app.py`: Modern web interface for file uploads and Q&A.
- `requirements.txt`: Project dependencies.

---
Built during **Hackathon 2026** for superior note recovery.
