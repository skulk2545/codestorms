import os
import cv2
import sqlite3
import pandas as pd
import numpy as np
import pypdfium2 as pdfium
from paddleocr import PaddleOCR
from spellchecker import SpellChecker
import argparse
from datetime import datetime
import re
from sentence_transformers import SentenceTransformer, util
import torch

class NoteRetriever:
    """Handles local semantic search over extracted notes."""
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        # Small and fast model for hackathons
        self.model = SentenceTransformer(model_name)
        self.corpus = []
        self.page_mapping = []
        self.corpus_embeddings = None

    def add_notes(self, pages_results):
        """Indexes pages for search."""
        self.corpus = []
        self.page_mapping = []
        
        for res in pages_results:
            text = res['text']
            # 1. Split into natural paragraphs from the new line-joining logic
            page_blocks = [p.strip() for p in text.split('\n') if len(p.strip()) > 30]
            
            # 2. Add complete paragraph blocks
            for block in page_blocks:
                self.corpus.append(block)
                self.page_mapping.append(res['page'])
            
            # 3. Add sliding window of blocks ONLY if we have many blocks, to preserve context
            if len(page_blocks) > 2:
                for i in range(len(page_blocks) - 1):
                    # Join 2 large blocks for extra detail
                    joined_block = page_blocks[i] + " " + page_blocks[i+1]
                    self.corpus.append(joined_block)
                    self.page_mapping.append(res['page'])
        
        # Deduplicate corpus to keep answers clean
        seen_corpus = set()
        unique_corpus = []
        unique_mapping = []
        for i, text in enumerate(self.corpus):
            if text not in seen_corpus:
                unique_corpus.append(text)
                unique_mapping.append(self.page_mapping[i])
                seen_corpus.add(text)
        
        self.corpus = unique_corpus
        self.page_mapping = unique_mapping
                
        if self.corpus:
            self.corpus_embeddings = self.model.encode(self.corpus, convert_to_tensor=True, show_progress_bar=False)

    def search(self, query, top_k=3):
        """Searches for answer in indexed notes and returns context for detailed answers."""
        if not self.corpus:
            return "No notes indexed for search."
            
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        # Calculate cosine similarity
        cos_scores = util.cos_sim(query_embedding, self.corpus_embeddings)[0]
        
        # Get top results
        top_results = torch.topk(cos_scores, k=min(top_k, len(self.corpus)))
        
        results = []
        for i in range(len(top_results.indices)):
            score = top_results.values[i].item()
            if score < 0.35: # Lowered threshold slightly to get more context
                continue
            idx = top_results.indices[i].item()
            results.append({
                "answer": self.corpus[idx],
                "page": self.page_mapping[idx],
                "confidence": score
            })
            
        if not results:
            return "ans not found in notes"
        
        return results

class HandwrittenOCRSystem:
    def __init__(self, output_dir="output", db_name="extracted_text.db"):
        self.output_dir = output_dir
        self.db_name = os.path.join(output_dir, db_name)
        # Initialize PaddleOCR with optimized settings for handwriting
        self.ocr = PaddleOCR(
            use_angle_cls=False, 
            lang='en', 
            show_log=False,
            det_db_thresh=0.3,
            det_db_box_thresh=0.5,
            rec_image_shape="3, 48, 320"
        )
        self.spell = SpellChecker()
        self.retriever = NoteRetriever()
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        self._init_db()

    def _init_db(self):
        """Initialize SQLite database for storing results."""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS extractions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT,
                page_number INTEGER,
                raw_text TEXT,
                clean_text TEXT,
                confidence FLOAT,
                timestamp DATETIME
            )
        ''')
        conn.commit()
        conn.close()

    def preprocess_image(self, image_np, use_thresholding=False):
        """Apply advanced filters to improve handwritten OCR accuracy."""
        # image_np is from PIL (RGB), convert to grayscale
        if len(image_np.shape) == 3 and image_np.shape[2] == 3:
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        elif len(image_np.shape) == 3 and image_np.shape[2] == 4:
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGBA2GRAY)
        else:
            gray = image_np

        if not use_thresholding:
            return gray

        # 1. Contrast Enhancement using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        contrasted = clahe.apply(gray)
        
        # 2. Sharpening to define handwriting strokes better
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(contrasted, -1, kernel)
        
        # 3. Median filter (faster than Bilateral)
        denoised = cv2.medianBlur(sharpened, 3)
        
        # 4. Adaptive Thresholding: great for low-contrast scans
        denoised = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        return denoised

    def convert_pdf_to_images(self, pdf_path, dpi=300):
        """Convert PDF pages to PIL images using pypdfium2."""
        pdf = pdfium.PdfDocument(pdf_path) # converts pdf to img
        
        images = []
        for page in pdf:
            bitmap = page.render(scale=dpi/72)
            pil_image = bitmap.to_pil()
            images.append(pil_image)
        return images

    def auto_correct_word(self, word):
        """Attempts to auto-correct a word. Returns original if no logical correction found."""
        # Remove punctuation for spell checking
        clean_word = re.sub(r'[^\w]', '', word)
        if not clean_word or len(clean_word) < 2:
            return word
            
        # If word is already correct, keep it
        if clean_word.lower() in self.spell:
            return word
            
        # Try to find a correction
        correction = self.spell.correction(clean_word)
        if correction and correction.lower() != clean_word.lower():
            # Maintain original capitalization if possible
            if word[0].isupper():
                return correction.capitalize()
            return correction
            
        return word

    def clean_text(self, text, auto_correct=True):
        """Clean noisy OCR output and intelligently join broken lines into sentences."""
        # Pre-cleaning: Remove obscure symbols but keep basic punctuation
        text = re.sub(r'[^\w\s\.,!\?\-\(\):]', '', text)
        
        raw_lines = text.split('\n')
        processed_lines = []
        
        for line in raw_lines:
            line = line.strip()
            if not line: continue
            
            # Basic noise filter: single chars that aren't 'a' or 'i'
            words = line.split()
            if len(words) == 1 and len(re.sub(r'[^\w]', '', words[0])) == 1 and words[0].lower() not in "ai":
                continue
            
            if auto_correct:
                cleaned_words = []
                for word in words:
                    if not word.isdigit() and not (word.isupper() and len(word) > 1):
                        word = self.auto_correct_word(word)
                    cleaned_words.append(word)
                line = " ".join(cleaned_words)
            
            processed_lines.append(line)

        # --- Intelligent Line Joining & Structure Recovery ---
        if not processed_lines:
            return ""

        final_text_blocks = []
        current_block = processed_lines[0]

        for next_line in processed_lines[1:]:
            # CLEANUP: Remove leading noise/bullets from the NEXT line if we're joining
            clean_next = re.sub(r'^[\-\•\-\*]\s*', '', next_line)
            
            # DECISION: Should we join?
            # 1. Next line starts with lowercase -> Definitely join
            # 2. Current block doesn't end in punctuation -> Join
            # 3. Next line is very short -> Maybe a fragment, join
            
            starts_with_lower = next_line[0].islower() if next_line[0].isalpha() else False
            ends_with_punc = current_block.rstrip()[-1:] in ".!?"
            
            if starts_with_lower or not ends_with_punc:
                # Join logic
                current_block = current_block.rstrip() + " " + clean_next
            else:
                # Break logic
                final_text_blocks.append(current_block)
                current_block = next_line
        
        final_text_blocks.append(current_block)
        
        # FINAL PASS: Clean up bullet artifacts and spacing
        cleaned_blocks = []
        for block in final_text_blocks:
            # Remove double bullets like "- - " or "• -"
            block = re.sub(r'^[\-\•\*\s]+', '• ', block)
            # Ensure it ends with a dot if it looks like a sentence
            if len(block) > 30 and block[-1].isalnum():
                block += "."
            cleaned_blocks.append(block)
            
        return "\n".join(cleaned_blocks)

    def save_to_db(self, filename, page_num, raw, clean, conf):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO extractions (filename, page_number, raw_text, clean_text, confidence, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (filename, page_num, raw, clean, float(conf), datetime.now()))
        conn.commit()
        conn.close()

    def process_pdf(self, pdf_path, dpi=150, auto_correct=False, use_thresholding=False):
        """Processes PDF and returns structured results."""
        filename = os.path.basename(pdf_path)
        pages = self.convert_pdf_to_images(pdf_path, dpi=dpi)
        results = []
        
        for i, page in enumerate(pages):
            page_np = np.array(page)
            processed_img = self.preprocess_image(page_np, use_thresholding=use_thresholding)
            processed_bgr = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)
            
            ocr_result = self.ocr.ocr(processed_bgr, cls=True)
            
            page_raw_text = []
            page_confidences = []
            
            if ocr_result and ocr_result[0]:
                for line in ocr_result[0]:
                    text = line[1][0]
                    conf = line[1][1]
                    page_raw_text.append(text)
                    page_confidences.append(conf)
            
            raw_text_str = "\n".join(page_raw_text)
            clean_text_str = self.clean_text(raw_text_str, auto_correct=auto_correct)
            avg_conf = np.mean(page_confidences) if page_confidences else 0
            
            self.save_to_db(filename, i+1, raw_text_str, clean_text_str, avg_conf)
            results.append({
                "page": i+1,
                "text": clean_text_str,
                "confidence": avg_conf
            })
        
        # Update retriever with new results
        self.retriever.add_notes(results)
        return results

    def pdf_to_text(self, pdf_path):
        """High-level method to get text directly from PDF."""
        results = self.process_pdf(pdf_path, dpi=150, auto_correct=False)
        return "\n".join([f"--- Page {r['page']} ---\n{r['text']}" for r in results])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Handwritten PDF to Text Converter")
    parser.add_argument("pdf", help="Path to the handwritten PDF file")
    parser.add_argument("--out", help="Output text file path", default="extracted_notes.txt")
    parser.add_argument("--dpi", type=int, default=150, help="DPI for scanning")
    parser.add_argument("--no-correct", action="store_false", dest="auto_correct", help="Disable auto-correction")
    args = parser.parse_args()
    
    if not os.path.exists(args.pdf):
        print(f"[!] Error: File {args.pdf} not found.")
    else:
        system = HandwrittenOCRSystem()
        import time
        try:
            start_time = time.time()
            results = system.process_pdf(args.pdf, dpi=args.dpi, auto_correct=args.auto_correct)
            end_time = time.time()
            text_output = "\n".join([f"--- Page {r['page']} ---\n{r['text']}" for r in results])
            with open(args.out, "w", encoding="utf-8") as f:
                f.write(text_output)
            print(f"[+] Success! Clean text extracted to: {args.out}")
            print(f"[+] Total execution time: {end_time - start_time:.2f} seconds")
        except Exception as e:
            print(f"[!] Critical Error: {str(e)}")
