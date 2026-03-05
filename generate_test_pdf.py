from PIL import Image
import os
import sys

def convert_img_to_pdf(img_path, pdf_path):
    try:
        image = Image.open(img_path)
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        image.save(pdf_path, "PDF", resolution=100.0)
        print(f"[+] Successfully converted {img_path} to {pdf_path}")
    except Exception as e:
        print(f"[!] Error: {e}")

if __name__ == "__main__":
    # Path to the image I just generated
    img_file = r"C:\Users\HP\.gemini\antigravity\brain\214d50c3-0233-4423-ac8b-715b72dbdddf\sample_handwritten_notes_1772690041993.png"
    output_pdf = "test_handwriting.pdf"
    
    if os.path.exists(img_file):
        convert_img_to_pdf(img_file, output_pdf)
    else:
        print(f"[!] Target image not found at {img_file}")
