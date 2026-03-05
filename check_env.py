import sys
import shutil
import subprocess

def check_python_package(package):
    try:
        __import__(package)
        print(f"[+] {package} is installed.")
        return True
    except ImportError:
        print(f"[!] {package} is NOT installed.")
        return False

print("--- Environment Check ---")
# Added pypdfium2, removed pdf2image check dependency on system tools
packages = ["cv2", "paddleocr", "pypdfium", "spellchecker", "numpy", "pandas", "streamlit"]
all_p_ok = all([check_python_package(p) for p in packages])

print("\n--- System Status Check ---")
if all_p_ok:
    print("\n[READY] Your environment is fully configured!")
    print("HandScript OCR will run entirely using Python libraries.")
else:
    print("\n[NOT READY] Please install missing packages using:")
    print("pip install -r requirements.txt")
