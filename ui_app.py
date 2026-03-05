import tkinter as tk
import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import threading
import os
import sys
from app import HandwrittenOCRSystem

# Set appearance and theme
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class HandwritingUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("HandScript OCR - Premium Handwritten Notes Converter")
        self.geometry("1100x700")

        # Initialize the OCR System
        self.ocr_system = HandwrittenOCRSystem()
        self.pdf_path = None
        self.processing = False

        # Create Layout
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Left Sidebar for Controls
        self.sidebar = ctk.CTkFrame(self, width=250, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        self.logo_label = ctk.CTkLabel(self.sidebar, text="HandScript OCR", font=ctk.CTkFont(size=24, weight="bold"))
        self.logo_label.pack(pady=(20, 30))

        self.select_btn = ctk.CTkButton(self.sidebar, text="Select Handwritten PDF", command=self.select_file, height=40)
        self.select_btn.pack(padx=20, pady=10)

        self.convert_btn = ctk.CTkButton(self.sidebar, text="Convert to Text", command=self.start_conversion, state="disabled", height=40, fg_color="#2ecc71", hover_color="#27ae60")
        self.convert_btn.pack(padx=20, pady=10)

        self.status_label = ctk.CTkLabel(self.sidebar, text="Status: Ready", text_color="gray")
        self.status_label.pack(pady=10)

        self.progress_bar = ctk.CTkProgressBar(self.sidebar)
        self.progress_bar.pack(padx=20, pady=10)
        self.progress_bar.set(0)

        self.appearance_mode_label = ctk.CTkLabel(self.sidebar, text="Appearance Mode:", anchor="w")
        self.appearance_mode_label.pack(padx=20, pady=(100, 0))
        self.appearance_mode_optionemenu = ctk.CTkOptionMenu(self.sidebar, values=["Light", "Dark", "System"], command=self.change_appearance_mode)
        self.appearance_mode_optionemenu.pack(padx=20, pady=10)
        self.appearance_mode_optionemenu.set("Dark")

        # Main Content Area
        self.main_content = ctk.CTkTabview(self, width=800)
        self.main_content.grid(row=0, column=1, sticky="nsew", padx=20, pady=10)
        self.main_content.add("Extracted Text")
        self.main_content.add("Page Preview")

        # Extracted Text Tab
        self.text_area = ctk.CTkTextbox(self.main_content.tab("Extracted Text"), font=ctk.CTkFont(size=14))
        self.text_area.pack(expand=True, fill="both", padx=10, pady=10)

        # Page Preview Tab
        self.preview_label = ctk.CTkLabel(self.main_content.tab("Page Preview"), text="No page selected")
        self.preview_label.pack(expand=True, fill="both")

    def select_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf")])
        if file_path:
            self.pdf_path = file_path
            self.status_label.configure(text=f"Selected: {os.path.basename(file_path)}", text_color="white")
            self.convert_btn.configure(state="normal")
            self.text_area.delete("1.0", tk.END)
            self.text_area.insert("1.0", f"Selected: {file_path}\nReady to convert...")

    def change_appearance_mode(self, new_appearance_mode: str):
        ctk.set_appearance_mode(new_appearance_mode)

    def start_conversion(self):
        if not self.pdf_path or self.processing:
            return

        self.processing = True
        self.convert_btn.configure(state="disabled")
        self.select_btn.configure(state="disabled")
        self.status_label.configure(text="Processing...", text_color="#f1c40f")
        self.progress_bar.set(0)
        
        # Run in thread to keep UI responsive
        thread = threading.Thread(target=self.run_ocr)
        thread.start()

    def run_ocr(self):
        try:
            # We override path to images for previewing in UI later if needed
            pages = self.ocr_system.convert_pdf_to_images(self.pdf_path)
            total_pages = len(pages)
            all_text = ""

            for i, page in enumerate(pages):
                self.status_label.configure(text=f"Processing Page {i+1}/{total_pages}...")
                self.progress_bar.set((i + 1) / total_pages)
                
                # Use current logic but extract results for UI
                filename = os.path.basename(self.pdf_path)
                import numpy as np
                import cv2
                page_np = np.array(page)
                ocr_result = self.ocr_system.ocr.ocr(page_np, cls=True)
                
                page_raw_text = []
                if ocr_result[0]:
                    for line in ocr_result[0]:
                        page_raw_text.append(line[1][0])
                
                raw_text_str = "\n".join(page_raw_text)
                clean_text_str = self.ocr_system.clean_text(raw_text_str)
                
                # Append to UI
                self.text_area.insert(tk.END, f"\n--- Page {i+1} ---\n{clean_text_str}\n")
                self.text_area.see(tk.END)
                all_text += f"\n--- Page {i+1} ---\n{clean_text_str}\n"

            self.status_label.configure(text="Conversion Complete!", text_color="#2ecc71")
            messagebox.showinfo("Success", "OCR Conversion finished successfully!")
            
        except Exception as e:
            self.status_label.configure(text="Error occurred", text_color="#e74c3c")
            messagebox.showerror("Error", f"Failed to process PDF: {str(e)}")
        finally:
            self.processing = False
            self.convert_btn.configure(state="normal")
            self.select_btn.configure(state="normal")
            self.progress_bar.set(1)

if __name__ == "__main__":
    app = HandwritingUI()
    app.mainloop()
