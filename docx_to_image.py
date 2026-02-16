import os
import subprocess

def convert_docx_to_pdf(docx_path, pdf_path):
    subprocess.run([
        "libreoffice", "--headless", "--convert-to", "pdf",
        docx_path, "--outdir", os.path.dirname(pdf_path)
    ], check=True)

def pdf_to_png(pdf_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    subprocess.run([
        "pdftoppm", pdf_path, os.path.join(output_folder, "page"),
        "-png"
    ], check=True)

def process_folder(input_folder, output_folder):
    for file in os.listdir(input_folder):
        if file.endswith(".docx"):
            docx_path = os.path.join(input_folder, file)
            pdf_path = docx_path.replace(".docx", ".pdf")

            convert_docx_to_pdf(docx_path, pdf_path)

            base_name = os.path.splitext(os.path.basename(docx_path))[0]
            temp_folder = os.path.join(output_folder, base_name)

            pdf_to_png(pdf_path, temp_folder)

            print("✅ Converted:", file)

# Run for train and test
process_folder("train", "train_images")
process_folder("test", "test_images")
