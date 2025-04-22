import os
import camelot

pdf_folder = os.path.join("data/pdf/ls/2024/", "kerala")
csv_folder = os.path.join("data/csv/", "kerala") 

pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]
os.makedirs(csv_folder, exist_ok=True)

for pdf_file in pdf_files:
    file_base = os.path.splitext(pdf_file)[0]
    pdf_path = os.path.join(pdf_folder, pdf_file)
    csv_path = os.path.join(csv_folder, file_base + ".csv")

    print(f"Converting {pdf_file} to CSV...")
    output_subfolder = os.path.join(csv_folder, file_base)
    os.makedirs(output_subfolder, exist_ok=True)

    output_path = os.path.join(output_subfolder, file_base + ".csv")
    tables = camelot.read_pdf(pdf_path,pages='1-end')
    tables.export(output_path,f='csv')