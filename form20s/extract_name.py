import os
import pytesseract
import re
import pandas as pd
from pdf2image import convert_from_path
from PIL import Image

pdf_folder = os.path.join("data/pdf/ls/2024/")
state_folders = [f for f in os.listdir(pdf_folder)]
data_folder = os.path.join("data/")

df = pd.DataFrame(columns=['file_name', 'state', 'no_electors', 'no_assembly', 'name_assembly'])

for state_folder in state_folders:
    pdfs = [f for f in os.listdir(os.path.join(pdf_folder, state_folder)) if f.endswith(".pdf")]
    for pdf in pdfs:
        pdf_path = os.path.join(pdf_folder, state_folder, pdf)
        print('Processing:', pdf_path)
        image = convert_from_path(pdf_path, last_page=1)[0]
        bounding_box = (600, 300, 1800, 400)
        cropped_image = image.crop(bounding_box)
        # cropped_image.show()
        text = pytesseract.image_to_string(cropped_image)
        lines = text.splitlines()

        no_electors = re.search(r'\d+$', lines[0]).group()
        no_assembly = re.search(r'(\d+)-', text).group(1)
        name_assembly = re.search(r'-(.*?)\s+Assembly Election', text).group(1).strip()

        new_row = pd.DataFrame([{
            'file_name': pdf,
            'state': state_folder,
            'no_electors': no_electors,
            'no_assembly': no_assembly,
            'name_assembly': name_assembly
        }])

        df = pd.concat([df, new_row], ignore_index=True)

df.to_excel(os.path.join(data_folder, 'pdf_ac_linkage.xlsx'), index=False)
