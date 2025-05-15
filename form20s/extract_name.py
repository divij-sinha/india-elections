import os
import pytesseract
from pdf2image import convert_from_path
from PIL import Image

pdf_folder = os.path.join("data/pdf/ls/2024/")
state_folders = [f for f in os.listdir(pdf_folder)]

# for state_folder in state_folders:
pdfs = [f for f in os.listdir(os.path.join(pdf_folder, state_folders[5])) if f.endswith(".pdf")]
for pdf in pdfs[0:20]:
    pdf_path = os.path.join(pdf_folder, state_folders[5], pdf)
    print('Processing:', pdf_path)
    image = convert_from_path(pdf_path)[0]
    print(image.size)
    bounding_box = (600, 300, 1800, 400)
    cropped_image = image.crop(bounding_box)
    cropped_image.show()
    text = pytesseract.image_to_string(cropped_image)
    print(f"Page 1:\n{text}\n")
