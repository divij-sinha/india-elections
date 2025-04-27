# %%
import base64
import json
import multiprocessing as mp
import os
import re
import uuid
from functools import partial
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from pdf2image import convert_from_path
from PIL import Image, ImageEnhance, ImageFilter
from pytesseract import image_to_string

from electoral_rolls.get import StateCode
from utils.claude import talk_claude

age_gender_pattern = re.compile("Age(.*?)Gender(.*?)$")
name_pattern = re.compile(r"(?:Name)(.*?)$")

# TODO: Not parsing some names, ages correctly


def extract_roll_images(images: list[Image]):
    roll_images = []
    for image in images:
        image = np.array(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                if w * h > 20_000:
                    cropped_rect = image[y : (y + h), x : (x + w)]
                    cropped_rect = Image.fromarray(cropped_rect).convert("L")
                    roll_images.append(cropped_rect)

    return roll_images


def extract_text_from_images(images: list[Image]):
    for image in images:
        yield extract_text_from_image(image)


def extract_text_from_image(image: Image, lang: str | None = None, img_folder: str | None | Path = None):
    lang = "eng" if lang is None else lang + "+eng"
    old_image = image
    image = ImageEnhance.Contrast(image).enhance(2.5)
    img_id = str(uuid.uuid4())
    if img_folder is not None:
        img_folder
        image.save(f"{img_folder}/id-{img_id}.jpg")
    image = image.crop((0, image.height // 5, image.width // 1.33, image.height))
    text = image_to_string(image, lang=lang)
    d = parse_text(text)
    d["img_id"] = img_id
    d["is_deleted_230"] = (
        np.array(
            ImageEnhance.Contrast(
                old_image.rotate(-35)
                .crop((old_image.width // 10, old_image.height // 10, old_image.width // 1.6, old_image.height // 2.5))
                .filter(ImageFilter.MaxFilter(3))
                .filter(ImageFilter.MedianFilter(7))
            ).enhance(5)
        )
        .mean()
        .round()
    )
    return d


def parse_text(text: str):
    if "പ്രായം" in text:
        text = re.sub(r"പ്രായം", "Age", text)
    if "ലിംഗം" in text:
        text = re.sub(r"ലിംഗം", "Gender", text)
    if "അച്ഛന്റെ" in text:
        text = re.sub(r"അച്ഛന്റെ", "Fathers", text)
    if "ഭര്‍ത്താവിന്റെ" in text:
        text = re.sub(r"ഭര്‍ത്താവിന്റെ", "Husbands", text)
    if ("പേര്" in text) or ("പേര" in text):
        text = re.sub(r"പേര്|പേര", "Name", text)
    if ("സ്ത്രീ" in text) or ("സ്തീ" in text) or ("സ്ത്ര" in text):
        text = re.sub(r"സ്ത്രീ|സ്തീ|സ്ത്ര", "Female", text)
    if "പുരുഷന്‍" in text:
        text = re.sub(r"പുരുഷന്‍", "Male", text)
    if "വിട്ടു നമ്പര്" in text:
        text = re.sub(r"വിട്ടു നമ്പര്", "Address", text)
    text = re.sub(r"[\u200c\u200d]", "", text)
    text_lines = [_ for _ in text.splitlines("\n") if len(_) > 0]
    d = {"Extra": ""}
    past_key = None
    for line in text_lines:
        line = line.replace("Photo", "").replace("Available", "")
        if "Gender" in line:
            match = re.search(age_gender_pattern, line)
            d["Age"] = re.sub(r"[^0-9]", "", match.group(1)).strip()
            d["Gender"] = re.sub(r"[^a-zA-Z\s]", "", match.group(2)).strip()
            past_key = "Gender"
        elif ":" in line:
            line_split = line.split(":")
            key = re.sub(r"[^a-zA-Z\s]", "", line_split[0]).strip()
            value = line_split[1].strip()
            d[key] = value
            past_key = key
        elif "Name" in line:
            match = re.search(name_pattern, line)
            d["Name"] = re.sub(r"[^\w\s]", "", match.group(1), flags=re.UNICODE).strip()
            past_key = "Name"
        elif past_key:
            d[past_key] += " " + line
        else:
            d["Extra"] += " " + line

    return d


def pdf_to_pieces(
    pdf_path: str, rolls: bool = True, last_page: bool = False, first_page: bool = False, process_polars: bool = False
):
    lang = pdf_path.split(".")[-2].split("-")[-3].lower()
    images = convert_from_path(pdf_path)

    first_page_table = None
    last_page_table = None

    if first_page:
        first_page_table = image_to_string(images[0])

    if last_page:
        cropped_image = images[-2].crop(
            (images[-2].width // 1.55, images[-2].height // 11, images[-2].width, images[-2].height // 1.5)
        )
        cropped_image = ImageEnhance.Contrast(cropped_image).enhance(2)
        resp = get_table_claude(cropped_image)
        last_page_table = json.loads(resp.content)["content"][0]["text"]

    texts, roll_images = None, None
    pdf_path = Path(pdf_path)
    img_folder = pdf_path.parent / "roll_images"
    img_folder.mkdir(parents=True, exist_ok=True)

    if rolls:
        roll_images = extract_roll_images(images[2:-2])
        texts = []
        extract_text_from_image_lang = partial(extract_text_from_image, lang=lang, img_folder=img_folder)
        with mp.Pool(processes=7) as pool:
            texts = pool.map(extract_text_from_image_lang, roll_images)

    if process_polars:
        import polars as pl

        if last_page:
            last_page_table = pl.read_csv(BytesIO(bytes(last_page_table, "utf-8")))

        # if first_page:
        #     first_page_table = pl.read_csv(BytesIO(bytes(first_page_table, "utf-8")))

        if rolls:
            texts = pl.DataFrame(texts)

    return {"rolls": (texts, roll_images), "first_page": first_page_table, "last_page": last_page_table}


def get_table_claude(image: Image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": img_str,
                    },
                },
                {
                    "type": "text",
                    "text": "What is the table in this image? DO NOT REPLY WITH ANYTHING ELSE. REPLY ONLY WITH THE TABLE IN THIS IMAGE. IT SHOULD BE IN CSV FORMAT. NO MORE AND NO LESS.",
                },
            ],
        }
    ]
    resp = talk_claude(messages=messages)
    return resp


def process_state(state_code: StateCode):
    root_data_dir = "data/pdf/voter_rolls"
    dists = os.listdir(f"{root_data_dir}/{state_code.value}")
    for dist in dists:
        acs = os.listdir(f"{root_data_dir}/{state_code.value}/{dist}")
        for ac in acs:
            parts = os.listdir(f"{root_data_dir}/{state_code.value}/{dist}/{ac}")
            for part in parts:
                files = os.listdir(f"{root_data_dir}/{state_code.value}/{dist}/{ac}/{part}")
                pdf_files = [file for file in files if file.endswith(".pdf")]
                for pdf_file in pdf_files:
                    pdf_path = f"{root_data_dir}/{state_code.value}/{dist}/{ac}/{part}/{pdf_file}"
                    print(f"Processing {pdf_path}")
                    data = pdf_to_pieces(pdf_path, rolls=True, last_page=False, first_page=False, process_polars=True)
                    print(f"Processed {pdf_path}!")
                    data["rolls"][0].write_parquet(
                        f"{root_data_dir}/{state_code.value}/{dist}/{ac}/{part}/{pdf_file.replace('.pdf', '.parquet')}",
                    )
    print(f"Processed {state_code}!")
    return data


# %%
