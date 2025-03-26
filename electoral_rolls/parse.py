# %%
import multiprocessing as mp
import re

import cv2
import numpy as np
from pdf2image import convert_from_path
from PIL import Image, ImageEnhance, ImageFilter
from pytesseract import image_to_string

age_gender_pattern = re.compile("Age(.*?)Gender(.*?)$")


def calc_deleted_score(image: Image) -> float:
    deleted_image = image
    deleted_image = deleted_image.rotate(-30).crop((0, 0, deleted_image.width // 1.4, deleted_image.height // 2))
    # deleted_image = deleted_image.filter(ImageFilter.RankFilter(rank=8, size=3))
    # enhancer = ImageEnhance.Contrast(deleted_image)
    # deleted_image = enhancer.enhance(2.0)
    threshold = 130
    deleted_image = deleted_image.point(lambda p: 255 if p > threshold else 0)
    deleted_image = deleted_image.filter(ImageFilter.RankFilter(rank=9, size=5))
    deleted_text = image_to_string(deleted_image)
    deleted_score = len(re.findall("D|E|L|T|2", deleted_text, flags=re.IGNORECASE))

    deleted_mean = np.array(deleted_image).mean()

    if deleted_score > 2:
        deleted_image.show()

    ## deleted 2
    deleted_image = image
    deleted_image = deleted_image.rotate(-40).crop((0, 0, deleted_image.width // 1.4, deleted_image.height // 2))
    deleted_image = ImageEnhance.Contrast(deleted_image).enhance(4)
    deleted_image = deleted_image.filter(ImageFilter.MaxFilter(3))
    deleted_text = image_to_string(deleted_image)
    deleted_score_2 = len(re.findall("D|E|L|T|2", deleted_text, flags=re.IGNORECASE))

    deleted_mean_2 = np.array(deleted_image).mean()

    if deleted_score_2 > 2:
        deleted_image.show()
    return deleted_score, deleted_score_2, deleted_mean, deleted_mean_2


def extract_roll_images(images: list[Image]):
    roll_images = []
    for image in images:
        ## horizontal split
        # horizontal_breaks = [(50, 843), (843, 1637), (1637, 2430)]
        # for start, end in horizontal_breaks:
        #     col = image.crop((start, 0, end, images[0].height))
        image = np.array(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # Edge detection using Canny
        edges = cv2.Canny(blurred, 50, 150)
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Iterate through contours
        for contour in contours:
            # Approximate the contour to a polygon
            approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
            # Check if the polygon has 4 sides (rectangle)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                if w * h > 20_000:
                    # print(x, (x + w), y, (y + h))
                    cropped_rect = image[y : (y + h), x : (x + w)]
                    cropped_rect = Image.fromarray(cropped_rect).convert("L")
                    roll_images.append(cropped_rect)

    return roll_images


def extract_text_from_images(images: list[Image]):
    for image in images:
        yield extract_text_from_image(image)


def extract_text_from_image(image: Image):
    # is_deleted_score, is_deleted_score_2, deleted_mean, deleted_mean_2 = calc_deleted_score(image)
    old_image = image
    image = ImageEnhance.Contrast(image).enhance(2.5)
    text = image_to_string(image)
    d = parse_text(text)
    # d["is_deleted_score"] = is_deleted_score
    # d["is_deleted_score_2"] = is_deleted_score_2
    # d["image_mean"] = np.array(image).mean()
    # d["is_deleted_mean"] = deleted_mean
    # d["is_deleted_mean_2"] = deleted_mean_2
    # d["image_pre_rotate"] = (
    #     np.array(old_image.rotate(-40).crop((0, 0, old_image.width // 1.4, old_image.height // 2))).mean().round()
    # )
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
    # if d["image_pre_rotate_filter"] < 230:
    #     old_image.show()
    # print(d["image_pre_rotate_filter"])
    # d["image_post_rotate"] = np.array(image.rotate(-40).crop((0, 0, image.width // 1.4, image.height // 2))).mean().round()
    return d


def parse_text(text: str):
    text_lines = [_ for _ in text.splitlines() if len(_) > 0]
    d = {"Extra": []}
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
        # elif "Photo" in line or "Available" in line:
        #     pass
        elif past_key:
            d[past_key] += " " + line
        else:
            d["Extra"].append(line)

    return d


def pdf_to_pieces(pdf_path: str):
    images = convert_from_path(pdf_path)
    roll_images = extract_roll_images(images[2:-2])
    texts = []
    cnt = 0
    with mp.Pool(processes=7) as pool:
        texts = pool.map(extract_text_from_image, roll_images)
    # for text in extract_text_from_images(roll_images):
    #     texts.append(text)
    # try:
    #     r = parse_text(text)
    #     r["is_deleted_score"] = is_deleted_score

    # except:
    #     cnt += 1
    #     print(text)
    #     print(f"{idx=}")
    #     pass

    print(f"{cnt=}")
    return texts, images, roll_images


# %%
# test_images = [
#     roll_images[69],
#     roll_images[8],
#     roll_images[113],
#     roll_images[383],
#     roll_images[696],
#     roll_images[29],
#     roll_images[-5],
#     roll_images[1024],
# ]
# %%
