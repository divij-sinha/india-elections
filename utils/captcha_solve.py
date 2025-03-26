# %%

import base64
import os
import pathlib
from io import BytesIO

import cv2
import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from scipy import ndimage
from torchvision import transforms

from utils.claude import talk_claude
from utils.pytorch_model import CharacterCNN, get_predicted_character

current_dir = pathlib.Path(__file__).parent.resolve()


def image_manip(image: Image) -> Image:
    image = image.filter(ImageFilter.RankFilter(rank=1, size=3))
    image = image.resize((image.width * 4, image.height * 4), Image.LANCZOS)
    image = image.split()[1]
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2.0)
    img_array = np.array(image)
    img_array = remove_lines(img_array)
    image = Image.fromarray(img_array)
    threshold = 100  # Higher threshold to keep more of the text
    image = image.point(lambda p: 255 if p > threshold else 0)
    image = image.filter(ImageFilter.MedianFilter(9))
    img_array = np.array(image)
    kernel = np.ones((2, 2), np.uint8)
    img_array = ndimage.binary_erosion(img_array, structure=kernel).astype(np.uint8) * 255
    img_array = ndimage.binary_dilation(img_array, structure=kernel).astype(np.uint8) * 255
    # pre_line_image = Image.fromarray(img_array)
    img_array = remove_lines(img_array)
    image = Image.fromarray(img_array)

    # image = image.filter(ImageFilter.ModeFilter(10))

    # cv2.bitwise_and(img, line_mask)
    return image


def remove_lines(img):
    # Detect lines using HoughLinesP
    # edges = cv2.Canny(img, 50, 150, apertureSize=3)
    # lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)
    # Create a mask of lines
    line_mask = np.zeros_like(img)
    ## add diagonal line
    cv2.line(line_mask, (2, img.shape[0]), (img.shape[1], 2), 255, 5)
    result = cv2.bitwise_or(img, line_mask)
    return result


# async def save_images_for_training(session: AsyncClient, offset: int = 100):
#     for i in range(30):
#         captcha = await generate_captcha(session=session)
#         base64_string = captcha["captcha"].split(",")[-1]
#         image_data = base64.b64decode(base64_string)
#         image = Image.open(BytesIO(image_data))
#         new_image = image_manip(image)
#         image.save(f"data/captcha_images/raw/{i+offset}.jpg")
#         new_image.save(f"data/captcha_images/processed/none.timesitalic.exp{i+offset}.png")
#         for idx, (s, e) in enumerate([(90, 210), (190, 310), (290, 410), (390, 510), (490, 610), (590, 710)]):
#             cp_img = new_image.crop((s, 0, e, 320))
#             cp_img.save(f"data/captcha_images/cropped/none.timesitalic.exp{i+offset}.crop{idx}.png")


# %%
# def adjust_white_space():
#     all_x = []
#     all_y = []
#     root_path = "data/captcha_images/cropped/"
#     for p in os.listdir(root_path):
#         image = Image.open(root_path + p)
#         h = []
#         for i in range(320):
#             h.append(255.0 - np.array(image)[i].mean())
#         h = np.array(h)
#         zpoints = np.where(np.array(h) < 10)[0]
#         len_zpoints = len(zpoints)

#         bpoints = []
#         prev = None

#         for idx, i in enumerate(zpoints):
#             if idx == 0 or idx == len_zpoints - 1:
#                 bpoints.append(i)
#                 prev = i
#                 continue

#             if i - prev > 1:
#                 bpoints.append(prev)
#                 bpoints.append(i)

#             prev = i

#         bpoints = bpoints[1:-1]
#         start = bpoints[0]
#         end = bpoints[-1]
#         image = image.crop((0, start, image.size[0], end))
#         all_x.append(image.size[0])
#         all_y.append(image.size[1])
#         image.save("data/captcha_images/cropped_y/" + p)


def solve_torch(image: Image):
    device = torch.device("mps")
    model = CharacterCNN().to(device)
    model_path = os.path.join(current_dir, "captcha_model.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((320, 120)),
            transforms.ToTensor(),
        ]
    )
    full_text = ""
    for idx, (s, e) in enumerate([(90, 210), (190, 310), (290, 410), (390, 510), (490, 610), (590, 710)]):
        cp_img = image.crop((s, 0, e, 320))
        cp_img = cp_img.convert("L")
        image_tensor = transform(cp_img).unsqueeze(0).to(device)
        with torch.no_grad():
            output_img = model(image_tensor)
        predicted_char_img = get_predicted_character(output_img)
        full_text += predicted_char_img
    return full_text


def solve_captcha_claude(image: Image):
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
                    "text": "What is the text in this image? DO NOT REPLY WITH ANYTHING ELSE. REPLY ONLY WITH THE TEXT IN THIS IMAGE. IT SHOULD BE 6 CHARACTERS ONLY. NO MORE AND NO LESS.",
                },
            ],
        }
    ]
    resp = talk_claude(messages=messages)
    return resp
