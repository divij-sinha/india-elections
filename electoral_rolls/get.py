# %%
import asyncio
import base64
import json
import logging
import os
from enum import Enum
from io import BytesIO

import pytesseract
from dotenv import load_dotenv
from httpx import AsyncClient, HTTPError
from PIL import Image

from utils.pickle_cache import pickle_cache

load_dotenv()

logging.basicConfig(
    filename="voter_scraper.log",
    encoding="utf-8",
    filemode="a",
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
    level=logging.INFO,
)

STOPCNT = os.environ.get("STOPCNT", 10)
FAILURE_RATE = os.environ.get("FAILURE_RATE", 0.2)

semaphore = asyncio.Semaphore(os.environ.get("SEMAPHORE", 10))

headers = {
    "Accept": "*/*",
    "User-Agent": "curl/8.7.1",
    "Host": "gateway-voters.eci.gov.in",
    "Origin": "https://voters.eci.gov.in",
    "Content-Type": "application/json",
}
base_url = "https://voters.eci.gov.in"
session = AsyncClient(base_url=base_url, headers=headers, timeout=300, verify=False)


class StateCode(Enum):
    ANDAMAN_NICOBAR_ISLANDS = "U01"
    ANDHRA_PRADESH = "S01"
    ARUNACHAL_PRADESH = "S02"
    ASSAM = "S03"
    ASSAM_BYE_ELECTION = "S50"
    BIHAR = "S04"
    CHANDIGARH = "U02"
    CHHATTISGARH = "S26"
    DADRA_NAGAR_HAVELI_DAMAN_DIU = "U03"
    GOA = "S05"
    GUJARAT = "S06"
    HARYANA = "S07"
    HIMACHAL_PRADESH = "S08"
    JAMMU_AND_KASHMIR = "U08"
    JHARKHAND = "S27"
    KARNATAKA = "S10"
    KERALA = "S11"
    LADAKH = "U09"
    LAKSHADWEEP = "U06"
    MADHYA_PRADESH = "S12"
    MAHARASHTRA = "S13"
    MANIPUR = "S14"
    MEGHALAYA = "S15"
    MIZORAM = "S16"
    NCT_OF_DELHI = "U05"
    NAGALAND = "S17"
    ODISHA = "S18"
    PUDUCHERRY = "U07"
    PUNJAB = "S19"
    RAJASTHAN = "S20"
    SIKKIM = "S21"
    TAMIL_NADU = "S22"
    TELANGANA = "S29"
    TRIPURA = "S23"
    UTTAR_PRADESH = "S24"
    UTTARAKHAND = "S28"
    WEST_BENGAL = "S25"


def retry_decorator(func):
    async def wrapper(*args, **kwargs):
        for i in range(2):
            try:
                return await func(*args, **kwargs)
            except HTTPError as e:
                logging.error(f"Failed with {e}")
                await asyncio.sleep(1)
                continue
        raise Exception("Failed to fetch")

    return wrapper


async def generic_get(session: AsyncClient, url: str, **kwargs: str):
    url = url.format(**kwargs)
    async with semaphore:
        response = await session.get(url, headers=headers)
    logging.debug(f"Fetched {url}")
    # tree = etree.HTML(response.text)
    result = json.loads(response.text)
    return result


async def generic_post(session: AsyncClient, url: str, request_data: dict, **kwargs: str):
    url = url.format(**kwargs)
    async with semaphore:
        response = await session.post(url, headers=headers, data=json.dumps(request_data))
    logging.debug(f"Fetched {url}")
    # tree = etree.HTML(response.text)
    result = json.loads(response.text)
    return result


@pickle_cache
async def get_publish_roll(session: AsyncClient, state_code: StateCode):
    url = "https://gateway-voters.eci.gov.in/api/v1/printing-publish/get-publish-roll"
    request_data = {"stateCd": state_code.value}
    return await generic_post(session=session, url=url, request_data=request_data)


@pickle_cache
async def get_constituencies(session: AsyncClient, state_code: StateCode):
    url = "https://gateway-voters.eci.gov.in/api/v1/common/constituencies?stateCode={state_code.value}"
    return await generic_get(session=session, url=url, state_code=state_code)


@pickle_cache
async def get_districts(session: AsyncClient, state_code: StateCode):
    url = f"https://gateway-voters.eci.gov.in/api/v1/common/districts/{state_code.value}"
    return await generic_get(session=session, url=url, state_code=state_code)


@pickle_cache
async def get_acs(session: AsyncClient, district_code: str):
    url = f"https://gateway-voters.eci.gov.in/api/v1/common/acs/{district_code}"
    return await generic_get(session=session, url=url, district_code=district_code)


@pickle_cache
async def get_part_list(session: AsyncClient, state_code: StateCode, district_code: str, ac_no: int):
    url = "https://gateway-voters.eci.gov.in/api/v1/printing-publish/get-part-list"
    request_data = {"stateCd": state_code.value, "districtCd": district_code, "acNumber": ac_no}
    return await generic_post(session=session, url=url, request_data=request_data)


async def generate_captcha(session: AsyncClient):
    url = "https://gateway-voters.eci.gov.in/api/v1/captcha-service/generateCaptcha/EROLL"
    return await generic_get(session=session, url=url)


async def get_solved_captcha(session: AsyncClient, mode: str):
    captcha = await generate_captcha(session=session)
    base64_string = captcha["captcha"].split(",")[-1]
    image_data = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_data))
    image.save(f"data/captcha_images/raw/{captcha["id"]}.jpg")  ## For future training
    # image.show()
    if mode == "manual":
        value = input("Enter the text: ")
    elif mode == "torch":
        from utils.captcha_solve import image_manip, solve_torch

        new_image = image_manip(image)
        value = solve_torch(new_image)
    elif mode == "auto":
        from utils.captcha_solve import image_manip

        new_image = image_manip(image)
        custom_config = r"--oem 1 --psm 8 -c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyz0123456789"
        value = pytesseract.image_to_string(new_image, config=custom_config)
    elif mode == "claude":
        from utils.captcha_solve import solve_captcha_claude

        resp = solve_captcha_claude(image)
        value = json.loads(resp.content)["content"][0]["text"]

    return {"id": captcha["id"], "value": value}


async def get_ge_roll(session: AsyncClient, state_code: StateCode, district_code: str, ac_no: int, part_no: int):
    url = "https://gateway-voters.eci.gov.in/api/v1/printing-publish/generate-published-geroll"
    captcha = await get_solved_captcha(session=session, mode="torch")
    request_data = {
        "stateCd": state_code.value,
        "districtCd": district_code,
        "acNumber": ac_no,
        "partNumber": part_no,
        "captcha": captcha["value"],
        "captchaId": captcha["id"],
        "langCd": "ENG",
    }
    logging.debug(request_data)
    return await generic_post(session=session, url=url, request_data=request_data)


def save_result(result: dict, f_path: str):
    if result["statusCode"] == 200:
        os.makedirs(f_path, exist_ok=True)
        with open(f_path + result["refId"], "wb") as f:
            f.write(base64.b64decode(result["file"]))
        return 0
    else:
        logging.error(f"Failed to get: {f_path} with {result}")
        return -1


async def run_full_state(state_code: StateCode):
    districts = await get_districts(session, state_code)
    district_codes = [(d["districtCd"], d["districtValue"]) for d in districts]
    for district_code, d_val in district_codes:
        acs = await get_acs(session, district_code)
        ac_nos = [(a["asmblyNo"], a["asmblyName"]) for a in acs]
        for ac_no, ac_val in ac_nos[1:]:
            parts = await get_part_list(session, state_code, district_code, ac_no)
            part_nos = [(p["partNumber"], p["partName"]) for p in parts["payload"]]
            tasks = []
            for part_no, part_val in part_nos[:STOPCNT]:
                tasks.append(asyncio.create_task(save_part(state_code, district_code, d_val, ac_no, ac_val, part_val, part_no)))
            results = await asyncio.gather(*tasks)
            if results.count(-1) / len(results) > FAILURE_RATE:
                logging.error(f"High failure rate! for {state_code} {district_code}-{d_val} {ac_no}-{ac_val}")
                return -1
            return 0  ## early return for testing

    return 0


async def save_part(state_code, district_code, d_val, ac_no, ac_val, part_val, part_no):
    f_path = f"data/pdf/voter_rolls/{state_code.value}/{district_code}-{d_val}/{ac_no}-{ac_val}/{part_no}-{part_val}/"
    if os.path.exists(f_path):
        logging.info(f"Already exists {f_path}")
        return 1
    result = await get_ge_roll(session, state_code, district_code, ac_no, part_no)
    while result["statusCode"] == 400 and result["message"] == "Invalid Catpcha":  ## typo in captcha
        logging.error("Failed captcha trying again!")
        result = await get_ge_roll(session, state_code, district_code, ac_no, part_no)
    return save_result(result, f_path)


# %%
