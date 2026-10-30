# %%
## New ECI download flow (2026): encrypted request bodies, roll type selection, CDN/file-id downloads
import asyncio
import base64
import json
import logging
import os
import random
from functools import wraps

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from httpx import AsyncClient, HTTPError

from electoral_rolls.get import (
    FAILURE_RATE,
    StateCode,
    get_acs,
    get_acs_languages,
    get_districts,
    get_part_list,
    get_solved_captcha,
    parse_lang,
)
from utils.pickle_cache import pickle_cache

## Parts per AC, unset -> all parts
STOPCNT = int(os.environ["STOPCNT"]) if os.environ.get("STOPCNT") else None
geroll_semaphore = asyncio.Semaphore(int(os.environ.get("GEROLL_SEMAPHORE", 1)))

headers = {
    "Accept": "*/*",
    "User-Agent": "Mozilla/5.0",
    "Origin": "https://voters.eci.gov.in",
    "Content-Type": "application/json",
    "appName": "VSP",
    "applicationName": "VSP",
    "PLATFORM-TYPE": "ECIWEB",
    "channelidobo": "VSP",
}
session = AsyncClient(headers=headers, timeout=300)

## Public key and misKey are from the site's frontend JS
PUBLIC_KEY = serialization.load_der_public_key(
    base64.b64decode(
        "MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEArb7++BxL/YN8OIln+6FL9Gnw5DNmQ/VFZXss+J+TuQyJc891JbqbijxYQNEin2c2u+CnpXpoGQ/1gUSzDMJeNS3sNSlIUykp2dt7xIm/cmV4sZ/c769vCxVRosMfRaZJnBAah+m1X26lEhnOo0wpAB9Txr8RIyBe6h7PiQWykeJeh6UacOBBX28kgkq7+vJhW8HgB38lt32XRocznRYwS9LqR7ZweFmQhTr1+EGrqiEKCOCxMYgHR2SQckb96hZ9kWzfzeun4bUO5oXKJciLkiS1IgKieADEvYLgu129ZIpn1H+8H+8ikNNVETqEDDMtqcQcQmWppJvcWHaXAs+f8QIDAQAB"
    )
)
OAEP = padding.OAEP(mgf=padding.MGF1(hashes.SHA256()), algorithm=hashes.SHA256(), label=None)
MIS_KEY = "EROLLA32DVI09AJH"
ROLL_YEAR = int(os.environ.get("ROLL_YEAR", 2024))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 8))  ## site allows up to 10 parts per request


def encrypt_body(data: dict):
    ## AES-256-GCM over the json body, AES key wrapped with RSA-OAEP(SHA-256)
    key, iv = os.urandom(32), os.urandom(12)
    ciphertext = AESGCM(key).encrypt(iv, json.dumps(data).encode(), None)
    return {
        "encryptedPayload": base64.b64encode(ciphertext).decode(),
        "encryptedKey": base64.b64encode(PUBLIC_KEY.encrypt(key, OAEP)).decode(),
        "iv": base64.b64encode(iv).decode(),
    }


def encrypt_params(*params):
    ## Same scheme for query params, urlsafe b64: [wrapped key, iv, *encrypted params]
    def b64(b: bytes):
        return base64.urlsafe_b64encode(b).decode().rstrip("=")

    key, iv = os.urandom(32), os.urandom(12)
    aes = AESGCM(key)
    return [b64(PUBLIC_KEY.encrypt(key, OAEP)), b64(iv)] + [b64(aes.encrypt(iv, json.dumps(p).encode(), None)) for p in params]


@pickle_cache
async def get_publish_eroll_type(session: AsyncClient, state_code: StateCode, year: int):
    yek, rotcev, state_cd, year_enc, mis_key = encrypt_params(state_code.value, year, MIS_KEY)
    url = f"https://gateway-voters.eci.gov.in/api/v1/printing-publish/get-publish-eroll-type?stateCd={state_cd}&year={year_enc}&misKey={mis_key}"
    response = await session.get(url, headers={"accept_yek": yek, "accept_rotcev": rotcev})
    return json.loads(response.text)


def retry(func):
    ## Like get.retry, but also retries bad responses (non-json, missing keys, bad base64) instead of crashing the run
    @wraps(func)
    async def wrapper(*args, **kwargs):
        cnt = 3
        while cnt > 0:
            try:
                result = await func(*args, **kwargs)
            except (HTTPError, ValueError, KeyError) as e:
                logging.error(f"Failed with {e!r}")
                await asyncio.sleep(2)
                cnt -= 1
                continue
            if result.get("statusCode", 0) == 200:
                return result
            logging.error(f"Failed to acquire GE Roll with {result} Trying again")
            cnt -= 1

        logging.error(f"Tries exhausted, giving up for {args} ; {kwargs}")
        return {"statusCode": 999, "message": "Internal retries exhausted"}

    return wrapper


async def generate_published_pdfs(
    session: AsyncClient, state_code: StateCode, district_code: str, ac_no: int, part_nos: list[int], lang_code: str, roll_id: str
):
    url = "https://gateway-voters.eci.gov.in/api/v1/printing-publish/generate-published-pdfs"
    captcha = await get_solved_captcha(session=session, mode="torch")
    request_data = {
        "stateCd": state_code.value,
        "acNumber": ac_no,
        "partNumberList": part_nos,
        "districtCd": district_code,
        "captcha": captcha["value"],
        "captchaId": captcha["id"],
        "langCd": lang_code,
        "publishedRollId": roll_id,
        "misKey": MIS_KEY,
    }
    logging.debug(request_data)
    response = await session.post(url, content=json.dumps(encrypt_body(request_data)))
    return json.loads(response.text)


async def download_file(session: AsyncClient, ref_id: str, file_ref: str):
    ## refId CDN -> file_ref is a static path, otherwise a short-lived fileId
    if ref_id == "CDN":
        response = await session.get(f"https://voters.eci.gov.in/eroll/{file_ref}")
        response.raise_for_status()
        return file_ref.split("/")[-1], response.content
    url = f"https://gateway-vpd.eci.gov.in/api/v1/ext-printing-publish/get-published-file?fileId={file_ref}"
    file_result = json.loads((await session.get(url, headers={"CurrentRole": "citizen", "Accept": "application/json"})).text)
    if file_result.get("statusCode", 0) != 200:
        raise ValueError(f"Failed to fetch file {file_ref} with {file_result}")
    return file_result["refId"], base64.b64decode(file_result["payload"])


def file_part_no(name: str):
    ## "2024-FC-EROLLGEN-S18-61-FinalRoll-Revision2-ORI-3-WI.pdf" -> 3
    return int(name.rsplit("-", 2)[-2])


@retry
async def get_roll_files(
    session: AsyncClient, state_code: StateCode, district_code: str, ac_no: int, part_nos: list[int], lang_code: str, roll_id: str
):
    ## One captcha for the whole batch; files are keyed by the part number in their name, not by payload order
    result = await generate_published_pdfs(session, state_code, district_code, ac_no, part_nos, lang_code, roll_id)
    if result.get("statusCode", 0) != 200 or not result.get("payload"):
        return result
    files = {}
    for file_ref in result["payload"]:
        if file_ref is None:
            continue
        try:
            name, content = await download_file(session, result["refId"], file_ref)
            files[file_part_no(name)] = (name, content)
        except (HTTPError, ValueError, KeyError) as e:
            logging.error(f"Failed to download {file_ref} with {e!r}")
    return {"statusCode": 200, "files": files}


def save_result(file: tuple[str, bytes] | None, f_path: str):
    if file is None:
        logging.error(f"Failed to get: {f_path}, missing from batch")
        return -1
    name, content = file
    try:
        os.makedirs(f_path, exist_ok=True)
        with open(f_path + name + "tmp", "wb") as f:
            f.write(content)
        os.replace(f_path + name + "tmp", f_path + name)
        return 0
    except Exception as e:
        logging.error(f"Failed to save {f_path} with error {e}")
        return -1


async def run_full_state(state_code: StateCode):
    logging.info(f"Running full state for {state_code}")
    roll_types = await get_publish_eroll_type(session, state_code, ROLL_YEAR)
    roll_id = f"{state_code.value}-{ROLL_YEAR}-GEN-FIR"  ## General Election roll, site default is the latest roll instead
    if roll_id not in [r["id"] for r in roll_types.get("payload") or []]:
        logging.error(f"Roll {roll_id} not published for {state_code}, got {roll_types}")
        return -1
    logging.info(f"Using roll {roll_id} for {state_code}")
    districts = await get_districts(session, state_code)
    district_codes = [(d["districtCd"], d["districtValue"].strip()) for d in districts]
    logging.info(f"Got {len(district_codes)} districts for {state_code}")
    for district_code, d_val in district_codes:
        logging.info(f"Running district {district_code}-{d_val}")
        acs = await get_acs(session, district_code)
        ac_nos = [(a["asmblyNo"], a["asmblyName"].strip()) for a in acs]
        logging.info(f"Got {len(ac_nos)} acs for {state_code} {district_code}-{d_val}")
        for ac_no, ac_val in ac_nos:
            logging.info(f"Running ac {ac_no}-{ac_val}")
            parts = await get_part_list(session, state_code, district_code, ac_no)
            languages = await get_acs_languages(session, state_code, district_code, ac_no)
            lang_code = parse_lang(languages)
            part_nos = [(p["partNumber"], p["partName"].strip()) for p in parts["payload"]]
            logging.info(f"Got {len(part_nos)} parts for {state_code} {district_code}-{d_val} {ac_no}-{ac_val}")
            logging.info(f"STOP COUNT at {STOPCNT}")
            part_nos = part_nos[:STOPCNT]
            todo = [(p, v) for p, v in part_nos if not part_done(part_path(roll_id, district_code, d_val, ac_no, ac_val, p, v))]
            logging.info(f"{len(part_nos) - len(todo)} parts already downloaded")
            tasks = []
            for i in range(0, len(todo), BATCH_SIZE):
                tasks.append(
                    asyncio.create_task(
                        save_parts(state_code, district_code, d_val, ac_no, ac_val, todo[i : i + BATCH_SIZE], lang_code, roll_id)
                    )
                )
            results = [1] * (len(part_nos) - len(todo)) + [r for batch in await asyncio.gather(*tasks) for r in batch]
            if not results:
                continue
            if results.count(-1) / len(results) > FAILURE_RATE:
                logging.error(f"High failure rate! for {state_code} {district_code}-{d_val} {ac_no}-{ac_val}")
                # return -1

    return 0


def part_path(roll_id, district_code, d_val, ac_no, ac_val, part_no, part_val):
    return f"data/pdf/voter_rolls/{roll_id}/{district_code}-{d_val}/{ac_no}-{ac_val}/{part_no}-{part_val}/"


def part_done(f_path):
    ## Only a finished pdf counts, a folder holding just a leftover "...pdftmp" is downloaded again
    return os.path.isdir(f_path) and any(f.endswith(".pdf") for f in os.listdir(f_path))


async def save_parts(state_code, district_code, d_val, ac_no, ac_val, parts, lang_code, roll_id):
    ## parts: [(part_no, part_val)], fetched with a single captcha / request
    f_paths = {p: part_path(roll_id, district_code, d_val, ac_no, ac_val, p, v) for p, v in parts}
    async with geroll_semaphore:
        await asyncio.sleep(random.randint(1, 3))
        result = await get_roll_files(session, state_code, district_code, ac_no, list(f_paths), lang_code, roll_id)
    if "files" not in result:
        logging.error(f"Failed to get: {list(f_paths.values())} with {result}")
        return [-1] * len(parts)
    return [save_result(result["files"].get(p), f_path) for p, f_path in f_paths.items()]


# %%
