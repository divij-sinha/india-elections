import asyncio
import logging
import os
from functools import partial
from typing import Iterable

from httpx import AsyncClient, HTTPError
from lxml import etree

logging.basicConfig(
    filename="state_scraper.log",
    encoding="utf-8",
    filemode="a",
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
    level=logging.INFO,
)

headers = {
    "Accept": "*/*",
    "User-Agent": "curl/8.7.1",
}


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


async def dl_state_page_with_format(
    state: str,
    url: str,
    items: Iterable,
    **kwargs: str,
) -> None:
    logging.info(f"Fetching {state}")
    os.makedirs(f"data/pdf/ls/2024/{state}", exist_ok=True)
    async with AsyncClient(
        base_url=url, headers=headers, timeout=300, verify=False, **kwargs
    ) as session:
        tasks = [link_saver(session, "delhi", url.format(item)) for item in items]
        await asyncio.gather(*tasks)
    logging.info(f"Fetched {state}")


async def dl_state_pages_with_links(
    state: str,
    urls: list[str] | str,
    item_filter: dict,
    item_lambda: callable = None,
    **kwargs: str,
) -> None:
    logging.info(f"Fetching {state}")
    os.makedirs(f"data/pdf/ls/2024/{state}", exist_ok=True)

    if isinstance(urls, str):
        urls = [urls]

    for url in urls:
        logging.info(f"Fetching {url}")
        async with AsyncClient(
            base_url=url, headers=headers, timeout=300, verify=False, **kwargs
        ) as session:
            response = await session.get(url, headers=headers)
            logging.debug(f"Fetched {url}")
            tree = etree.HTML(response.text)
            items = tree.xpath("//a/@href")
            items = [item for item in items if item.lower().endswith(".pdf")]
            for filter in item_filter["positive"]:
                items = [item for item in items if filter.lower() in item.lower()]
            for filter in item_filter["negative"]:
                items = [item for item in items if filter.lower() not in item.lower()]
            if item_lambda is not None:
                items = [item_lambda(item) for item in items]
            logging.info("Parsed items")
            tasks = [link_saver(session, state, item) for item in items]
            await asyncio.gather(*tasks)
        logging.info(f"Fetched {url} for {state}")


@retry_decorator
async def link_saver(session: AsyncClient, state: str, url: str) -> None:
    response = await session.get(url)
    if response.status_code == 200:
        fname = url.split("/")[-1]
        with open(f"data/pdf/ls/2024/{state}/{fname}", "wb") as f:
            f.write(response.read())
            logging.debug(f"Downloaded {fname}")


async def dl_state(state: str):
    if state == "delhi":
        await dl_delhi()
    elif state == "telangana":
        await dl_telangana()
    elif state == "maharashtra":
        await dl_maharashtra()
    elif state == "tamilnadu":
        await dl_tamilnadu()
    elif state == "karnataka":
        await dl_karnataka()
    elif state == "assam":
        await dl_assam()
    else:
        raise NotImplementedError(f"State {state} not implemented")


# async def fetch(session: ClientSession, url: str) -> ClientResponse | None:
#     response = await session.get(url, ssl=False)
#     if response.status == 200:
#         return response
#     else:
#         raise Exception(f"Failed to fetch {url} with status {response.status}")


# async def saver_helper(
#     session: ClientSession, state: str, url: str, item: str | int | None = None
# ) -> None:
#     if item is not None:
#         cur_url = url.format(item)
#     else:
#         cur_url = url
#     response = await fetch(session, cur_url)
#     fname = cur_url.split("/")[-1]
#     with open(f"data/pdf/ls/2024/{state}/{fname}", "wb") as f:
#         f.write(await response.read())
#         print(f"Downloaded {fname}")
#         # time.sleep(0.5)

#     return None


# async def dl_delhi():
#     url = "https://www.ceodelhi.gov.in/OnlineERMS/FORM_20_2024/PC{}.pdf"
#     os.makedirs("data/pdf/ls/2024/delhi", exist_ok=True)

#     async with ClientSession() as session:
#         tasks = [saver_helper(session, "delhi", url, i) for i in range(1, 8)]
#         await asyncio.gather(*tasks)


# async def dl_telangana():
#     url = "https://ceotelangana.nic.in/HOP-2024/INDEX_2024.html"
#     os.makedirs("data/pdf/ls/2024/telangana", exist_ok=True)

#     async with ClientSession(headers=headers) as session:
#         response = await fetch(session, url)
#         response_text = await response.text()
#         tree = etree.HTML(response_text)
#         items = tree.xpath("//a/@href")
#         urls = ["https://ceotelangana.nic.in/HOP-2024/" + item for item in items]
#         print("FETCHED PDF URLS")
#         tasks = [saver_helper(session, "telangana", url) for url in urls]
#         await asyncio.gather(*tasks)


# async def dl_maharashtra():
#     url = "https://ceoelection.maharashtra.gov.in/ceo/Form20-Part-II.aspx"
#     os.makedirs("data/pdf/ls/2024/maharashtra", exist_ok=True)

#     async with ClientSession(headers=headers) as session:
#         response = await fetch(session, url)
#         response_text = await response.text()
#         tree = etree.HTML(response_text)
#         items = tree.xpath("//a/@href")
#         urls = []
#         for item in items:
#             if item.lower().endswith(".pdf") and "form20" in item.lower():
#                 if "https" not in item:
#                     urls.append("https://ceoelection.maharashtra.gov.in/" + item)
#                 else:
#                     urls.append(item)
#         print("FETCHED PDF URLS")
#         tasks = [saver_helper(session, "maharashtra", url) for url in urls]
#         await asyncio.gather(*tasks)


# async def dl_tamilnadu():
#     urls = [
#         "https://www.elections.tn.gov.in/GELS2024_Form20.aspx",
#         "https://www.elections.tn.gov.in/GELS2024_Form20_Part2.aspx",
#     ]
#     os.makedirs("data/pdf/ls/2024/tamilnadu", exist_ok=True)

#     for url in urls:
#         async with ClientSession(headers=headers) as session:
#             response = await fetch(session, url)
#             response_text = await response.text()
#             tree = etree.HTML(response_text)
#             items = tree.xpath("//a/@href")
#             urls = []
#             for item in items:
#                 if item.lower().endswith(".pdf") and "form20" in item.lower():
#                     if "https" not in item:
#                         urls.append("https://www.elections.tn.gov.in/" + item)
#                     else:
#                         urls.append(item)
#             print("FETCHED PDF URLS")
#             tasks = [saver_helper(session, "tamilnadu", url) for url in urls]
#             await asyncio.gather(*tasks)


# async def dl_assam():
#     urls = [
#         "https://ceoassam.nic.in/lok_sabha/2024/form20-part-II-2024.html",
#         "https://ceoassam.nic.in/lok_sabha/2024/form20-Part_I_2024.html",
#     ]
#     os.makedirs("data/pdf/ls/2024/assam", exist_ok=True)

#     httpx_client = httpx.Client(http2=True, verify=False, timeout=30.0)
#     for url in urls:
#         async with ClientSession(
#             headers=headers,
#             skip_auto_headers={"X-Content-Security-Policy"},
#             trust_env=True,
#         ) as session:
#             print(session.headers)
#             response = httpx_client.get(url, headers=headers)
#             response_text = response.text
#             tree = etree.HTML(response_text)
#             items = tree.xpath("//a/@href")
#             urls = [
#                 "https://ceoassam.nic.in/lok_sabha/" + item.replace("..", "")
#                 for item in items
#                 if item.lower().endswith(".pdf") and "form20" in item.lower()
#             ]
#             print("FETCHED PDF URLS")
#             print(urls)
#             tasks = [saver_helper(session, "assam", url) for url in urls]
#             await asyncio.gather(*tasks)

dl_delhi = partial(
    dl_state_page_with_format,
    state="delhi",
    url="https://www.ceodelhi.gov.in/OnlineERMS/FORM_20_2024/PC{}.pdf",
    items=range(1, 8),
)

dl_karnataka = partial(
    dl_state_pages_with_links,
    state="karnataka",
    urls="https://ceo.karnataka.gov.in/327/_gallery_/en",
    item_filter={"positive": [], "negative": ["media_to_upload"]},
)

dl_assam = partial(
    dl_state_pages_with_links,
    state="assam",
    urls=[
        "https://ceoassam.nic.in/lok_sabha/2024/form20-part-II-2024.html",
        "https://ceoassam.nic.in/lok_sabha/2024/form20-Part_I_2024.html",
    ],
    item_filter={"positive": ["form20"], "negative": []},
    http2=True,
)

dl_telangana = partial(
    dl_state_pages_with_links,
    state="telangana",
    urls=[
        "https://ceotelangana.nic.in/HOP-2024/INDEX_2024.html",
    ],
    item_filter={"positive": [], "negative": []},
    item_lambda=lambda item: "https://ceotelangana.nic.in/HOP-2024/" + item,
)

dl_maharashtra = partial(
    dl_state_pages_with_links,
    state="maharashtra",
    urls=[
        "https://ceoelection.maharashtra.gov.in/ceo/Form20-Part-II.aspx",
    ],
    item_filter={"positive": [], "negative": []},
    http2=True,
)

dl_tamilnadu = partial(
    dl_state_pages_with_links,
    state="tamilnadu",
    urls=[
        "https://www.elections.tn.gov.in/GELS2024_Form20.aspx",
        "https://www.elections.tn.gov.in/GELS2024_Form20_Part2.aspx",
    ],
    item_filter={"positive": ["form20"], "negative": []},
    http2=True,
)

if __name__ == "__main__":
    # asyncio.run(dl_assam())
    # asyncio.run(dl_delhi())
    # asyncio.run(dl_karnataka())
    # asyncio.run(dl_tamilnadu())
    # asyncio.run(dl_telangana())
    asyncio.run(dl_maharashtra())
    pass
