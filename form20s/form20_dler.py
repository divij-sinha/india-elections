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
        tasks = [link_saver(session, state, url.format(item)) for item in items]
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
    elif state == "kerala":
        await dl_kerala()
    elif state == "andhrapradesh":
        await dl_andhrapradesh()
    elif state == "westbengal":
        await dl_westbengal()
    elif state == "bihar":
        await dl_bihar()
    elif state == "himanchalpradesh":
        await dl_himanchalpradesh
    elif state == "manipur":
        await dl_manipur
    elif state == "meghalaya":
        await dl_meghalaya
    elif state == "mizoram":
        await dl_mizoram
    elif state == "odisha":
        await dl_odisha
    elif state == "punjab":
        await dl_punjab
    else:
        raise NotImplementedError(f"State {state} not implemented")


dl_delhi = partial(
    dl_state_page_with_format,
    state="delhi",
    url="https://www.ceodelhi.gov.in/OnlineERMS/FORM_20_2024/PC{}.pdf",
    items=range(1, 8),
)

dl_kerala = partial(
    dl_state_page_with_format,
    state="kerala",
    url="https://www.ceo.kerala.gov.in/ceokerala/pdf/GE-2024/LAC_WISE_RESULTS/{}.pdf",
    items=[str(i).zfill(3) for i in range(1, 140)],
)

dl_mizoram = partial(
    dl_state_page_with_format,
    state="mizoram",
    url="https://ceo.mizoram.gov.in/storage/form_20/Form20-S16_ac_no{}.pdf",
    items=range(1, 40),
)

dl_odisha = partial(
    dl_state_page_with_format,
    state="odisha",
    url="https://ceoodisha.nic.in/repo/docs/form20_2024/loksabha/{}.pdf",
    items=[str(i).zfill(3) for i in range(1, 147)],
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

dl_punjab = partial(
    dl_state_pages_with_links,
    state="punjab",
    urls=[
        "https://ceopunjab.gov.in/pcform20?year=46"
    ],
    item_filter={"positive": [], "negative": []},
)

dl_westbengal = partial(
    dl_state_pages_with_links,
    state="westbengal",
    urls=[
        "https://ceowestbengal.nic.in/Candidate/164",
    ],
    item_filter={"positive": [], "negative": []},
    item_lambda=lambda item: "https://ceowestbengal.nic.in/" + item,
)

dl_andhrapradesh = partial(
    dl_state_pages_with_links,
    state="andhrapradesh",
    urls=[
        "https://ceoandhra.nic.in/ceoap_new/ceo/form20_ac.html",
    ],
    item_filter={"positive": [], "negative": []},
    item_lambda=lambda item: "https://ceoandhra.nic.in/ceoap_new/ceo/" + item,
)

dl_himanchalpradesh = partial(
    dl_state_pages_with_links,
    state="himanchalpradesh",
    urls=[
        "https://ceohimachal.nic.in/CommonControls/ViewCMSHtmlPage?qs=LdEF%2BAuYkB4BNe%2BBaDRW9Q%3D%3D",
    ],
    item_filter={"positive": ["form20"], "negative": []},
    http2=True,
)

dl_bihar = partial(
    dl_state_pages_with_links,
    state="bihar",
    urls=[
        "https://ceobihar.nic.in/GELS2024Form20_Part1_Part2.html"

    ],
    item_filter={"positive": [], "negative": []},
    http2=True,
)

dl_manipur = partial(
    dl_state_pages_with_links,
    state="manipur",
    urls=[
        "https://ceomanipur.nic.in/ResultSheets/LokSabhaElection/ge2024resultsheet"
    ],
    item_filter={"positive": ["form20"], "negative": []},
    item_lambda=lambda item: "https://ceomanipur.nic.in/ResultSheets/LokSabhaElection/" + item,
)

dl_meghalaya = partial(
    dl_state_pages_with_links,
    state="meghalaya",
    urls=[
        "https://ceomeghalaya.nic.in/electionconducted/ls2024/lok-sabha-2024.html",
    ],
    item_filter={"positive": ["form_20"], "negative": []},
    item_lambda=lambda item: "https://ceomeghalaya.nic.in/electionconducted/ls2024/" + item,
)

if __name__ == "__main__":
    # asyncio.run(dl_delhi())
    # asyncio.run(dl_karnataka())
    # asyncio.run(dl_tamilnadu())
    # asyncio.run(dl_telangana())
    # asyncio.run(dl_maharashtra())
    # asyncio.run(dl_assam())
    # asyncio.run(dl_westbengal())
    # asyncio.run(dl_andhrapradesh())
    # asyncio.run(dl_kerala())
    # asyncio.run(dl_bihar())
    # asyncio.run(dl_himanchalpradesh())
    # asyncio.run(dl_manipur())
    # asyncio.run(dl_meghalaya())
    # asyncio.run(dl_mizoram())
    # asyncio.run(dl_odisha())
    asyncio.run(dl_punjab())
    pass
