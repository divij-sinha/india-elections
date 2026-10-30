import asyncio

from aiohttp import ClientSession

baseurl = "https://www.eci.gov.in/EBooks/ge_2024_results_ebook/files/mobile/{}.jpg"


async def page_saver(session: ClientSession, item: str | int | None = None) -> None:
    cur_url = baseurl.format(item)
    response = await session.get(cur_url, ssl=False)
    fname = cur_url.split("/")[-1]
    with open(f"data/pdf/resultbook/{fname}", "wb") as f:
        f.write(await response.read())
        print(f"Downloaded {fname}")

    return None


async def main():
    async with ClientSession(headers={"User-Agent": "curl/8.7.1"}) as session:
        tasks = [page_saver(session, i) for i in range(1, 163)]
        await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
