"""Fetch only the files a form20_dler.py entry is still missing.

Lists links exactly as the downloader entry does (nothing is downloaded while
listing), skips files already in data/pdf/ls/2024/<state>/, and saves the rest
with the downloader's own link_saver, a few at a time. A failed file is logged
and skipped instead of stopping the run. Run from the repo root:

    uv run python form20s/fetch_missing.py dl_maharashtra_2024 "_incoming/maharashtra" [concurrency]
"""

import asyncio
import os
import sys
import time

from httpx import AsyncClient, Limits

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import form20_dler as d  # noqa: E402


def log(msg):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


async def main(entry_name, state, concurrency):
    entry, real_saver = getattr(d, entry_name), d.link_saver
    picked = []

    async def record(session, st, url, fname=None):
        picked.append((url, fname))

    d.link_saver = record
    await entry()
    d.link_saver = real_saver

    folder = f"data/pdf/ls/2024/{state}"
    os.makedirs(folder, exist_ok=True)
    todo = [(u, f) for u, f in picked if not os.path.exists(f"{folder}/{f or u.split('/')[-1]}")]
    log(f"{state}: {len(picked)} links, {len(todo)} missing")

    sem = asyncio.Semaphore(concurrency)
    async with AsyncClient(headers=d.headers, timeout=600, verify=False, limits=Limits(max_connections=concurrency)) as session:

        async def one(u, f):
            async with sem:
                try:
                    await real_saver(session, state, u, f)
                    # link_saver silently skips non-200 responses (e.g. 404)
                    if not os.path.exists(f"{folder}/{f or u.split('/')[-1]}"):
                        raise Exception("not saved (non-200 response; see state_scraper.log)")
                    log(f"saved {f or u.split('/')[-1]}")
                except Exception as e:
                    log(f"FAILED {u}: {e!r}")
                    return u

        failed = [u for u in await asyncio.gather(*(one(u, f) for u, f in todo)) if u]
    log(f"DONE {state}: {len(os.listdir(folder))} of {len(picked)} files present; {len(failed)} failed this run")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 4))
