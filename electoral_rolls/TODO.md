# Electoral rolls downloader TODO

## A broken file blocks NOT_IN_ROLL for the rest of its batch (get_new.py)

When one file in a batch fails to download (e.g. Odisha Angul AC 61 part 37, AC 62 part 107 return non-JSON from
`get-published-file` every time), we can't tell which part it belonged to: for fileId downloads the part number only
comes back in the response. So `get_roll_files` doesn't trust any "missing from batch" parts in that batch and they're
logged as failures instead of getting a `NOT_IN_ROLL.txt` marker. They're re-requested on every rerun and inflate the
failure count in the ntfy pushes.

Seen 2026-10-30: batch [37, 249-252, 257-259] of AC 61, all logged as failures, though 249-259 aren't in the 2024 roll.

Planned fix: when a batch has a failed download, re-request each unresolved part on its own (one captcha each, only in
these rare batches). Single-part answers are unambiguous: `File not found` -> mark NOT_IN_ROLL, a file -> save it, a
failed download -> real failure pinned to that part.

## Memory: large transient buffers per file, heap never shrinks (get_new.py)

Not a true leak, but on a 3.7 GB / no-swap box RSS sits at ~1.5-1.9 GB (Odisha run, GEROLL_SEMAPHORE=4, BATCH_SIZE=8) and
a spike could get the download OOM-killed (nothing restarts it). Seen 2026-10-30: `[heap]` was 1.48 GB of 1.55 GB RSS;
glibc raises its mmap threshold after the first large free, so the 5-7 MB buffers land in the brk heap and freed holes
are never returned to the OS.

Where it comes from:
- `download_file` holds each ~5 MB PDF ~4 times at once (`response.content`, cached `.text`, the JSON `payload` str,
  decoded bytes) -> ~25 MB per file, up to 32 files in flight (4 batches x 8).
- `get_roll_files` / `save_parts` keep a batch's 8 decoded PDFs (~40 MB) until all are written to the CIFS mount.
- `solve_torch` (utils/captcha_solve.py) reloads the model on every captcha. No leak, just churn.

Planned fix:
- Start with `MALLOC_MMAP_THRESHOLD_=1048576 MALLOC_TRIM_THRESHOLD_=4194304` (pins the mmap threshold so big buffers are
  returned on free).
- `download_file`: `json.loads(response.content)` instead of `.text`, pop `payload` and drop the response before decoding.
- `save_parts`: clear the batch result after saving, `ctypes.CDLL("libc.so.6").malloc_trim(0)`.
- Cache the loaded torch model (`functools.cache`).
- If still tight: write each file as soon as it decodes instead of collecting the batch, or lower GEROLL_SEMAPHORE.
