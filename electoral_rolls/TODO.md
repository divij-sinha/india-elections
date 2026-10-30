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
