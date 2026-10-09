#!/usr/bin/env python3
"""Download Matrix Market archives from a dataset manifest.

Archives are unpacked into ``<root>/<name>/<name>.mtx``, which is the layout
used by the KernelPerf dataset loader.  Only Matrix Market files are copied
from an archive; archive paths are never written directly to the destination.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tarfile
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath
from urllib.request import ProxyHandler, Request, build_opener


def _matrix_member(archive: tarfile.TarFile, name: str) -> tarfile.TarInfo:
    candidates = [member for member in archive.getmembers() if member.isfile()]
    exact = [member for member in candidates if PurePosixPath(member.name).name == f"{name}.mtx"]
    if exact:
        return exact[0]
    matrix_market = [member for member in candidates if member.name.lower().endswith(".mtx")]
    if len(matrix_market) == 1:
        return matrix_market[0]
    raise RuntimeError(f"archive does not contain an unambiguous .mtx file for {name}")


def download_item(opener, item: dict, root: Path, force: bool, retries: int) -> Path:
    name = str(item["name"])
    local_dir = str(item.get("local_dir") or name)
    destination = root / local_dir / f"{name}.mtx"
    if destination.is_file() and not force:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_url = str(item["source_url"])
    if source_url.startswith("https://sparse.tamu.edu/"):
        source_url = source_url.replace(
            "https://sparse.tamu.edu/", "http://sparse-files.engr.tamu.edu/", 1
        )
    request = Request(source_url, headers={"User-Agent": "QiWu-KernelPerf/1.0"})
    for attempt in range(retries + 1):
        archive_path: Path | None = None
        temporary_path: Path | None = None
        try:
            with opener.open(request, timeout=180) as response:
                with tempfile.NamedTemporaryFile(
                    dir=destination.parent, suffix=".tar.gz", delete=False
                ) as archive_file:
                    archive_path = Path(archive_file.name)
                    while chunk := response.read(1024 * 1024):
                        archive_file.write(chunk)
            with tarfile.open(name=archive_path, mode="r:gz") as archive:
                member = _matrix_member(archive, name)
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise RuntimeError(f"cannot read {member.name} from {item['source_url']}")
                with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as temporary:
                    temporary_path = Path(temporary.name)
                    shutil.copyfileobj(extracted, temporary)
            temporary_path.replace(destination)
            return destination
        except Exception:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            if archive_path is not None:
                archive_path.unlink(missing_ok=True)
            if attempt == retries:
                raise
            time.sleep(min(2**attempt, 30))
        finally:
            if archive_path is not None:
                archive_path.unlink(missing_ok=True)
    raise RuntimeError(f"download failed for {item['matrix_id']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--proxy", help="HTTP proxy URL, e.g. http://127.0.0.1:17890")
    parser.add_argument("--limit", type=int, default=0, help="download only the first N entries")
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
    if not isinstance(manifest, list):
        raise SystemExit("manifest must be a JSON array")
    handlers = [ProxyHandler({"http": args.proxy, "https": args.proxy})] if args.proxy else []
    opener = build_opener(*handlers)
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise SystemExit("shard-index must be within shard-count")
    entries = manifest[: args.limit] if args.limit > 0 else manifest
    entries = entries[args.shard_index :: args.shard_count]
    args.root.mkdir(parents=True, exist_ok=True)
    workers = max(1, args.workers)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(download_item, opener, item, args.root, args.force, args.retries): item
            for item in entries
        }
        failures = 0
        for index, future in enumerate(as_completed(futures), 1):
            item = futures[future]
            try:
                output = future.result()
            except Exception as error:
                failures += 1
                print(
                    f"[{index}/{len(entries)}] FAILED {item['matrix_id']}: "
                    f"{type(error).__name__}: {error}",
                    flush=True,
                )
                continue
            print(f"[{index}/{len(entries)}] {item['matrix_id']} -> {output}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
