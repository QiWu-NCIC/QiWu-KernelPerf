#!/usr/bin/env python3
"""Download Matrix Market archives from a dataset manifest.

Archives are unpacked into ``<root>/<name>/<name>.mtx``, which is the layout
used by the KernelPerf dataset loader.  Only Matrix Market files are copied
from an archive; archive paths are never written directly to the destination.
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import tarfile
import tempfile
import time
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
    destination = root / name / f"{name}.mtx"
    if destination.is_file() and not force:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(str(item["source_url"]), headers={"User-Agent": "QiWu-KernelPerf/1.0"})
    for attempt in range(retries + 1):
        try:
            with opener.open(request, timeout=180) as response:
                payload = response.read()
            break
        except Exception:
            if attempt == retries:
                raise
            time.sleep(min(2**attempt, 30))
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        member = _matrix_member(archive, name)
        extracted = archive.extractfile(member)
        if extracted is None:
            raise RuntimeError(f"cannot read {member.name} from {item['source_url']}")
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as temporary:
            temporary_path = Path(temporary.name)
            shutil.copyfileobj(extracted, temporary)
    temporary_path.replace(destination)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--proxy", help="HTTP proxy URL, e.g. http://127.0.0.1:17890")
    parser.add_argument("--limit", type=int, default=0, help="download only the first N entries")
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not isinstance(manifest, list):
        raise SystemExit("manifest must be a JSON array")
    handlers = [ProxyHandler({"http": args.proxy, "https": args.proxy})] if args.proxy else []
    opener = build_opener(*handlers)
    entries = manifest[: args.limit] if args.limit > 0 else manifest
    args.root.mkdir(parents=True, exist_ok=True)
    for index, item in enumerate(entries, 1):
        output = download_item(opener, item, args.root, args.force, args.retries)
        print(f"[{index}/{len(entries)}] {item['matrix_id']} -> {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
