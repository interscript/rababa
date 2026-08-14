#!/usr/bin/env python3
"""Fetch the FULL Tashkeela corpus from SourceForge.

Two archives cover the complete corpus:
  1. avcorpus.tar.bz2 — classical Arabic books (Shamela Library, ~75M words)
  2. Tashkeela-arabic-diacritized-text-utf8-0.3.zip — MSA subset (~2M words)

Output: data/tashkeela-raw/ — one flat dir with extracted text files,
ready to be fed to scripts/clean_tashkeela_sadeed.py --tree.

For each .htm file in avcorpus we extract <body> text, strip HTML tags.
For each .htm.txt file in v0.3 we keep as-is (already plain text).

Usage:
  python scripts/fetch_tashkeela_full.py --out-dir data/tashkeela-raw

Cite: T. Zerrouki, A. Balla, "Tashkeela: Novel corpus of Arabic
vocalized texts, data for auto-diacritization systems", Data in Brief (2017).
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

AVCORPUS_URL = "https://sourceforge.net/projects/tashkeela/files/avcorpus.tar.bz2/download"
V03_URL = "https://sourceforge.net/projects/tashkeela/files/Tashkeela-arabic-diacritized-text-utf8-0.3.zip/download"

# HTML tag stripper — strips tags but preserves text content.
_TAG_RE = re.compile(r"<[^>]+>")
_SCRIPT_STYLE_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.DOTALL | re.IGNORECASE)
_WS_RE = re.compile(r"\s+")
_ENTITY_RE = re.compile(r"&#(\d+);")


def _decode_entities(text: str) -> str:
    return _ENTITY_RE.sub(lambda m: chr(int(m.group(1))), text)


def strip_html(html: str) -> str:
    """Crude but fast HTML→text: drop script/style, strip tags, decode entities."""
    html = _SCRIPT_STYLE_RE.sub("", html)
    # Body only — best effort, falls back to whole doc if no <body>.
    body_match = re.search(r"<body[^>]*>(.*?)</body>", html, re.DOTALL | re.IGNORECASE)
    if body_match:
        html = body_match.group(1)
    html = _TAG_RE.sub(" ", html)
    html = _decode_entities(html)
    html = _WS_RE.sub(" ", html).strip()
    return html


def download(url: str, dst: Path, expected_min_bytes: int = 1024 * 1024) -> None:
    """curl -L with resume support. Skips if dst already has the full file."""
    if dst.is_file() and dst.stat().st_size >= expected_min_bytes:
        print(f"  exists: {dst} ({dst.stat().st_size:,} bytes)")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {url}")
    subprocess.run(
        ["curl", "-sL", "--max-time", "1800", "-C", "-", "-o", str(dst), url],
        check=True,
    )
    print(f"  → {dst} ({dst.stat().st_size:,} bytes)")


def extract_avcorpus(archive: Path, out_dir: Path) -> int:
    """Extract .tar.bz2, write each .htm as a flat .txt under out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n_files = 0
    with tarfile.open(archive, "r:bz2") as tar:
        for member in tar.getmembers():
            if not member.isfile() or not member.name.endswith(".htm"):
                continue
            # Extract text content from HTML.
            f = tar.extractfile(member)
            if f is None:
                continue
            # Source files declare windows-1256, but content is actually UTF-8.
            raw = f.read()
            try:
                html = raw.decode("utf-8", errors="ignore")
            except Exception:
                continue
            text = strip_html(html)
            if not text:
                continue
            # Sanitize filename — use a hash to keep flat naming stable.
            import hashlib
            h = hashlib.sha256(member.name.encode("utf-8")).hexdigest()[:16]
            dst = out_dir / f"avcorpus-{h}.txt"
            dst.write_text(text, encoding="utf-8")
            n_files += 1
    return n_files


def extract_v03(archive: Path, out_dir: Path) -> int:
    """Extract .zip, copy each .htm.txt (and no-extension files) as flat .txt."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n_files = 0
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            # Skip doc/, toolz/, anything outside texts.txt/.
            if "texts.txt/" not in info.filename:
                continue
            try:
                raw = zf.read(info)
            except Exception:
                continue
            text = raw.decode("utf-8", errors="ignore").strip()
            if not text:
                continue
            import hashlib
            h = hashlib.sha256(info.filename.encode("utf-8")).hexdigest()[:16]
            dst = out_dir / f"v03-{h}.txt"
            dst.write_text(text, encoding="utf-8")
            n_files += 1
    return n_files


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out-dir", type=Path, required=True,
                   help="Output dir for extracted raw text files")
    p.add_argument("--cache-dir", type=Path,
                   default=Path(tempfile.gettempdir()) / "rababa-tashkeela-cache",
                   help="Where to cache downloaded archives")
    p.add_argument("--skip-avcorpus", action="store_true",
                   help="Skip the classical corpus (debug option)")
    p.add_argument("--skip-v03", action="store_true",
                   help="Skip the MSA v0.3 corpus (debug option)")
    args = p.parse_args(argv)

    print("=== Fetching FULL Tashkeela corpus ===\n")
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    total_files = 0

    if not args.skip_avcorpus:
        print("[1/2] Classical Arabic (avcorpus.tar.bz2)")
        av_archive = args.cache_dir / "avcorpus.tar.bz2"
        download(AVCORPUS_URL, av_archive, expected_min_bytes=100 * 1024 * 1024)
        n = extract_avcorpus(av_archive, args.out_dir / "avcorpus")
        print(f"  extracted: {n} classical books → {args.out_dir / 'avcorpus'}\n")
        total_files += n

    if not args.skip_v03:
        print("[2/2] Modern Standard Arabic (v0.3 zip)")
        v03_archive = args.cache_dir / "tashkeela-v03.zip"
        download(V03_URL, v03_archive, expected_min_bytes=100 * 1024 * 1024)
        n = extract_v03(v03_archive, args.out_dir / "v03")
        print(f"  extracted: {n} MSA files → {args.out_dir / 'v03'}\n")
        total_files += n

    print(f"Done. {total_files} files extracted under {args.out_dir}")
    print(f"\nNext: python scripts/clean_tashkeela_sadeed.py \\")
    print(f"          --in-dir {args.out_dir} --out-dir data/tashkeela-full --tree")
    return 0


if __name__ == "__main__":
    sys.exit(main())
