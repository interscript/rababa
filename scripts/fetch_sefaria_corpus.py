#!/usr/bin/env python3
"""Fetch pointed (nikud) Hebrew texts from Sefaria API.

The Tanakh (Hebrew Bible) is the most reliable fully-pointed Hebrew
source — ~400K words of text with complete vowel points. It's Biblical
Hebrew (older grammar than modern), but the diacritization patterns
transfer well enough for v0.5.0.

For v1.0.0, expand to:
  - Mishnah (some pointed)
  - Siddur / prayer books (fully pointed)
  - Piyyutim (religious poetry, often pointed)

Usage:
  python scripts/fetch_sefaria_corpus.py
  python scripts/fetch_sefaria_corpus.py --out data/sefaria-tanakh --books genesis,exodus
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

DEFAULT_OUT = Path(__file__).resolve().parent.parent / "data" / "sefaria-tanakh"

# Tanakh book refs (Sefaria API uses these slugs).
TANAKH_BOOKS = [
    # Torah (5)
    "Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy",
    # Nevi'im (Prophets, 8)
    "Joshua", "Judges", "I%20Samuel", "II%20Samuel",
    "I%20Kings", "II%20Kings", "Isaiah", "Jeremiah",
    "Ezekiel", "Hosea", "Joel", "Amos", "Obadiah", "Jonah",
    "Micah", "Nahum", "Habakkuk", "Zephaniah", "Haggai",
    "Zechariah", "Malachi",
    # Ketuvim (Writings, 11)
    "Psalms", "Proverbs", "Job", "Song%20of%20Songs",
    "Ruth", "Lamentations", "Ecclesiastes", "Esther",
    "Daniel", "Ezra", "Nehemiah", "I%20Chronicles", "II%20Chronicles",
]

# Mishnah — 63 tractates across 6 sedarim. Sefaria slug format: "Mishnah <Name>".
MISHNAH_TRACTATES = [
    # Seder Zeraim (11)
    "Berakhot", "Peah", "Demai", "Kilayim", "Sheviit", "Terumot",
    "Maaserot", "Maaser%20Sheni", "Hallah", "Orlah", "Bikkurim",
    # Seder Moed (12)
    "Shabbat", "Eruvin", "Pesachim", "Shekalim", "Yoma", "Sukkah",
    "Beitzah", "Rosh%20Hashanah", "Taanit", "Megillah", "Moed%20Katan", "Chagigah",
    # Seder Nashim (7)
    "Yevamot", "Ketubot", "Nedarim", "Nazir", "Sotah", "Gittin", "Kiddushin",
    # Seder Nezikin (10)
    "Bava%20Kamma", "Bava%20Metzia", "Bava%20Batra", "Sanhedrin", "Makkot",
    "Shevuot", "Avodah%20Zarah", "Horayot", "Eduyot", "Avot",
    # Seder Kodashim (11)
    "Zevachim", "Menachot", "Chullin", "Bechorot", "Arakhin",
    "Temurah", "Keritot", "Meilah", "Tamid", "Midot", "Kinnim",
    # Seder Tohorot (12)
    "Keilim", "Oholot", "Nega%27im", "Parah", "Tohorot", "Mikva%27ot",
    "Makhshirin", "Zavim", "Tevul%20Yom", "Yadayim", "Uktzin",
]
MISHNAH_BOOKS = [f"Mishnah%20{t}" for t in MISHNAH_TRACTATES]

# Siddurim — major rite families. Each is a large prayer book.
SIDDURIM_BOOKS = [
    "Siddur%20Ashkenaz",
    "Siddur%20Sefard",
    "Siddur%20Edot%20HaMizrach",
    "Siddur%20Rom",
]

# All pointed-Hebrew sources we know how to fetch.
ALL_BOOKS = TANAKH_BOOKS + MISHNAH_BOOKS + SIDDURIM_BOOKS


def fetch_chapter(book: str, chapter: int) -> str | None:
    """Fetch a single chapter's pointed Hebrew text via Sefaria API v3."""
    url = (
        f"https://www.sefaria.org/api/v3/texts/{book}.{chapter}"
        f"?version=hebrew&return_format=text_only"
    )
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        # v3 API returns actual text under .versions[0].text
        versions = data.get("versions") or []
        if not versions:
            return None
        text = versions[0].get("text")
        if isinstance(text, list):
            # Join per-verse list into a single line.
            return " ".join(str(v) for v in text if v)
        return str(text) if text else None
    except Exception as e:
        print(f"  ! {book}.{chapter}: {e}", file=sys.stderr)
        return None


def is_pointed(hebrew_text: str) -> bool:
    """True if text contains niqqud codepoints (Hebrew vowel points)."""
    # Hebrew vowel marks occupy 0x05B0..0x05BC and 0x05BD (meteg), 0x05C1..0x05C2 (sin/shin dots).
    return any("ְ" <= c <= "ֽ" or c in "ׁׂ" for c in hebrew_text)


def clean_html(s: str) -> str:
    """Strip HTML tags, Sefaria markers, and cantillation marks (trop).

    Cantillation (U+0591..U+05AF) tells you HOW to chant, not vowel
    pronunciation — Nakdimon-style models don't predict it, so strip.
    Keeps niqqud (U+05B0..U+05BC), sin/shin dots (U+05C1, U+05C2).
    """
    s = re.sub(r"<[^>]+>", "", s)              # HTML tags
    s = re.sub(r"\*\*\s*!!\s*\$\d+\$\s*!!\s*\*\*", "", s)  # Sefaria section markers
    # Strip cantillation marks (taamim).
    s = "".join(c for c in s if not ("֑" <= c <= "֯"))
    # Strip paragraph markers (׃ פ ס) used as verse separators in Tanakh.
    s = s.replace("׃", " ").replace(" פ ", " ").replace(" ס ", " ")
    return s.strip()


def fetch_book(book_slug: str, max_chapters: int = 200) -> list[str]:
    """Fetch all chapters of a book. Returns list of pointed Hebrew lines."""
    lines: list[str] = []
    for chapter in range(1, max_chapters + 1):
        text = fetch_chapter(book_slug, chapter)
        if text is None:
            break  # Book has no more chapters.
        text = clean_html(text)
        if not text or not is_pointed(text):
            continue
        # Split into sentence-ish chunks on common verse separators.
        for verse in re.split(r"[.·;]", text):
            verse = verse.strip()
            if len(verse) >= 20:  # filter very short fragments
                lines.append(verse)
        # Be polite — don't hammer the API.
        time.sleep(0.15)
    return lines


def split_lines(lines: list[str], seed: int = 42) -> dict[str, list[str]]:
    """80/10/10 split by line."""
    import random
    rng = random.Random(seed)
    shuffled = list(lines)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--books", default=",".join(ALL_BOOKS),
                   help="Comma-separated Sefaria book slugs (default: Tanakh+Mishnah+Siddurim)")
    p.add_argument("--max-chapters", type=int, default=200)
    p.add_argument("--include", choices=["tanakh", "mishnah", "siddurim", "all"], default="all",
                   help="Convenience selector: limit to one corpus subset")
    p.add_argument("--dry-run", action="store_true",
                   help="Fetch only Genesis chapter 1 to verify plumbing")
    args = p.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)

    # Apply --include convenience selector.
    if args.include == "tanakh":
        books = TANAKH_BOOKS
    elif args.include == "mishnah":
        books = MISHNAH_BOOKS
    elif args.include == "siddurim":
        books = SIDDURIM_BOOKS
    else:
        books = args.books.split(",")

    if args.dry_run:
        books = ["Genesis"]
        text = fetch_chapter("Genesis", 1)
        if text:
            print("Genesis 1 (first 200 chars):")
            print(clean_html(text)[:200])
        return 0

    all_lines: list[str] = []
    for book in books:
        book = book.strip()
        print(f"Fetching {book}...", flush=True)
        lines = fetch_book(book, args.max_chapters)
        print(f"  → {len(lines)} pointed lines")
        all_lines.extend(lines)

    print(f"\nTotal pointed lines: {len(all_lines)}")
    if len(all_lines) < 1000:
        print("WARNING: too few lines for training. Check API access.", file=sys.stderr)

    splits = split_lines(all_lines)
    for split_name, lines in splits.items():
        out_file = args.out / f"{split_name}.txt"
        out_file.write_text("\n".join(lines), encoding="utf-8")
        print(f"  {split_name}.txt: {len(lines)} lines → {out_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
