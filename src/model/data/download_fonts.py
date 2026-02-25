"""
download_fonts.py — Download OFL-licensed Google Fonts with both Latin and
Cyrillic (Russian) coverage.

Usage
-----
    python src/model/data/download_fonts.py [--output data/fonts]

The script queries the Google Fonts API, filters for fonts that:
  - Are licensed under the Open Font License (OFL / apache compatible)
  - Include the "cyrillic" subset (which covers Russian)

Downloaded font files are saved to data/fonts/<family>/<variant>.ttf.
This directory is gitignored — never commit font binaries.

Requirements
------------
    pip install fonttools requests

Environment variable (optional)
--------------------------------
    GOOGLE_FONTS_API_KEY   Your Google Fonts API key.
    If not set, the script falls back to direct GitHub raw downloads from
    the google/fonts repository.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import zipfile
from io import BytesIO
from pathlib import Path

# Gracefully require `requests` only at runtime.
try:
    import requests
except ImportError as e:
    raise ImportError("requests is required. Run: pip install requests") from e

try:
    from fontTools.ttLib import TTFont
except ImportError as e:
    raise ImportError("fonttools is required. Run: pip install fonttools") from e

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GOOGLE_FONTS_API_URL = "https://www.googleapis.com/webfonts/v1/webfonts"
GOOGLE_FONTS_GITHUB_ZIP = (
    "https://github.com/google/fonts/archive/refs/heads/main.zip"
)
# OFL identifier strings recognised in the license field.
OFL_IDENTIFIERS = {"ofl", "open font license"}

# Characters that must be present in the font (Latin style refs + all Cyrillic).
REQUIRED_LATIN = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
REQUIRED_CYRILLIC = list("АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя")
REQUIRED_CHARS = REQUIRED_LATIN + REQUIRED_CYRILLIC


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _font_has_required_coverage(font_path: Path) -> bool:
    """Return True if the font file contains every required glyph."""
    try:
        tt = TTFont(str(font_path), lazy=True)
        try:
            cmap = tt.getBestCmap()
            if cmap is None:
                return False
            return all(ord(ch) in cmap for ch in REQUIRED_CHARS)
        finally:
            tt.close()
    except Exception:
        return False


def _is_ofl(license_str: str) -> bool:
    low = license_str.lower()
    return any(ofl in low for ofl in OFL_IDENTIFIERS)


def _download_via_api(output_dir: Path, api_key: str) -> list[Path]:
    """Download eligible fonts using the Google Fonts REST API."""
    print("Querying Google Fonts API…")
    resp = requests.get(
        GOOGLE_FONTS_API_URL,
        params={"key": api_key, "subset": "cyrillic", "sort": "alpha"},
        timeout=30,
    )
    resp.raise_for_status()
    items = resp.json().get("items", [])
    print(f"  Found {len(items)} fonts with cyrillic subset from API.")

    downloaded: list[Path] = []
    for item in items:
        family = item.get("family", "Unknown")
        license_str = item.get("license", "")
        if not _is_ofl(license_str):
            continue  # Skip non-OFL fonts.

        family_dir = output_dir / re.sub(r"[^\w\-]", "_", family)
        family_dir.mkdir(parents=True, exist_ok=True)

        for variant, url in item.get("files", {}).items():
            dest = family_dir / f"{variant}.ttf"
            if dest.exists():
                downloaded.append(dest)
                continue
            try:
                r = requests.get(url, timeout=30)
                r.raise_for_status()
                dest.write_bytes(r.content)
                if _font_has_required_coverage(dest):
                    downloaded.append(dest)
                    print(f"  ✓ {family} ({variant})")
                else:
                    dest.unlink()  # Remove fonts lacking full coverage.
            except Exception as exc:
                print(f"  ✗ {family} ({variant}): {exc}", file=sys.stderr)

    return downloaded


def _download_via_github(output_dir: Path) -> list[Path]:
    """
    Fallback: download the google/fonts GitHub archive and extract OTF/TTF files
    from the ofl/ subtree that have both Latin and Cyrillic coverage.

    Note: The full archive is ~1.5 GB.  This is a one-time operation.
    """
    print("Downloading google/fonts GitHub archive (this may take a few minutes)…")
    resp = requests.get(GOOGLE_FONTS_GITHUB_ZIP, stream=True, timeout=120)
    resp.raise_for_status()

    data = BytesIO()
    total = 0
    for chunk in resp.iter_content(chunk_size=1 << 20):
        data.write(chunk)
        total += len(chunk)
        print(f"  Downloaded {total / 1e6:.1f} MB…", end="\r")
    print()

    downloaded: list[Path] = []
    with zipfile.ZipFile(data) as zf:
        font_entries = [
            n for n in zf.namelist()
            if re.search(r"/ofl/", n) and n.endswith((".ttf", ".otf"))
        ]
        print(f"  Extracting {len(font_entries)} OFL font files…")
        for entry in font_entries:
            parts = Path(entry).parts  # e.g. fonts-main/ofl/roboto/Roboto-Regular.ttf
            if len(parts) < 4:
                continue
            family_name = parts[2]
            file_name = parts[-1]
            dest_dir = output_dir / family_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / file_name
            if not dest.exists():
                dest.write_bytes(zf.read(entry))
            if _font_has_required_coverage(dest):
                downloaded.append(dest)
            else:
                dest.unlink()

    return downloaded


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Download OFL Google Fonts with Latin+Cyrillic coverage.")
    parser.add_argument(
        "--output",
        default="data/fonts",
        help="Directory to save downloaded fonts (default: data/fonts)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    api_key = os.environ.get("GOOGLE_FONTS_API_KEY", "")
    if api_key:
        fonts = _download_via_api(output_dir, api_key)
    else:
        print(
            "GOOGLE_FONTS_API_KEY not set — falling back to GitHub archive download.\n"
            "Set the environment variable for faster, targeted downloads:\n"
            "  export GOOGLE_FONTS_API_KEY=<your_key>\n"
            "  Get a key at: https://developers.google.com/fonts/docs/developer_api\n"
        )
        fonts = _download_via_github(output_dir)

    print(f"\n✅ Downloaded {len(fonts)} eligible font files to {output_dir}/")
    print("   These fonts have OFL license and full Latin+Cyrillic (Russian) coverage.")


if __name__ == "__main__":
    main()
