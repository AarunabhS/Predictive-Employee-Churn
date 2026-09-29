"""Restore the documented public input, verifying both upstream and converted hashes."""
from __future__ import annotations
import hashlib
import io
import re
import urllib.request
import zipfile
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch():
    target = ROOT / "data" / TARGET
    if target.exists():
        if sha256(target.read_bytes()) != TARGET_SHA256:
            raise ValueError("Existing input differs from the documented sample. It was left untouched; use --data with analysis.py for custom inputs.")
        print(f"Verified existing input: {target.relative_to(ROOT)}")
        return
    request = urllib.request.Request(URL, headers={"User-Agent": "portfolio-reproducibility/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        content = response.read()
    if sha256(content) != SOURCE_SHA256:
        raise ValueError("Upstream checksum mismatch; no input file was written. Review the publisher's version before updating the expected hash.")
    converted = convert(content)
    if sha256(converted) != TARGET_SHA256:
        raise ValueError("Converted checksum differs; no input file was written. Use the tested requirements.txt versions.")
    target.parent.mkdir(parents=True, exist_ok=True)
    # Never replace an existing file, even if it appeared during the download.
    with target.open("xb") as stream:
        stream.write(converted)
    print(f"Downloaded and verified: {target.relative_to(ROOT)}")


TARGET = 'hr.csv'
URL = 'https://www.kaggle.com/api/v1/datasets/download/liujiaqi/hr-comma-sepcsv'
SOURCE_SHA256 = '8f8e9aec8a8239b30844b0e1abe9353ccc14422182aa75764d25dc4df0edded1'
TARGET_SHA256 = '2510e274a90547f34c7b0db5a4ab70282c2710eb54252f14921bf980b81a928c'

def convert(content):
    return zipfile.ZipFile(io.BytesIO(content)).read("HR_comma_sep.csv")

if __name__ == "__main__":
    try:
        fetch()
    except (ValueError, OSError) as error:
        raise SystemExit(f"Download error: {error}")
