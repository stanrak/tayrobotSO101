"""Download the pinned, unmodified SO-101 model and verify its assets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "assets" / "robotstudio_so101"
COMMIT = "3abca2e302625cec9429c49806121ea2855cef81"
SOURCE = f"https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/{COMMIT}/robotstudio_so101"
MANIFEST = ROOT / "assets" / "model_manifest.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify() -> dict:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for name, expected in manifest["sha256"].items():
        path = DEST / name
        if not path.is_file() or digest(path) != expected:
            raise RuntimeError(f"Model integrity check failed: {path}. Run scripts/fetch_model.py to restore it.")
    return manifest


def fetch() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    expected = json.loads(MANIFEST.read_text(encoding="utf-8"))["sha256"] if MANIFEST.exists() else {}

    def download(name: str) -> None:
        path = DEST / name
        if name in expected and path.exists() and digest(path) == expected[name]:
            return
        with urllib.request.urlopen(f"{SOURCE}/{name}", timeout=60) as response:
            content = response.read()
        actual = hashlib.sha256(content).hexdigest()
        if name in expected and actual != expected[name]:
            raise RuntimeError(f"Downloaded asset hash mismatch: {name}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        print(f"Downloaded {name} ({len(content):,} bytes)", flush=True)

    names = ["so101.xml", "scene.xml", "LICENSE", "README.md"]
    for name in names:
        download(name)
    xml = ET.parse(DEST / "so101.xml")
    names += sorted({f"assets/{node.attrib['file']}" for node in xml.findall(".//mesh") if "file" in node.attrib})
    for name in names[4:]:
        download(name)
    manifest = {
        "repository": "https://github.com/google-deepmind/mujoco_menagerie",
        "commit": COMMIT,
        "model": "robotstudio_so101",
        "sha256": {name: digest(DEST / name) for name in names},
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    verify()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="Check local hashes without downloading anything.")
    args = parser.parse_args()
    if args.verify:
        verify()
        print("Pinned SO-101 model integrity verified.")
    else:
        fetch()
