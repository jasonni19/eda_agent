"""Retrieve pinned public benchmark files. No remote code is imported or executed."""
import hashlib
from pathlib import Path
import urllib.request

REVISION = "3b04cdba5f8f9433bb135dfc3be2ef5c64ff5179"
ROOT = Path(__file__).resolve().parents[1] / "benchmark_models" / "california-housing"
EXPECTED = {
    "config.json": "54123721e71f830ecf3891fcaffd85c68cf06e68fa66b5b4bb8179924093cf30",
    "model.py": "916138de82263a7d7f21e6fe87c797d279f137fd417ab4b127abd1d6a4359cbe",
    "README.md": "2453b5ee122c96490def145babb9f039220547645c48766bec99f197873afa07",
    "LICENSE": "4f981de7b3e6fa60f33796a05441986327dd0cc186f31e1e3ea4c9f69f773fac",
    "xgb.ubj": "cc2160311b6ff9b338997e541a632895549e12921ba40b73b040a3d8d0ee5a10",
    "source_app.py": "9495955e67c7578d4ec57f1b852b985194ba19557b175d30de5f5ff291d84ed9",
}


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    for name, expected in EXPECTED.items():
        target = ROOT / name
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == expected:
            print(name, "verified")
            continue
        url = ("https://huggingface.co/spaces/shalev396/california-housing/resolve/main/app.py" if name == "source_app.py" else
               f"https://huggingface.co/shalev396/california-housing/resolve/{REVISION}/{name}")
        with urllib.request.urlopen(url, timeout=60) as response:
            content = response.read(20 * 1024 * 1024 + 1)
        if len(content) > 20 * 1024 * 1024 or hashlib.sha256(content).hexdigest() != expected:
            raise ValueError(f"Source changed or download invalid: {name}. No file replaced.")
        target.write_bytes(content)
        print(name, "downloaded and verified")


if __name__ == "__main__":
    main()
