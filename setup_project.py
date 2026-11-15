"""Install FireForecaster dependencies, including pinned upstream U-TAE.

Run with the Python interpreter/environment you will use for training:
    python setup_project.py
"""

from __future__ import annotations

import hashlib
from importlib import metadata
from pathlib import Path
import subprocess
import sys
import tempfile
from urllib.request import Request, urlopen


UPSTREAM_COMMIT = "987874e27a98bb399277f2d55eb010744d26f881"
PACKAGE_NAME = "fireforecaster-utae"
PACKAGE_VERSION = "0.1.1+987874e27a98"
UPSTREAM_FILES = {
    "src/backbones/utae.py": "c0f33cc875713cb348d25c4be8dac2435464bf81960c8288a7ed5f5718417f93",
    "src/backbones/ltae.py": "e358812f75eb1d74a5607a7d4bd28538241a5a215b0b67eefa59b7ea457bc36e",
    "src/backbones/positional_encoding.py": "b5a2498ebe76c0095bc5d155345324dc86443b0bb940b5189cffbe59e722fd7e",
    "src/backbones/convlstm.py": "9d501a1d59828f922a7827b76e2c797ac25471737f2b9d0e0516d0be52f890c6",
    "LICENSE": "ef8db3a2433282609ef4262e51899ddf7b7b0190d1244ee12cc99d16cde8e52c"
}
PROJECT_ROOT = Path(__file__).resolve().parent


def pip(*arguments: str) -> None:
    subprocess.run(
        [sys.executable, "-m", "pip", "--disable-pip-version-check", *arguments],
        check=True,
    )


def download(relative_path: str, expected_sha256: str) -> str:
    url = (
        "https://raw.githubusercontent.com/VSainteuf/utae-paps/"
        f"{UPSTREAM_COMMIT}/{relative_path}"
    )
    request = Request(url, headers={"User-Agent": "FireForecaster-setup"})
    with urlopen(request, timeout=60) as response:
        content = response.read()
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise RuntimeError(f"Upstream checksum mismatch: {relative_path}")
    return content.decode("utf-8")


def install_utae() -> None:
    try:
        if metadata.version(PACKAGE_NAME) == PACKAGE_VERSION:
            print(f"U-TAE {PACKAGE_VERSION} is already installed.", flush=True)
            return
    except metadata.PackageNotFoundError:
        pass

    # Downloads and wheel-building files are cleaned up on success or failure.
    with tempfile.TemporaryDirectory(prefix="fireforecaster-utae-") as temporary:
        root = Path(temporary)
        package = root / "utae"
        package.mkdir()
        for relative_path, digest in UPSTREAM_FILES.items():
            print(f"Fetching pinned U-TAE file: {relative_path}", flush=True)
            text = download(relative_path, digest)
            if relative_path == "LICENSE":
                (root / "LICENSE").write_text(text, encoding="utf-8")
                (package / "LICENSE").write_text(text, encoding="utf-8")
                continue
            # Package-local imports replace the upstream experiment repo's src namespace.
            text = text.replace("from src.backbones.", "from .")
            if relative_path.endswith("ltae.py"):
                text = text.replace(
                    "torch.stack([self.Q for _ in range(sz_b)], dim=1).view(",
                    "self.Q[:, None, :].expand(-1, sz_b, -1).reshape(",
                )
            (package / Path(relative_path).name).write_text(text, encoding="utf-8")

        (package / "__init__.py").write_text(
            'from .utae import UTAE\n\n__all__ = ["UTAE"]\n', encoding="utf-8"
        )
        (package / "UPSTREAM.md").write_text(
            "# U-TAE provenance\n\n"
            "Source: https://github.com/VSainteuf/utae-paps\n"
            f"Commit: {UPSTREAM_COMMIT}\n"
            "Copyright (c) 2021 VSainteuf. MIT license included.\n\n"
            "Changes: package-relative imports, an export in __init__.py, and\n"
            "tensor expansion instead of per-pixel attention-query copies.\n"
            "Model calculations are unchanged. This distribution is a local\n"
            "packaging wrapper, not an official upstream PyPI release.\n"
            "Paper: https://arxiv.org/abs/2107.07933\n",
            encoding="utf-8",
        )
        (root / "pyproject.toml").write_text(
            f'''[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "{PACKAGE_NAME}"
version = "{PACKAGE_VERSION}"
description = "Pinned upstream U-TAE packaged for FireForecaster"
requires-python = ">=3.10"
dependencies = ["torch>=2.0", "numpy>=1.24"]
license = {{file = "LICENSE"}}

[project.urls]
Upstream = "https://github.com/VSainteuf/utae-paps/tree/{UPSTREAM_COMMIT}"

[tool.setuptools]
packages = ["utae"]

[tool.setuptools.package-data]
utae = ["LICENSE", "UPSTREAM.md"]
''',
            encoding="utf-8",
        )
        wheels = root / "wheels"
        pip("wheel", "--no-deps", "--no-build-isolation", "--wheel-dir", str(wheels), str(root))
        wheel, = wheels.glob("*.whl")
        # Dependencies were installed by main(); preserve the existing CUDA build.
        pip("install", "--no-deps", str(wheel))


def main() -> None:
    print(f"Installing into Python: {sys.executable}", flush=True)
    pip("install", "-r", str(PROJECT_ROOT / "requirements.txt"))
    pip("install", "setuptools>=68", "wheel")
    install_utae()
    # Verify from outside the project, proving no local clone or path edit is needed.
    with tempfile.TemporaryDirectory(prefix="fireforecaster-import-check-") as check_dir:
        subprocess.run(
            [sys.executable, "-I", "-c", "from utae import UTAE; print('U-TAE import OK')"],
            cwd=check_dir,
            check=True,
        )
    print("Setup complete. Your model can use: from utae import UTAE", flush=True)


if __name__ == "__main__":
    main()
