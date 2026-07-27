#!/usr/bin/env python3
"""Cross-platform installer for python-mercuryapi (M7E RFID reader Python wrapper).

Works on Linux, macOS, and Windows.

Strategy (in order of preference):
  1. Pre-built wheel — if Python 3.11 on Linux x86_64, installs instantly.
  2. Source build — clones python-mercuryapi, builds the C API, compiles the
     Python extension. Requires the Mercury API zip from Novanta/Jadak.
  3. Manual instructions — printed if the zip is missing.

Usage::

    python Readers/mercury/install.py
    python Readers/mercury/install.py --zip mercuryapi-BILBO-1.37.3.29-1.zip

Windows notes:
    Requires Visual Studio Build Tools (or full VS) with C++ workload.
    The script detects VS automatically via ``where vswhere``.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Optional

# ── Paths ───────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_WHEEL = _HERE / "python_mercuryapi-0.5.4-cp311-cp311-linux_x86_64.whl"
_REPO_URL = "https://github.com/lefty01/python-mercuryapi.git"

# Mercury API version — update if you get a newer zip
_API_VER = "1.37.3.29"


# ═════════════════════════════════════════════════════════════════════════════
#  Step 1: try the pre-built wheel
# ═════════════════════════════════════════════════════════════════════════════

def _wheel_compatible() -> bool:
    """The wheel is built for Python 3.11, Linux x86_64."""
    return (
        sys.platform == "linux"
        and platform.machine() in ("x86_64", "amd64")
        and sys.version_info[:2] == (3, 11)
    )


def _install_wheel() -> bool:
    if not _WHEEL.exists():
        return False
    if not _wheel_compatible():
        print(f"  Wheel is for Python 3.11 Linux x86_64, but you have "
              f"Python {sys.version_info.major}.{sys.version_info.minor} "
              f"on {platform.machine()}. Skipping wheel.")
        return False
    print("  → Installing pre-built wheel (Python 3.11 / Linux x86_64) …")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", str(_WHEEL)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return True


# ═════════════════════════════════════════════════════════════════════════════
#  Step 2: build from source
# ═════════════════════════════════════════════════════════════════════════════

def _find_zip(search_dir: Path | None = None) -> Path | None:
    """Look for a mercuryapi-*.zip file."""
    candidates: list[Path] = []
    dirs: list[Path] = [search_dir] if search_dir else []
    dirs.append(_HERE)           # Readers/mercury/  (bundled zip)
    dirs.append(_HERE.parent)    # Readers/
    dirs.append(Path.cwd())

    for d in dirs:
        if d is None or not d.exists():
            continue
        for pat in ("mercuryapi-*.zip", "mercuryapi*.zip", "*Mercury*API*.zip"):
            candidates.extend(d.glob(pat))

    return candidates[0] if candidates else None


def _find_cmake_windows() -> Optional[str]:
    """Find cmake on Windows."""
    for name in ("cmake", "cmake.exe"):
        p = shutil.which(name)
        if p:
            return p
    # Check Visual Studio bundled cmake
    for root in (os.environ.get("ProgramFiles", "C:\\Program Files"),
                 os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)")):
        for sub in ("CMake", "Microsoft Visual Studio"):
            base = Path(root) / sub
            if base.exists():
                for cm in base.rglob("bin/cmake.exe"):
                    return str(cm)
    return None


def _build_windows(api_dir: Path) -> None:
    """Build on Windows using Visual Studio + cmake."""
    src_dir = api_dir / "c" / "src" / "api"
    if not src_dir.exists():
        raise FileNotFoundError(f"Source not found: {src_dir}")

    cmake = _find_cmake_windows()
    if cmake is None:
        # Fall back to setup-win.py if cmake isn't available
        _build_windows_setuppy(api_dir)
        return

    build_dir = api_dir / "build"
    build_dir.mkdir(exist_ok=True)

    subprocess.check_call(
        [cmake, str(src_dir), "-DCMAKE_BUILD_TYPE=Release"],
        cwd=str(build_dir),
    )
    subprocess.check_call(
        [cmake, "--build", ".", "--config", "Release"],
        cwd=str(build_dir),
    )

    # Now build the Python extension
    env = os.environ.copy()
    env["MERCURYAPI_DIR"] = str(api_dir)
    subprocess.check_call(
        [sys.executable, "setup-win.py", "build_ext", "--inplace"],
        cwd=str(api_dir.parent) if api_dir.parent.name == "python-mercuryapi"
        else str(api_dir),
        env=env,
    )


def _build_windows_setuppy(api_dir: Path) -> None:
    """Fallback: use setup-win.py directly."""
    import shutil
    repo_dir = api_dir.parent  # python-mercuryapi checkout

    # setup-win.py expects a specific mercuryapi version layout
    subprocess.check_call(
        [sys.executable, "setup-win.py", "build_ext", "--inplace"],
        cwd=str(repo_dir),
    )


def _build_unix(repo_dir: Path) -> None:
    """Build on Linux / macOS using the Makefile."""
    subprocess.check_call(["make"], cwd=str(repo_dir))
    subprocess.check_call(
        [sys.executable, "setup.py", "install"],
        cwd=str(repo_dir),
    )


def _build_source(zip_path: Path) -> bool:
    """Clone python-mercuryapi, unpack zip, build, install."""
    print(f"  → Using Mercury API zip: {zip_path.name}")

    with tempfile.TemporaryDirectory(prefix="mercury_build_") as tmp:
        tmp_dir = Path(tmp)
        repo_dir = tmp_dir / "python-mercuryapi"

        # 1. Clone
        print("  → Cloning python-mercuryapi …")
        subprocess.check_call(
            ["git", "clone", "--depth", "1", _REPO_URL, str(repo_dir)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

        # 2. Copy zip into repo dir
        dest_zip = repo_dir / zip_path.name
        shutil.copy2(zip_path, dest_zip)

        # 3. Update Makefile's APIZIP if needed
        makefile = repo_dir / "Makefile"
        if makefile.exists():
            text = makefile.read_text()
            # Point APIZIP at our zip
            if "APIZIP ?=" in text:
                text = text.replace(
                    text.split("APIZIP ?=")[1].split("\n")[0].strip(),
                    zip_path.name,
                )
            makefile.write_text(text)

        # 4. Build
        print("  → Building C library + Python extension …")
        if sys.platform == "win32":
            # Unpack and build
            with zipfile.ZipFile(dest_zip) as zf:
                zf.extractall(repo_dir)
            api_dir = next(repo_dir.glob("mercuryapi-*"))
            _build_windows(api_dir)
            # Install
            subprocess.check_call(
                [sys.executable, "setup-win.py", "install"],
                cwd=str(repo_dir),
            )
        else:
            _build_unix(repo_dir)

    return True


# ═════════════════════════════════════════════════════════════════════════════
#  Step 3: manual instructions
# ═════════════════════════════════════════════════════════════════════════════

def _print_manual() -> None:
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║  Mercury API zip not found.                                            ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                        ║
║  The Mercury API is proprietary software from Novanta / Jadak.         ║
║  You need to download it manually (free registration required):        ║
║                                                                        ║
║    https://novanta.com/precision-medicine/product/thingmagic-mercury-api/
║                                                                        ║
║  Or search for "ThingMagic Mercury API BILBO download".                ║
║                                                                        ║
║  After downloading the zip, run:                                       ║
║                                                                        ║
║    python Readers/mercury/install.py --zip mercuryapi-BILBO-XXX.zip    ║
║                                                                        ║
║  Or place the zip in the project root and run the installer.           ║
║                                                                        ║
║  Linux users: if you have Python 3.11, the pre-built wheel works       ║
║  without downloading the zip — just run the installer.                 ║
║                                                                        ║
╚══════════════════════════════════════════════════════════════════════════╝
""")


# ═════════════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    import argparse
    p = argparse.ArgumentParser(
        description="Install python-mercuryapi for the M7E RFID reader.")
    p.add_argument("--zip", dest="zip_path", metavar="PATH",
                   help="Path to mercuryapi-BILBO-*.zip")
    args = p.parse_args()

    print("Installing python-mercuryapi …")
    print(f"  Platform: {sys.platform} / {platform.machine()}")
    print(f"  Python:   {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")

    # 1. Try pre-built wheel
    if _install_wheel():
        _verify()
        return

    # 2. Find or accept zip
    zip_path = Path(args.zip_path) if args.zip_path else _find_zip()
    if zip_path is None:
        _print_manual()
        sys.exit(1)

    if not zip_path.exists():
        print(f"  Zip not found: {zip_path}")
        _print_manual()
        sys.exit(1)

    # 3. Build from source
    try:
        _build_source(zip_path.resolve())
        print("  Build complete.")
        _verify()
    except subprocess.CalledProcessError as e:
        print(f"\n  Build failed: {e}")
        if sys.platform == "win32":
            print("\n  Windows build requires Visual Studio with C++ tools.")
            print("  Install from: https://visualstudio.microsoft.com/downloads/")
            print("  (Select 'Desktop development with C++' workload)")
        elif sys.platform == "darwin":
            print("\n  macOS build requires Xcode Command Line Tools:")
            print("    xcode-select --install")
        else:
            print("\n  Linux build requires build tools:")
            print("    sudo apt install build-essential git  # Debian/Ubuntu")
            print("    sudo dnf install make gcc git         # Fedora")
        sys.exit(1)


def _verify() -> None:
    """Try importing mercury to confirm it works."""
    try:
        import mercury  # noqa: F401
        print("  ✅ Verified: `import mercury` succeeded.")
    except ImportError:
        print("  ⚠️  Install ran but `import mercury` failed. "
              "Check the output above.")


if __name__ == "__main__":
    main()
