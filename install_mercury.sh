#!/usr/bin/env bash
set -e

# ── Install python-mercuryapi for Null-Gesture ────────────────────────────
#
# Option A — Pre-built wheel (Linux x86_64, Python 3.11):
#   pip install wheels/python_mercuryapi-*.whl
#
# Option B — Build from source (any platform):
#   1. Download the Mercury API C library zip from:
#      https://novanta.com/precision-medicine/product/thingmagic-mercury-api/
#      (free registration — click "ThingMagic Mercury API BILBO" under Software)
#   2. bash install_mercury.sh /path/to/mercuryapi-BILBO-1.37.x.xx.zip

cd "$(dirname "$0")"

ZIP_PATH="${1:-}"
SRC_DIR="mercuryapi_src"

if [ -f "venv/bin/python" ]; then
    PYTHON="venv/bin/python"
elif [ -f ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
else
    PYTHON=$(command -v python3 || command -v python)
fi

echo "Using Python: $PYTHON ($($PYTHON --version 2>&1))"

# Try pre-built wheel first if no zip provided
if [ -z "$ZIP_PATH" ] && [ -d "wheels" ]; then
    WHEEL=$(ls wheels/python_mercuryapi-*.whl 2>/dev/null | head -1)
    if [ -n "$WHEEL" ]; then
        echo "Found pre-built wheel: $WHEEL"
        $PYTHON -m pip install "$WHEEL"
        echo ""
        echo "✅ Done! Test with: $PYTHON -c 'import mercury; print(mercury.Reader)'"
        exit 0
    fi
fi

# Build from source
if [ ! -d "$SRC_DIR" ]; then
    echo "Cloning python-mercuryapi..."
    git clone https://github.com/lefty01/python-mercuryapi.git "$SRC_DIR"
fi

cd "$SRC_DIR"

if [ -n "$ZIP_PATH" ] && [ -f "$ZIP_PATH" ]; then
    ln -sf "$ZIP_PATH" "$(basename "$ZIP_PATH")"
fi

FOUND_ZIP=$(ls -1 mercuryapi-*.zip ../mercuryapi-*.zip 2>/dev/null | head -1)
if [ -z "$FOUND_ZIP" ]; then
    echo ""
    echo "================================================================="
    echo " NEED MERCURY API ZIP"
    echo "================================================================="
    echo ""
    echo "Download 'ThingMagic Mercury API BILBO' from:"
    echo "  https://novanta.com/precision-medicine/product/thingmagic-mercury-api/"
    echo ""
    echo "Then run:  bash install_mercury.sh /path/to/mercuryapi-BILBO-1.37.x.xx.zip"
    echo ""
    echo "Or if on Linux x86_64 with Python 3.11, use the pre-built wheel:"
    echo "  pip install wheels/python_mercuryapi-*.whl"
    echo ""
    exit 1
fi

ZIP_NAME=$(basename "$FOUND_ZIP")
ZIP_VER=$(echo "$ZIP_NAME" | sed 's/mercuryapi-BILBO-//; s/\.zip//')
CURRENT_VER=$(grep '^APIVER' Makefile | head -1 | cut -d?= -f2 | xargs)
if [ "$ZIP_VER" != "$CURRENT_VER" ]; then
    sed -i "s/APIZIP ?=.*/APIZIP ?= $ZIP_NAME/" Makefile
    sed -i "s/APIVER ?=.*/APIVER ?= $ZIP_VER/" Makefile
fi
if [ ! -f "$ZIP_NAME" ]; then
    ln -sf "$FOUND_ZIP" "$ZIP_NAME"
fi

echo "Building python-mercuryapi..."
make
$PYTHON setup.py build install

echo ""
echo "✅ Done! Test with: $PYTHON -c 'import mercury; print(mercury.Reader)'"
