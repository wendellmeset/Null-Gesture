#!/usr/bin/env bash
set -e

# ── Install python-mercuryapi ──────────────────────────────────────────────
# This script builds the Python wrapper around ThingMagic's Mercury API C library.
#
# Prerequisites:
#   1. Go to https://novanta.com/precision-medicine/product/thingmagic-mercury-api/
#      (or https://www.jadaktech.com/documents-downloads/thingmagic-mercury-api-1-37-2/)
#   2. Download "ThingMagic® Mercury API BILBO" (the Software zip)
#   3. Place the .zip file in this directory (mercuryapi_src/)
#   4. Run this script
#
# If the filename is different from mercuryapi-BILBO-1.37.2.24.zip,
# update the APIZIP variable below.

cd "$(dirname "$0")"

APIZIP="mercuryapi-BILBO-1.37.2.24.zip"
APIVER="1.37.2.24"
SRC_DIR="mercuryapi_src"

if [ ! -d "$SRC_DIR" ]; then
    echo "Cloning python-mercuryapi..."
    git clone https://github.com/lefty01/python-mercuryapi.git "$SRC_DIR"
fi

cd "$SRC_DIR"

# Check if the zip exists (either in src dir or was passed in)
if [ ! -f "$APIZIP" ]; then
    # Try to find any mercuryapi zip in the parent directory
    PARENT_ZIP=$(ls ../mercuryapi-*.zip 2>/dev/null | head -1)
    if [ -n "$PARENT_ZIP" ]; then
        echo "Found zip: $PARENT_ZIP"
        ln -sf "$PARENT_ZIP" "$APIZIP"
    else
        echo ""
        echo "================================================================="
        echo " NEEDED: $APIZIP"
        echo "================================================================="
        echo ""
        echo "Download 'ThingMagic Mercury API BILBO' from:"
        echo "  https://novanta.com/precision-medicine/product/thingmagic-mercury-api/"
        echo ""
        echo "Then copy the zip file here:"
        echo "  $(pwd)/"
        echo ""
        echo "If the filename is different, update APIZIP in this script."
        echo ""
        exit 1
    fi
fi

echo "Building python-mercuryapi..."
make
echo ""
echo "Installing into virtualenv..."
PYTHON=$(which python3 || which python)
"$PYTHON" setup.py build install
echo ""
echo "Done! Test with: python -c 'import mercury; print(\"OK\")'"
