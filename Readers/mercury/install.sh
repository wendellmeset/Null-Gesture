#!/usr/bin/env bash
# Install python-mercuryapi (Linux / macOS)
# ==========================================
#
# Tries the pre-built wheel first (Python 3.11, Linux x86_64).
# Falls back to building from source using the Mercury API zip.
#
# Usage:
#   bash Readers/mercury/install.sh
#   bash Readers/mercury/install.sh /path/to/mercuryapi-BILBO-XXX.zip

set -euo pipefail
cd "$(dirname "$0")/../.."  # project root

WHEEL="Readers/mercury/python_mercuryapi-0.5.4-cp311-cp311-linux_x86_64.whl"
APIZIP="${1:-}"

# ── Python detection ─────────────────────────────────────────────────────

PYTHON=""
for p in python3 python; do
    if command -v "$p" &>/dev/null; then
        PYTHON="$p"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    echo "ERROR: Python not found." >&2
    exit 1
fi

PY_VER="$("$PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
ARCH="$("$PYTHON" -c 'import platform; print(platform.machine())')"
OS="$("$PYTHON" -c 'import sys; print(sys.platform)')"

echo "Installing python-mercuryapi …"
echo "  Platform: $OS / $ARCH"
echo "  Python:   $PY_VER"

# ── Step 1: try pre-built wheel ──────────────────────────────────────────

if [ "$OS" = "linux" ] && [ "$ARCH" = "x86_64" ] && [ "$PY_VER" = "3.11" ] && [ -f "$WHEEL" ]; then
    echo "  → Installing pre-built wheel (Python 3.11 / Linux x86_64) …"
    "$PYTHON" -m pip install "$WHEEL" && \
        "$PYTHON" -c 'import mercury; print("  ✅ `import mercury` succeeded.")' && \
        exit 0
    echo "  Wheel install failed, falling back to source build."
fi

# ── Step 2: find or accept Mercury API zip ───────────────────────────────

if [ -z "$APIZIP" ]; then
    # Search for any mercuryapi zip (bundled location first, then cwd)
    for dir in "Readers/mercury" "."; do
        for f in "$dir"/mercuryapi-*.zip "$dir"/mercuryapi*.zip; do
            if [ -f "$f" ]; then
                APIZIP="$f"
                break 2
            fi
        done
    done
fi

if [ -z "$APIZIP" ] || [ ! -f "$APIZIP" ]; then
    echo ""
    echo "╔══════════════════════════════════════════════════════════════════════════╗"
    echo "║  Mercury API zip not found.                                            ║"
    echo "╠══════════════════════════════════════════════════════════════════════════╣"
    echo "║                                                                        ║"
    echo "║  Download from Novanta / Jadak (free registration):                    ║"
    echo "║    https://novanta.com/precision-medicine/product/thingmagic-mercury-api/"
    echo "║                                                                        ║"
    echo "║  Then run:                                                             ║"
    echo "║    bash Readers/mercury/install.sh mercuryapi-BILBO-XXX.zip            ║"
    echo "║                                                                        ║"
    echo "╚══════════════════════════════════════════════════════════════════════════╝"
    exit 1
fi

echo "  → Using Mercury API zip: $APIZIP"

# ── Step 3: build from source ────────────────────────────────────────────

BUILDDIR="$(mktemp -d)"
trap 'rm -rf "$BUILDDIR"' EXIT

echo "  → Cloning python-mercuryapi …"
git clone --depth 1 https://github.com/lefty01/python-mercuryapi.git "$BUILDDIR" 2>/dev/null

# Copy zip into checkout
cp "$APIZIP" "$BUILDDIR"/mercuryapi-BILBO.zip

# Update Makefile APIZIP
sed -i "s|APIZIP ?=.*|APIZIP ?= mercuryapi-BILBO.zip|" "$BUILDDIR"/Makefile 2>/dev/null || true

echo "  → Building C library + Python extension …"
cd "$BUILDDIR"

# Try make; if zip version doesn't match, just unzip and try
make PYTHON="$PYTHON" 2>&1 || {
    echo "  Make failed, trying manual build …"
    unzip -o mercuryapi-BILBO.zip -x '*.dll' '*.exe' '*.pdb' 2>&1 || true
    API_DIR="$(ls -d mercuryapi-*/ 2>/dev/null | head -1)"
    if [ -n "$API_DIR" ]; then
        make -C "${API_DIR}c/src/api" 2>&1 || true
        mkdir -p build/mercuryapi/include build/mercuryapi/lib
        find "$API_DIR"c/src/api -name '*.h' ! -name '*_imp.h' ! -path '*ltkc_win32*' -exec cp {} build/mercuryapi/include/ \; 2>/dev/null || true
        find "$API_DIR"c/src/api -name '*.a' -o -name '*.so.1' -exec cp {} build/mercuryapi/lib/ \; 2>/dev/null || true
    fi
    "$PYTHON" setup.py build_ext --inplace 2>&1 || true
}

echo "  → Installing …"
"$PYTHON" setup.py install

cd - >/dev/null

echo "  ✅ Done."
"$PYTHON" -c 'import mercury; print("  ✅ `import mercury` succeeded.")'
