# Null-Gesture

Gesture-detecting program using neural networks, IMU, ESP32, and RFID.

## Setup

### 1. Python environment

```bash
python3 -m venv venv
source venv/bin/activate       # Linux/macOS
# or: venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

---

### 2. RFID Reader (M7E Hecto): Platform Guides

The project uses a SparkFun Simultaneous RFID Reader - M7E Hecto. The Python driver
requires ThingMagic's Mercury API C library (free download, registration required).

**Download the library first (all platforms):**

1. Go to https://novanta.com/precision-medicine/product/thingmagic-mercury-api/
2. Click "ThingMagic Mercury API BILBO" under Software (free registration required)
3. Save the zip file (e.g. `mercuryapi-BILBO-1.37.x.xx.zip`)

Then follow your platform below.

---

#### Linux

**Option A - Pre-built wheel (fastest, same platform only):**

```bash
pip install mercuryapi_src/dist/python_mercuryapi-*.whl
```

**Option B - Build from source:**

```bash
# Prerequisites
sudo apt-get install unzip patch xsltproc gcc libreadline-dev python3-dev

# Install using the downloaded zip
bash install_mercury.sh /path/to/mercuryapi-BILBO-1.37.x.xx.zip
```

---

#### macOS

```bash
# Prerequisites
xcode-select --install        # Install Xcode Command Line Tools

# Clone the repo and build
git clone https://github.com/lefty01/python-mercuryapi.git mercuryapi_src
cd mercuryapi_src

# Copy the OS X patch (overwrites the Linux patch)
cp mercuryapi_osx.patch mercuryapi.patch

# Place the Mercury API zip in this directory
# (the one you downloaded from Novanta/Jadak)
# e.g. mercuryapi-BILBO-1.37.x.xx.zip

# Build
make

# Install
python3 setup.py build install
cd ..
```

---

#### Windows

**Option A - Pre-built installer (easiest):**

1. Download the latest Windows installer from:
   https://github.com/gotthardp/python-mercuryapi/releases
2. Run the `.exe` installer for your Python version

**Option B - Build from source (advanced):**

1. Download the Mercury API zip (see above)
2. Download [pthreads-win32](https://sourceforge.net/projects/pthreads4w/files/pthreads-w32-2-9-1-release.zip/download)
3. Follow the detailed build instructions in `mercuryapi_src/README.md` under "Build Instructions → Windows"
4. Requires Visual Studio 2017+ with C++ tools and Python extensions

---

#### Verify the installation (all platforms):

```bash
python -c "import mercury; print(mercury.Reader)"
# Should output: <class 'mercury.Reader'>
```

---

### 3. ESP32 IMU Reader

Reads IMU data from the ESP32 over serial and broadcasts it via TCP.

```bash
python esp32_reader.py
```

---

### 4. RFID Tag Scanner

```bash
# Test connection (detects reader, prints info, trial scan)
./rfid_data_reader.py --test

# Continuous scan with CSV logging
./rfid_data_reader.py
```

### 5. Troubleshooting

**Reader not found / timeout on first run:**

If the reader was previously used by the Universal Reader Assistant (URA),
it may be stuck in streaming mode. Unplug and replug the USB-C cable,
or run the script again (it will auto-detect and stop streaming).

**USB power:**

The M7E draws up to 700mA. If using a laptop USB port, you may need a
powered USB hub or external 5V supply for reliable operation.

**UART switch:**

The board has a switch labeled "UART". Make sure it's in the **USB** position
(when using USB-C), not **SER**.
