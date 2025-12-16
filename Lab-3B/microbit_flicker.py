import serial
import time

# --------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------
# Adjust this COM port to match your micro:bit (Windows example: COM5)
#   • On Windows: check in Device Manager → Ports (COM & LPT)
#   • On macOS/Linux: typically /dev/ttyACM0 or /dev/ttyUSB0
# --------------------------------------------------------------------
MICROBIT_PORT = "COM5"
BAUD_RATE = 115200

# --------------------------------------------------------------------
# 1️⃣  Read the oscillation frequency from file (written by controller)
# --------------------------------------------------------------------
try:
    with open("oscillation_frequency.txt", "r") as f:
        freq = float(f.read().strip())
except FileNotFoundError:
    print("❌  No oscillation_frequency.txt found — please copy it from Node1.")
    exit(1)

print(f"📡 Received oscillation frequency: {freq} Hz")

# --------------------------------------------------------------------
# 2️⃣  Compute blink interval
# --------------------------------------------------------------------
if freq <= 0:
    print("⚠️  Invalid frequency, using 1 Hz fallback")
    freq = 1.0
period = 1.0 / freq

# --------------------------------------------------------------------
# 3️⃣  Send blink commands to micro:bit
# --------------------------------------------------------------------
try:
    mb = serial.Serial(MICROBIT_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)  # wait for serial init
    print(f"💡 Flickering micro:bit at {freq:.2f} Hz")
    while True:
        mb.write(b"1\n")      # LED ON (micro:bit program interprets ‘1’ as on)
        time.sleep(period / 2)
        mb.write(b"0\n")      # LED OFF
        time.sleep(period / 2)
except Exception as e:
    print(f"❌  Error communicating with micro:bit: {e}")
finally:
    if 'mb' in locals():
        mb.close()
