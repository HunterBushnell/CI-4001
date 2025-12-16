# local_controller.py
# Run on your laptop.
# Requires 'pyserial' installed:
#   pip install pyserial

import subprocess
import time
import serial

PORT = "COM5"  # <-- change for your system
BAUD = 115200

THRESHOLD = 0.65   # pick any threshold

# open USB to micro:bit
ser = serial.Serial(PORT, BAUD)
time.sleep(2)

# -------------- Step A + B (brain simulation) ----------
tA = time.time()
result = subprocess.check_output(["python3", "simulate_network.py"])
tB = time.time()

firing_rate = float(result.decode().strip())

# -------------- Decision  -----------------------------
if firing_rate > THRESHOLD:
    command = "GO"
else:
    command = "WIGGLE"

# -------------- Step C (muscles) -----------------------
tC_start = time.time()
ser.write((command + "\n").encode())
tC_end = time.time()

print("Firing Rate:", firing_rate)
print("Step A+B latency (ms):", (tB - tA) * 1000)
print("Step C latency (ms):", (tC_end - tC_start) * 1000)

ser.close()
