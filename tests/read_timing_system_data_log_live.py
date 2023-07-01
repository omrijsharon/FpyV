import numpy as np
import serial
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Tuple
import binascii

from utils.port_selector import PortSelector

def compute_crc(data: str) -> int:
    crc = 0
    for char in data:
        crc ^= ord(char)
    return crc

def parse_message(message: str):
    if len(message) != 27:
        print("Invalid message length")
        return None, None, None

    header = message[0]
    data = message[1:25]
    crc_received = int(message[25:], 16)

    if header != '$':
        print("Invalid header")
        return None, None, None

    crc_calculated = compute_crc(data)
    if crc_calculated != crc_received:
        print(f"Data: {data}, CRC received: {crc_received:02X}, CRC calculated: {crc_calculated:02X}")
        return None, None, None

    timestamp = int(data[0:10])
    mac_address = ":".join([data[i:i + 2] for i in range(10, 22, 2)])
    rssi = -int(data[22:])

    return timestamp, mac_address, rssi


port_selector = PortSelector()
selected_port = port_selector.run()

ser = serial.Serial(selected_port, 115200, timeout=0.001)

# Initialize the data dictionary
data = defaultdict(lambda: np.empty((0, 2), dtype=int))

# Initialize the plot
fig, ax = plt.subplots()
i = 0
try:
    buffer = []
    while True:
        waiting = ser.in_waiting  # Find the number of bytes currently waiting in hardware
        buffer += [chr(c) for c in ser.read(waiting)]  # Read them, convert to ASCII
        line = "".join(buffer)
        line = line.split('\r\n')

        for entry in line:
            if entry.startswith("$") and len(entry) == 27:
                try:
                    timestamp, mac_address, rssi = parse_message(entry)
                    # print(mac_address, timestamp, rssi)

                    # Store data in the dictionary
                    data[mac_address] = np.vstack((data[mac_address], np.array([timestamp, rssi])))
                    # print the difference between the last two timestamps
                    # if len(data[mac_address]) > 1:
                    #     print(data[mac_address][-1][0] - data[mac_address][-2][0])
                    if i % 100 == 0:
                        ax.clear()
                        for mac, points in data.items():
                            x_values = points[:, 0]
                            y_values = points[:, 1]
                            ax.plot(x_values, y_values, label=mac)
                        ax.legend()
                        plt.pause(0.001)
                except ValueError:
                    print("Invalid message")

        buffer = []

except KeyboardInterrupt:
    ser.close()
    plt.close('all')