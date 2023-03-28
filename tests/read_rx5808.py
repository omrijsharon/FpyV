import serial
import numpy as np
import matplotlib.pyplot as plt

def check_which_port():
    import serial.tools.list_ports
    ports = list(serial.tools.list_ports.comports())
    ports_list = []
    for p in ports:
        ports_list.append(p)
    return ports_list


def get_device_port(device_name: str):
    return [p.device for p in check_which_port() if device_name.lower() in p.description.lower()]


def read_serial_stream():
    port = get_device_port("Uno")[0]
    ser = serial.Serial(port, 9600, timeout=0.001)
    freq_list = [5865, 5845, 5825, 5805, 5785, 5765, 5745, 5725,
                 5733, 5752, 5771, 5790, 5809, 5828, 5847, 5866,
                 5705, 5685, 5665, 5645, 5885, 5905, 5925, 5945,
                 5740, 5760, 5780, 5800, 5820, 5840, 5860, 5880,
                 5658, 5695, 5732, 5769, 5806, 5843, 5880, 5917
                 ]
    freq_rssi_dict = {k: 0 for k in freq_list}
    # init for bar plot
    ax, fig = plt.subplots()
    # Read data
    try:
        buffer = []
        while True:
            ax.clear()
            waiting = ser.in_waiting  # find num of bytes currently waiting in hardware
            buffer += [chr(c) for c in ser.read(waiting)]  # read them, convert to ascii
            # ser.inWaiting()
            # line = ser.readline(64).decode('ascii')
            line = "".join(buffer)
            print(len(buffer))
            buffer *= 0
            # clean text
            line = line.split('\r\n')
            freq_line = [l.split(r"\r")[0] for l in line if "Frequency: " in l]
            # rssi_line = [l for l in line if "RSSI: " in l]
            if len(freq_line) > 0:
                try:
                    freq_line_list = [int(l.split(" MHz")[0].split("Frequency: ")[-1]) for l in freq_line]
                    rssi_line_list = [int(l.split(" dBm")[0].split("RSSI: ")[-1]) for l in freq_line]
                    for freq, rssi in zip(freq_line_list, rssi_line_list):
                        freq_rssi_dict[freq] = rssi
                except:
                    print(freq_line)
            plt.bar(freq_rssi_dict.keys(), freq_rssi_dict.values())
            plt.ylim(0, 20)
            plt.pause(0.000001)
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
        ser.close()
        plt.close(fig)


if __name__ == '__main__':
    read_serial_stream()
