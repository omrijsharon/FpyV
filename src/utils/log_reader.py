import numpy as np
from orangebox import Parser
import pandas as pd


def blackbox_parser(path):
    """
    Parse a cleanflight (iNav/Betaflight) blackbox log file
    :param path: path to the blackbox log file
    :return: tuple of (drone's configuration, flight log data)
    """
    parser = Parser.load(path)
    data = pd.DataFrame(columns=parser.field_names)
    for frame in parser.frames():
        frame_data = np.empty(shape=(len(parser.field_names),))
        frame_data.fill(np.nan)
        frame_data[:len(frame.data)] = frame.data
        frame_data = pd.DataFrame({field_name: [d] for field_name, d in zip(parser.field_names, frame_data)})
        data = data.append(frame_data, ignore_index=True)
    return data


if __name__ == '__main__':
    path = r'C:\Users\omri_\OneDrive\Documents\fpv_black_box\Sector5_20220206_164033\BTFL_BLACKBOX_LOG_Sector5_20220206_164033.BBL'
    # parser = Parser.load(path)
    data = blackbox_parser(path)
    print()