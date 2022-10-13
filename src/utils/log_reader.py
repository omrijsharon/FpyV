import numpy as np
from orangebox import Parser
from tree import map_structure


def blackbox_parser(path):
    """
    Parse a cleanflight (iNav/Betaflight) blackbox log file
    :param path: path to the blackbox log file
    :return: tuple of (drone's configuration, flight log data)
    """
    parser = Parser.load(path)
    data = {name: np.array([]) for name in parser.field_names}
    for frame in parser.frames():
        frame_data = np.empty(shape=(len(parser.field_names),))
        frame_data.fill(np.nan)
        frame_data[:len(frame.data)] = frame.data
        frame_data = {field_name: d for field_name, d in zip(parser.field_names, frame_data)}
        data = map_structure(lambda x, y: np.append(x, y), data, frame_data)
    return parser.headers, data


if __name__ == '__main__':
    path = r'G:\Users\omrijsharon\Documents\fpv\blackbox_logs\Sector5_20220206_164033\BTFL_BLACKBOX_LOG_Sector5_20220206_164033.BBL'
    # parser = Parser.load(path)
    data = blackbox_parser(path)
    print()