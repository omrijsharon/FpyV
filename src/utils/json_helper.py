import json


def json_writer(dict_to_write, full_path):
    with open(full_path, 'w', encoding='utf-8') as f:
        json.dump(dict_to_write, f, ensure_ascii=False, indent=4)


def json_reader(path):
    with open(path) as f:
        data = json.load(f)
    return data