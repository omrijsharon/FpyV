import yaml


def yaml_writer(path, data):
    with open(path, 'w') as f:
        yaml.dump(data, f)


def yaml_reader(path):
    with open(path) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return data


if __name__ == '__main__':
    path = r'C:\Users\omrijsharon\PycharmProjects\FpyV\config\params.yaml'
    data = yaml_reader(path)
    print(data)