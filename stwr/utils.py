import csv


def read_config_experiments(path):
    with open(path, 'r', encoding='utf-8') as fh:
        configs = list(csv.DictReader(fh, delimiter='\t'))

    return configs