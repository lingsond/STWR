# Run this script only after running the prepare_konvens.py script
# This script need the dataset created by the above script.

from tqdm import tqdm, trange


DATA_DIR = "C:/My Projects/Python/STWR/data/"
BASE_DIR = "C:/My Projects/Python/STWR/data/"
TARGET_DIR = BASE_DIR + "03_processed/Konvens2020/"


def create_each():
    sets = ['direct', 'indirect', 'reported']
    for set in sets:
        tasks = ['train', 'dev', 'test']
        for task in tasks:
            filename = TARGET_DIR + task + '.txt'
            with open(filename, 'r', encoding='utf-8') as fh:
                page = fh.readlines()[1:]

            content = ["-DOCSTART- -X- -X- O"]
            for i, line in enumerate(tqdm(page[1:])):
                if line.strip() == '':
                    content.append('')
                    continue

                text = line.strip().split('\t')
                if set == 'direct':
                    other = ['B-IND', 'I-IND', 'B-REP', 'I-REP']
                elif set == 'indirect':
                    other = ['B-DIR', 'I-DIR', 'B-REP', 'I-REP']
                else:
                    other = ['B-DIR', 'I-DIR', 'B-IND', 'I-IND']
                if text[-1] in other:
                    text[-1] = 'O'

                content.append('\t'.join(text))

            filename = TARGET_DIR + set + '/' + task + '.txt'
            with open(filename, 'w', encoding='utf-8') as fh:
                for line in content:
                    fh.write(line + '\n')


def main():
    create_each()


if __name__ == '__main__':
    main()
