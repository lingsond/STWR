import os
import pprint
import random
import numpy as np
from tqdm import tqdm, trange
from sklearn.model_selection import train_test_split

if os.name == 'nt':
    BASE_DIR = "C:/My Projects/Python/STWR/data/"
    RAW_DIR = BASE_DIR + "01_raw/kern_rich/"
    use_cuda = False
else:
    BASE_DIR = "/home/stud/wangsadirdja/STWR/data/"
    RAW_DIR = BASE_DIR + "01_raw/direct-speech/kern_rich/"
    use_cuda = True

INTERIM_DIR = BASE_DIR + "02_interim/kern_rich/"
TARGET_DIR = BASE_DIR + "03_processed/kern_rich/farm/"


def combine_raw_to_interim():
    tasks = ["train", "dev", "test"]
    # tasks = ['train']
    for task in tasks:
        # print(f"Processing files for {task} set...")
        raw_dir = RAW_DIR + task + '/'
        files = [raw_dir + file for file in os.listdir(raw_dir) if file.endswith('.tsv')]

        content = ["-DOCSTART- -X- -X- O"]
        for file in tqdm(files):
            print(f"Processing file: {file}")
            with open(file, 'r', encoding='utf-8') as fh:
                page = fh.readlines()

            for line in page[1:]:
                if line.strip() == '':
                    continue

                text = line.strip().split('\t')
                # Column to be extracted:
                # 0: token, 1: speech, 4: sentstart
                try:
                    token = text[0]
                    speech = text[1]
                    start = text[4]
                except IndexError:
                    print(line)

                # Add new empty row if it's beginning of sentence
                if start.strip() == 'start':
                    content.append('')

                # Change cat into intended labels:
                # direct -> DIR, indirect -> IND, reported -> REP, x -> O
                if speech == '0':
                    speech = 'O'
                else:
                    speech = 'DIR'

                # Create new data row
                new_data = token + '\tO\tO\t' + speech
                content.append(new_data)

        filename = INTERIM_DIR + 'farm_' + task + '.tsv'
        with open(filename, 'w', encoding='utf-8') as fh:
            for line in content:
                fh.write(line + '\n')


def finalize_farm_ner():
    # In the temporary datasets, the labels are only set as DIR, IND, REP, and O.
    # Here, the labels are marked to comply with the IOB format.
    tasks = ["train", "dev", "test"]

    for task in tasks:
        filename = INTERIM_DIR + 'farm_' + task + '.tsv'
        print(f"Processing {filename} ...")
        with open(filename, 'r', encoding='utf-8') as fh:
            page = fh.readlines()

        content = ["-DOCSTART- -X- -X- O"]
        for i, line in enumerate(tqdm(page[1:])):
            if line.strip() == '':
                content.append('')
                continue

            text = line.strip().split('\t')
            prev = page[i].strip().split('\t')
            if text[-1] != 'O':
                if prev[-1] == '':
                    text[-1] = 'B-' + text[-1]
                else:
                    if text[-1] != prev[-1]:
                        text[-1] = 'B-' + text[-1]
                    else:
                        text[-1] = 'I-' + text[-1]

            content.append('\t'.join(text))

        filename = TARGET_DIR + task + '.txt'
        with open(filename, 'w', encoding='utf-8') as fh:
            for line in content:
                fh.write(line + '\n')


def main():
    # 01. Combine split raw data into one file in 02_interim
    combine_raw_to_interim()
    # 02. Correcting the labels and adding the B- and I- markings
    finalize_farm_ner()


if __name__ == '__main__':
    main()
