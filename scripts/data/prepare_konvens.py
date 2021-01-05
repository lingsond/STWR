import os
import pprint
import random
import numpy as np
from tqdm import tqdm, trange
from sklearn.model_selection import train_test_split


DATA_DIR = "C:/My Projects/Python/STWR/data/"
BASE_DIR = "C:/My Projects/Python/STWR/data/"
RAW_DIR = BASE_DIR + "01_raw/konvens-paper-2020/"
INTERIM_DIR = BASE_DIR + "02_interim/Konvens2020/"
TARGET_DIR = BASE_DIR + "03_processed/Konvens2020/"

NER_LABELS_ORIGINAL = ['direct', 'indirect', 'reported', 'x']


def get_file_list():
    filename = RAW_DIR + 'train/indirect_combined.tsv'
    with open(filename, 'r', encoding='utf-8') as fh:
        page = fh.readlines()

    files = [x.strip().split('\t')[-1] for x in page]
    files = list(set(files))
    return files


def combine_raw_to_interim():
    sets = ['val', 'test']
    for task in sets:
        file1 = SOURCE_DIR + task + "/direct_combined.tsv"
        file2 = SOURCE_DIR + task + "/indirect_combined.tsv"
        file3 = SOURCE_DIR + task + "/reported_combined.tsv"

        with open(file1, 'r', encoding='utf-8') as fh:
            page1 = fh.readlines()
            page1 = page1[1:]
        with open(file2, 'r', encoding='utf-8') as fh:
            page2 = fh.readlines()
            page2 = page2[1:]
        with open(file3, 'r', encoding='utf-8') as fh:
            page3 = fh.readlines()
            page3 = page3[1:]

        content = ["tok	normtok	lemma	pos	rfpos	sentstart	fictional	cat	file\n"]
        # labels = []
        for i, line in enumerate(tqdm(page1)):
            # Column to be checked: 7 (cat)
            text1 = line.split('\t')
            text2 = page2[i].split('\t')
            text3 = page3[i].split('\t')
            if text1[7] == 'x':
                if text2[7] != 'x':
                    text1[7] = text2[7]
                elif text3[7] != 'x':
                    text1[7] = text3[7]

            content.append('\t'.join(text1))

            # labels.append(text1[7])

        if task == 'train':
            # The file indirect_combined.tsv is longer than the other 2.
            # Adding the additional data from indirect_combined.tsv
            content = content + page2[len(page1):]

        if task == 'val':
            task = 'dev'

        filename = TARGET_DIR + task + '.tsv'
        with open(filename, 'w', encoding='utf-8') as fh:
            fh.writelines(content)


def convert_to_farm_ner():
    SOURCE_DIR = BASE_DIR + "02_interim/Konvens2020/"
    # TARGET_DIR = BASE_DIR + "03_processed/Konvens2020/"

    files = ["train", "dev", "test"]

    content = ["-DOCSTART- -X- -X- O"]
    for file in files:
        filename = SOURCE_DIR + file + '.tsv'
        print(f"Processing {filename} ...")
        with open(filename, 'r', encoding='utf-8') as fh:
            page = fh.readlines()

        for line in tqdm(page[1:]):
            text = line.strip().split('\t')

            # Column to be extracted:
            # 0: token, 2: lemma, 3: pos, 5: sentstart, 7: cat
            token = text[0]
            lemma = text[2]
            pos = text[3]
            start = text[5]
            cat = text[7]

            # Skip if token is 'EOF' as it is not part of text
            if token == 'EOF':
                continue

            # Add new empty row if it's beginning of sentence
            if start.strip() == 'yes':
                content.append('')

            # Change cat into intended labels:
            # direct -> DIR, indirect -> IND, reported -> REP, x -> O
            if cat == 'x':
                cat = 'O'
            elif cat == 'direct':
                cat = 'DIR'
            elif cat == 'indirect':
                cat = 'IND'
            elif cat == 'reported':
                cat = 'REP'

            # Create new data row
            new_data = token + '\t' + lemma + '\t' + pos + '\t' + cat
            content.append(new_data)

        filename = SOURCE_DIR + 'temp_' + file + '.txt'
        with open(filename, 'w', encoding='utf-8') as fh:
            for line in content:
                fh.write(line + '\n')


def finalize_farm_ner():
    # In the temporary datasets, the labels are only set as DIR, IND, REP, and O.
    # Here, the labels are marked to comply with the IOB format.
    SOURCE_DIR = BASE_DIR + "02_interim/Konvens2020/"
    TARGET_DIR = BASE_DIR + "03_processed/Konvens2020/"

    files = ["train", "dev", "test"]

    for file in files:
        filename = SOURCE_DIR + 'temp_' + file + '.txt'
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

        filename = TARGET_DIR + file + '.txt'
        with open(filename, 'w', encoding='utf-8') as fh:
            for line in content:
                fh.write(line + '\n')


def check_trainset():
    # It turns out that in the original train set, there are some null elements
    # and causing an error because the split data become shorter than 9 elements.
    # These faulty sentences were deleted manually.
    # This method is only to find those data causing the error.
    # filename = BASE_DIR + "02_interim/Konvens2020/" + 'train.tsv'
    filename = BASE_DIR + "03_processed/Konvens2020/" + 'train.txt'
    with open(filename, 'r', encoding='utf-8') as fh:
        page = fh.readlines()

    for i, line in enumerate(page):
        if line.strip() == '':
            continue
        text = line.strip().split('\t')
        if len(text) < 4:
            print(i)


def main():
    files = get_file_list()
    pprint.pprint(files)
    # 01. Combine split raw data into one file in 02_interim
    # combine_raw_to_interim()
    # 02. Convert interim data into FARM format for NER dataset
    # convert_to_farm_ner()
    # 03. Correcting the labels and adding the B- and I- markings
    # finalize_farm_ner()


if __name__ == '__main__':
    main()
    # check_trainset()
