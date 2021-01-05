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


def combine_raw_to_interim():
    sets = ['train', 'val', 'test']
    for task in sets:
        file1 = RAW_DIR + task + "/direct_combined.tsv"
        file2 = RAW_DIR + task + "/indirect_combined.tsv"
        file3 = RAW_DIR + task + "/reported_combined.tsv"

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

        filename = INTERIM_DIR + task + '.tsv'
        with open(filename, 'w', encoding='utf-8') as fh:
            fh.writelines(content)


def convert_to_farm_ner():
    tasks = ["train", "dev", "test"]

    for task in tasks:
        filename = INTERIM_DIR + task + '.tsv'
        print(f"Processing {filename} ...")
        with open(filename, 'r', encoding='utf-8') as fh:
            page = fh.readlines()

        content = ["-DOCSTART- -X- -X- O"]
        for line in tqdm(page[1:]):
            text = line.strip().split('\t')

            # Column to be extracted:
            # 0: token, 2: lemma, 3: pos, 5: sentstart, 7: cat
            token = text[0]
            # lemma = text[2]
            # pos = text[3]
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
            new_data = token + '\tO\tO\t' + cat
            content.append(new_data)

        filename = INTERIM_DIR + 'farm_ner_' + task + '.txt'
        with open(filename, 'w', encoding='utf-8') as fh:
            for line in content:
                fh.write(line + '\n')


def finalize_farm_ner():
    # In the temporary datasets, the labels are only set as DIR, IND, REP, and O.
    # Here, the labels are marked to comply with the IOB format.
    tasks = ["train", "dev", "test"]

    for task in tasks:
        filename = INTERIM_DIR + 'farm_ner_' + task + '.txt'
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


def check_dataset():
    # It turns out that in the original train set, there are some null elements
    # and causing an error because the split data become shorter than 9 elements.
    # These faulty sentences were deleted manually.
    # This method is only to find those data causing the error.
    filename = TARGET_DIR + 'train.txt'
    with open(filename, 'r', encoding='utf-8') as fh:
        page = fh.readlines()

    # Check if there are '\t' character in the tokens
    print()
    print('Check for \\t character in token...')
    errors = []
    for i, line in enumerate(tqdm(page[1:])):
        if line.strip() == '':
            continue
        text = line.strip().split('\t')
        if len(text) != 4:
            errors.append(i)
    if len(errors) > 0:
        pprint.pprint(f"Found errors in dataset on index: {errors} (Add 2 for line in file)")

    # Check if there are spaces in the tokens
    print('Check for spaces in token...')
    errors = []
    for i, line in enumerate(tqdm(page[1:])):
        if line.strip() == '':
            continue
        text = line.strip().split('\t')
        token = text[0]
        if len(token.split(' ')) > 1:
            errors.append(i)
    if len(errors) > 0:
        pprint.pprint(f"Found errors in dataset on index: {errors} (Add 2 for line in file)")


def correct_dataset():
    # List of error to be corrected in train.txt:
    # Err01: Line 694012 (index 694011): Only have 3 columns.
    #   -> Add '\tO' to that line.
    # Err02: Some lines have spaces in the token
    #   -> Split the token into 2 rows with the same NER
    tasks = ['train', 'dev', 'test']
    for task in tasks:
        filename = TARGET_DIR + 'temp_' + task + '.txt'
        with open(filename, 'r', encoding='utf-8') as fh:
            page = fh.readlines()

        # Err01
        if task == 'train':
            page[694011] = page[694011][:-1] + '\tO\n'

        # Err02
        content = [x.strip().split('\t') for x in tqdm(page[1:])]
        new_content = ["-DOCSTART- -X- -X- O"]
        for items in tqdm(content):
            if len(items) == 0:
                new_content.append('')
            else:
                tokens = items[0].split(' ')
                if len(tokens) == 1:
                    new_content.append('\t'.join(items))
                else:
                    label = items[-1]
                    for i, token in enumerate(tokens):
                        row_label = label
                        if label.startswith('B-') and i > 0:
                            row_label = 'I-' + label[2:]
                        text = token + '\tO\tO\t' + row_label
                        new_content.append(text)
        new_file = TARGET_DIR + task + '.txt'
        with open(new_file, 'w', encoding='utf-8') as fh:
            for line in new_content:
                fh.write(line + '\n')


def main():
    # 01. Combine split raw data into one file in 02_interim
    # combine_raw_to_interim()
    # 02. Convert interim data into FARM format for NER dataset
    # convert_to_farm_ner()
    # 03. Correcting the labels and adding the B- and I- markings
    # finalize_farm_ner()
    # 04. Validate and correcting errors
    check_dataset()
    # correct_dataset()


if __name__ == '__main__':
    main()
