import os
import pprint
import random
import numpy as np
from tqdm import tqdm, trange
from sklearn.model_selection import train_test_split


DATA_DIR = "C:/My Projects/Uni Würzburg/HiWi/STWR/corpus-master/data/main/txt/"
BASE_DIR = "C:/My Projects/Python/STWR/data/"
SOURCE_DIR = BASE_DIR + "02_interim/"
# SOURCE_DIR = DATA_DIR
TARGET_DIR = BASE_DIR + "03_processed/"


def convert_tsv_to_sentences(source_file: str) -> None:
    filename = SOURCE_DIR + source_file
    with open(filename, 'r', encoding='utf-8') as fh:
        page = fh.readlines()

    lines = [x.strip().split('\t') for x in page[1:]]

    sentence = ''
    stwr = []
    content = []

    for line in lines:
        # Check start of sentence
        sentence_start = line[5].strip().lower()
        if sentence_start == 'yes':
            if len(sentence) > 0:
                content.append([sentence, list(set(stwr))])
            sentence = ''
            stwr = []

        token = line[0].strip()
        sign = line[4][:3]
        # The structure of stwr is type, medium, id joined by a '.'.
        # We only need the type
        stwr_type = line[6].split('.')[0]

        if sentence_start != 'yes' and sign != 'SYM':
            sentence += ' '
        sentence += token
        stwr.append(stwr_type)

    if len(sentence) > 0:
        content.append([sentence, list(set(stwr))])

    temp = source_file.split('.')
    target_file = temp[0] + '.' + temp[2]
    filename = TARGET_DIR + target_file

    with open(filename, 'w', encoding='utf-8') as fh:
        header = 'text\tstwr\n'
        fh.write(header)
        for line in content:
            if len(line[1]) > 1 and '-' in line[1]:
                line[1].remove('-')
            stwr = ';'.join(line[1])
            if stwr == '-':
                stwr = 'NN'
            elif stwr == 'freeIndirect':
                stwr = 'FI'
            elif stwr == 'direct':
                stwr = 'DS'
            elif stwr == 'reported':
                stwr = 'RP'
            elif stwr == 'indirect':
                stwr = 'IS'
            sentence = line[0].replace('–––', ' - - -')
            text = sentence + '\t' + stwr + '\n'
            fh.write(text)


def convert_all_tsv():
    files = [file for file in os.listdir(DATA_DIR) if file.endswith('.tsv') and file.startswith('rwk_')]
    for file in tqdm(files):
        convert_tsv_to_sentences(file)


def prepare_dataset():
    files = [file for file in os.listdir(SOURCE_DIR) if file.endswith('.tsv')]

    # stwr = {'NN': 0, 'DS': 0, 'IS': 0, 'FI': 0, 'RP': 0, 'other': 0}
    stwr = ['NN', 'DS', 'IS', 'FI', 'RP']
    texts = []
    labels = []
    for file in tqdm(files):
        filename = SOURCE_DIR + file
        with open(filename, 'r', encoding='utf-8') as fh:
            page = fh.readlines()
        lines = [x.strip().split('\t') for x in page[1:]]
        for line in lines:
            # try:
            #     stwr[line[1]] += 1
            # except KeyError:
            #     stwr['other'] += 1
            if line[1] in stwr:
                texts.append(line[0])
                labels.append(line[1])

    test_size = 0.2
    dev_size = 0.25
    texts_train, texts_tests, labels_train, labels_tests = train_test_split(
        texts, labels, test_size=test_size, random_state=42, stratify=labels
    )

    texts_train, texts_dev, labels_train, labels_dev = train_test_split(
        texts_train, labels_train, test_size=dev_size, random_state=42, stratify=labels_train
    )

    # a, b = np.unique(labels_train, return_counts=True)
    # c, d = np.unique(labels_tests, return_counts=True)
    # e, f = np.unique(labels_dev, return_counts=True)
    # print(a)
    # print(f"Train by class: {b}")
    # print(f"Test by class: {d}")
    # print(f"Dev by class: {f}")

    # Distribution of   'DS' 'FI' 'IS' 'NN' 'RP'
    # Training set:     4091  164  948 9322 1770
    # Dev & Test set:   1364   55  316 3107  590

    random.seed(42)
    trainset = list(zip(texts_train, labels_train))
    random.shuffle(trainset)
    devset = list(zip(texts_dev, labels_dev))
    random.shuffle(devset)
    testset = list(zip(texts_tests, labels_tests))
    random.shuffle(testset)

    filename = TARGET_DIR + "train.tsv"
    with open(filename, 'w', encoding='utf-8') as fh:
        fh.write('text\tlabel\n')
        for line in tqdm(trainset):
            s = '\t'.join(line) + '\n'
            fh.write(s)

    filename = TARGET_DIR + "dev.tsv"
    with open(filename, 'w', encoding='utf-8') as fh:
        fh.write('text\tlabel\n')
        for line in tqdm(devset):
            s = '\t'.join(line) + '\n'
            fh.write(s)

    filename = TARGET_DIR + "test.tsv"
    with open(filename, 'w', encoding='utf-8') as fh:
        fh.write('text\tlabel\n')
        for line in tqdm(testset):
            s = '\t'.join(line) + '\n'
            fh.write(s)


def main():
    # fname = 'rwk_mkhz_20186-1.xmi.tsv'
    # convert_tsv_to_sentences(fname)
    # convert_all_tsv()
    prepare_dataset()


if __name__ == '__main__':
    main()