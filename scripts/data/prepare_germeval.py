import os
import pprint
import random
import numpy as np
from tqdm import tqdm, trange
from sklearn.model_selection import train_test_split


DATA_DIR = "C:/My Projects/Python/STWR/data/"
BASE_DIR = "C:/My Projects/Python/STWR/data/"
SOURCE_DIR = BASE_DIR + "01_raw/GermEval2014/"
# SOURCE_DIR = DATA_DIR
TARGET_DIR = BASE_DIR + "03_processed/GermEval2014/"

NER_LABELS = ['I-LOC', 'B-OTH', 'B-LOC', 'O', 'I-OTH', 'B-PER', 'B-ORG', 'I-ORG', 'I-PER']


def prepare_dataset():
    file_start = "NER-de-"
    files = ["train.tsv", "dev.tsv", "test.tsv"]
    # files = ["train.tsv"]

    content = ["-DOCSTART- -X- -X- O", ""]
    labels = []
    for file in tqdm(files):
        filename = SOURCE_DIR + file_start + file
        with open(filename, 'r', encoding='utf-8') as fh:
            page = fh.readlines()

        for line in page:
            if line[0] != '#':
                # Comments will be ignored
                text = line.strip()
                if text == '':
                    content.append(text)
                else:
                    text = text.split('\t')
                    if text[2] in NER_LABELS:
                        ner = text[2]
                    else:
                        ner = 'O'
                    new_text = text[1] + '\tO\tO\t' + ner
                    content.append(new_text)
                    labels.append(ner)

        filename = TARGET_DIR + file
        with open(filename, 'w', encoding='utf-8') as fh:
            for line in content:
                fh.write(line + '\n')

    labels = list(set(labels))
    print(labels)


def main():
    # fname = 'rwk_mkhz_20186-1.xmi.tsv'
    # convert_tsv_to_sentences(fname)
    # convert_all_tsv()
    prepare_dataset()


if __name__ == '__main__':
    main()