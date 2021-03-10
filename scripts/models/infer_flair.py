import os
import sys
import logging
from pathlib import Path
from pprint import pprint
import json
from ast import literal_eval
import numpy as np
import sklearn
from flair.data import Sentence
from flair.models import SequenceTagger
from tqdm import tqdm


if os.name == 'nt':
    BASE_DIR = "C:/My Projects/Python/STWR/"
    use_cuda = False
else:
    BASE_DIR = "/home/stud/wangsadirdja/STWR/"
    use_cuda = True

DATA_DIR = BASE_DIR + "data/03_processed/Konvens2020/direct/"
# MODEL_DIR = BASE_DIR + "models/farm-ner-konvens2020_bert-hgcrw_direct"
# MODEL_DIR = BASE_DIR + "models/farm-ner-konvens2020_lmgot01_direct"
# MODEL_DIR = BASE_DIR + "models/konvens2020/stwr_ner/"


def test_file_to_dict(testfile: str = 'test', extension: str = 'txt', sep: str = '\t'):
    # Returns 3 lists:
    # List 1: dictionary (each line = 1 dictionary with only 'text' in the dict)
    # List 2: all tokens (sentences are broken down to tokens)
    # List 3: label from each token in List 2
    filename = DATA_DIR + testfile + '.' + extension
    with open(filename, 'r', encoding='utf-8') as fh:
        page = fh.readlines()

    text = []
    token_list = []
    label_list = []

    tokens = []
    for line in page[1:]:
        temp = line.strip()
        # Check if it's a new sentence/context
        if temp == '':
            text.append(' '.join(tokens))
            tokens = []
        else:
            items = temp.split(sep=sep)
            # Only first and last column is needed
            token = items[0]
            label = items[-1]
            if '-' in label:
                label = label[2:]

            tokens.append(token)
            token_list.append(token)
            label_list.append(label)

    if len(tokens) > 0:
        text.append(' '.join(tokens))

    data = []
    for line in text:
        data.append({"text": line})
    return data, token_list, label_list


def tsv_to_list(filename: str) -> list:
    with open(filename, 'r', encoding='utf-8') as fh:
        page = fh.readlines()

    results = []
    for line in page:
        items = line.strip().split('\t')
        results.append([items[0], items[1]])

    return results


def json_to_list(filename: str) -> list:
    with open(filename, 'r', encoding='utf-8') as fh:
        page = literal_eval(fh.read())

    results = [x[0]['predictions'] for x in page]

    # Separating the result if it spans over more than 1 token
    new_results = []
    for line in results:
        new_line = []
        for each in line:
            context = each['context']
            tokens = context.split(' ')
            start = each['start']
            label = each['label']
            for token in tokens:
                end = len(token) + start
                new_line.append([token, start, end, label])
                start = end + 1
        new_results.append(new_line)

    return new_results


def text_to_list(data):
    text_list = [x['text'] for x in data]

    new_results = []
    for line in text_list:
        tokens = line.split(' ')
        start = 0
        new_line = []
        for token in tokens:
            end = len(token) + start
            new_line.append([token, start, end])
            start = end + 1
        new_results.append(new_line)

    return new_results


def fill_prediction_list(pred, real):
    if len(pred) != len(real):
        raise Exception("List have different length")
    new_preds = []
    for n in range(len(real)):
        preds = pred[n]
        current = 0
        for token in real[n]:
            if len(preds) == 0 or current >= len(preds) or token[0] != preds[current][0]:
                new_token = token + ['O']
                new_preds.append(new_token)
            else:
                new_preds.append(preds[current])
                current += 1

    pred_list = []
    label_list = []
    for preds in new_preds:
        pred_list.append(preds[0])
        label_list.append(preds[-1])

    return pred_list, label_list


def format_redewiedergabe_to_farm(results):
    pred_list = []
    label_list = []

    for result in results:
        pred_list.append(result[0])
        # Change direct to DIR, and x to O
        label = result[1]
        if label == 'direct':
            label = 'DIR'
        else:
            label = 'O'
        label_list.append(label)

    return pred_list, label_list

def scoring_result():
    basic_texts, golden_list, golden_label = test_file_to_dict()
    filename = 'results_infer_flair_konvens2020_direct.tsv'
    pred_result = tsv_to_list(filename)
    # real_list = text_to_list(basic_texts)
    pred_text, pred_label = format_redewiedergabe_to_farm(pred_result)

    fscore = np.round(
        sklearn.metrics.f1_score(pred_label, golden_label, pos_label='DIR'), 2)
    precision = np.round(
        sklearn.metrics.precision_score(pred_label, golden_label, pos_label='DIR'), 2)
    recall = np.round(
        sklearn.metrics.recall_score(pred_label, golden_label, pos_label='DIR'), 2)

    print(f'FI Score : {fscore}')
    print(f'Precision: {precision}')
    print(f'Recall   : {recall}')


def infer():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    basic_texts, golden_list, golden_label = test_file_to_dict()

    tagger = SequenceTagger.load('de-historic-direct')

    results = []
    texts = [x['text'] for x in basic_texts]
    for text in tqdm(texts):
        sentence = Sentence(text)
        tagger.predict(sentence)
        for token in sentence:
            result = token.text + '\t' + token.labels[0].value + '\n'
            results.append(result)
    # pprint.pprint(results)
    filename = 'results_infer_flair_konvens2020_direct.tsv'
    with open(filename, 'w', encoding='utf-8') as fh:
        fh.writelines(results)


def testing():
    basic_texts, golden_list, golden_label = test_file_to_dict()
    print('basic_texts')
    #pprint(basic_texts)
    print('golden_list')
    # pprint(golden_list)
    print('golden_label')
    pprint(golden_label)


if __name__ == "__main__":
    # exp_id = sys.argv[1]
    #lang_model = sys.argv[2]
    # Parameter1 can be '', 'direct', 'indirect', 'reported'
    # Parameter2 can be 'lmgot01', 'lmgot02', 'bert-hgcrw', 'bert-gc'
    #ner(task, lang_model)
    infer()
    scoring_result()
