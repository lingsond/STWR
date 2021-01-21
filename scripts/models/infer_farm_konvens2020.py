import os
import sys
import logging
from pathlib import Path
import pprint
import json
from ast import literal_eval
import numpy as np
import sklearn

from farm.data_handler.data_silo import DataSilo
from farm.data_handler.processor import NERProcessor
from farm.modeling.optimization import initialize_optimizer
from farm.infer import Inferencer
from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.language_model import LanguageModel
from farm.modeling.prediction_head import TokenClassificationHead
from farm.modeling.tokenization import Tokenizer
from farm.train import Trainer
from farm.utils import set_all_seeds, initialize_device_settings

if os.name == 'nt':
    BASE_DIR = "C:/My Projects/Python/STWR/"
    use_cuda = False
else:
    BASE_DIR = "/home/stud/wangsadirdja/STWR/"
    use_cuda = True
DATA_DIR = BASE_DIR + "data/03_processed/Konvens2020/direct/"
# MODEL_DIR = BASE_DIR + "models/farm-ner-konvens2020_bert-hgcrw_direct"
MODEL_DIR = BASE_DIR + "models/farm-ner-konvens2020_lmgot01_direct"
# MODEL_DIR = 'bert-base-german-cased'


def test_file_to_dict(testfile: str = 'test', extension: str = 'txt', sep: str = '\t'):
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


def scoring_result():
    basic_texts, golden_list, golden_label = test_file_to_dict()
    pred_result = json_to_list("test_infer_01.json")
    real_list = text_to_list(basic_texts)
    pred_text, pred_label = fill_prediction_list(pred_result, real_list)

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
    model = Inferencer.load(MODEL_DIR)
    results = []
    for text in basic_texts:
        result = model.inference_from_dicts(dicts=[text])
        results.append(result)
    # pprint.pprint(results)
    filename = 'infer_ner_konvens2020_direct_lmgot01.json'
    with open(filename, 'w', encoding='utf-8') as fh:
        fh.write(pprint.pformat(results, indent=2))
    # with open("test_infer.json", 'w') as fh:
    #     json.dump(result, fh, indent=2)

    model.close_multiprocessing_pool()


if __name__ == "__main__":
    #task = sys.argv[1]
    #lang_model = sys.argv[2]
    # Parameter1 can be '', 'direct', 'indirect', 'reported'
    # Parameter2 can be 'lmgot01', 'lmgot02', 'bert-hgcrw', 'bert-gc'
    #ner(task, lang_model)
    infer()
    # scoring_result()
