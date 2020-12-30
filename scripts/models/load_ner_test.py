import os
import logging
from pathlib import Path
from tqdm import tqdm
import pprint
import json
import re


if os.name == 'nt':
    BASE_DIR = "C:/My Projects/Python/STWR/"
    use_cuda = False
else:
    BASE_DIR = "/home/stud/wangsadirdja/STWR/"
    use_cuda = True
DATA_DIR = BASE_DIR + "data/03_processed/CoNLL2003/"
# DATA_DIR = BASE_DIR + "data/03_processed/GermEval2014/"
MODEL_DIR = BASE_DIR + "models/farm-ner-tutorial-germeval2014"
REPORT_DIR = BASE_DIR + "reports/"

project = "germeval2014"
report_id = "001"
lang_model = MODEL_DIR
ner_labels = ["[PAD]", "X", "O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-OTH", "I-OTH"]
evaluation_filename = "test.txt"
# delimiter = "\t"
delimiter = " "


def get_test_data_ner(sep: str = '\t'):
    test_file = DATA_DIR + evaluation_filename
    with open(test_file, 'r', encoding='utf-8') as fh:
        page = fh.readlines()
    test_data = []
    words = []
    for line in tqdm(page):
        if line.startswith('-DOCSTART-'):
            continue

        # Empty row indicates new sentence
        if line.strip() == '':
            if len(words) > 0:
                sentence = ' '.join(words)
                test_data.append(sentence)
                words = []
        else:
            temp = line.strip().split(sep)
            words.append(temp[0])

    if len(words) > 0:
        sentence = ' '.join(words)
        test_data.append(sentence)

    with open("test_data.txt", 'w') as fh:
        for line in test_data:
            fh.write(line + '\n')


def get_ner_gold_labels(sep: str = '\t', join_context: bool = False):
    # In this method, the tokens are joined into one context if they are sequential
    # and share the same label
    test_file = DATA_DIR + evaluation_filename
    with open(test_file, 'r', encoding='utf-8') as fh:
        page = fh.readlines()

    gold_labels = []
    predictions = []
    current_start = 0
    current_end = 0
    for line in tqdm(page):
        if line.startswith('-DOCSTART-'):
            continue

        # Empty row indicates new sentence -> new prediction list
        if line.strip() == '':
            if len(predictions) > 0:
                gold_labels.append(predictions)
                predictions = []
                current_start = 0
                current_end = 0
        else:
            temp = line.strip().split(sep)
            token = temp[0]
            token_label = temp[3]
            label = token_label
            if '-' in label:
                pos = label.find('-')
                label = label[pos+1:]
            if len(predictions) == 0:
                # First token (beginning of sentence)
                current_end = len(token)
                prediction = {
                    'context': token, 'label': label, 'start': current_start, 'end': current_end
                }
                predictions.append(prediction)
            else:
                last_label = predictions[-1]['label']
                if join_context and label == last_label:
                    # Label hasn't changed, so we combine current token with the previous one
                    # No new prediction is added to list. Only update previous prediction
                    new_context = predictions[-1]['context'] + ' ' + token
                    current_end += len(token) + 1
                    predictions[-1]['context'] = new_context
                    predictions[-1]['end'] = current_end
                else:
                    # Label has changed so a new prediction is added to list
                    current_start = current_end + 1
                    current_end = current_start + len(token)
                    prediction = {
                        'context': token, 'label': label,
                        'start': current_start, 'end': current_end
                    }
                    predictions.append(prediction)

    if len(predictions) > 0:
        gold_labels.append(predictions)

    # Remove all context with the label 'O'
    new_gold_labels = []
    for predictions in tqdm(gold_labels):
        new_predictions = [x for x in predictions if x['label'] != 'O']
        new_gold_labels.append(new_predictions)

    with open("gold_labels.txt", 'w') as fh:
        for line in new_gold_labels:
            fh.write(pprint.pformat(line, indent=2) + '\n')
    with open("gold_labels.json", 'w') as fh:
        json.dump(new_gold_labels, fh, indent=2)


def convert_result_labels():
    with open("result_sample.txt", 'r', encoding='utf-8') as fh:
        page = fh.read()
    results = eval(page)

    temp_results = [x['predictions'] for x in results]

    results = []
    new_predictions = []
    # If the context in the prediction is longer than one word, it will be split
    for predictions in temp_results:
        for item in predictions:
            start = item['start']
            label = item['label']
            words = item['context'].split(' ')
            for word in words:
                end = start + len(word)
                prediction = {
                    'context': word, 'label': label, 'start': start, 'end': end
                }
                new_predictions.append(prediction)
                start = end + 1
        results.append(new_predictions)
        new_predictions = []

    pprint.pprint(results)


if __name__ == "__main__":
    # get_test_data_ner(sep=delimiter)
    # get_ner_gold_labels(sep=delimiter)
    convert_result_labels()
