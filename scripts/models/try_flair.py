from flair.data import Sentence
from flair.models import SequenceTagger
from pprint import pprint


DATA_PATH = 'C:/My Projects/Python/STWR/data/03_processed/Konvens2020/direct/'
def parse_farm_text(pfile):
    with open(pfile, 'r', encoding='utf-8') as fh:
        page = fh.readlines()[1:]

    dataset = []
    sentence = []
    for line in page:
        if line.strip() == '':
            text = ' '.join(sentence)
            dataset.append(text)
            sentence = []
            continue

        token = line.strip().split('\t')[0]
        sentence.append(token)

    # If last line in file is not blank row, add last sentence to dataset
    if sentence:
        text = ' '.join(sentence)
        dataset.append(text)

    return dataset


def testing():
    t = SequenceTagger.load('de-historic-direct')
    sentences = [
        'Ich liebe Berlin.',
        'George Washington gang nach Washington.',
        'Achtung!!'
    ]
    for item in sentences:
        s = Sentence(item)
        t.predict(s)
        print(s)
        for token in s:
            print(token.text, token.label)

    # for entity in s.get_spans('ner'):
    #     print(entity)


def main():
    filename = DATA_PATH + 'test.txt'
    sentences = parse_farm_text(filename)
    pprint(sentences[:10])


if __name__ == '__main__':
    # main()
    testing()
