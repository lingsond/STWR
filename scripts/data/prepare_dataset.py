import os


DATA_DIR = "C:/My Projects/Uni Würzburg/HiWi/STWR/corpus-master/data/main/txt/"
BASE_DIR = "C:/My Projects/Python/STWR/data/"
# SOURCE_DIR = BASE_DIR + "01_raw/"
SOURCE_DIR = DATA_DIR
TARGET_DIR = BASE_DIR + "02_interim/"


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
    for file in files:
        convert_tsv_to_sentences(file)

def main():
    fname = 'rwk_mkhz_20186-1.xmi.tsv'
    convert_tsv_to_sentences(fname)


if __name__ == '__main__':
    main()