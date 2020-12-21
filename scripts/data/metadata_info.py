import pprint

DATA_DIR = "C:/My Projects/Python/STWR/data/01_raw/"
DATA_FILE = DATA_DIR + "metadata.tsv"


def main():
    with open(DATA_FILE, 'r', encoding='utf-8') as fh:
        page = fh.readlines()

    fictional = {'yes': 0, 'no': 0}     # position 7
    narrative = {'yes': 0, 'no': 0}     # position 9
    perspective = {'first': 0, 'first_plural': 0, 'third': 0}   # position 12

    lines = [x.strip().split('\t') for x in page]
    for i, line in enumerate(lines[1:]):
        fictional[line[7].strip()] += 1
        narrative[line[9].strip()] += 1
        try:
            perspective[line[12].strip()] += 1
        except KeyError:
            perspective[line[12].strip()] = 1
        except:
            print(f"something went wrong in line {i}")


    pprint.pprint(fictional)
    pprint.pprint(narrative)
    pprint.pprint(perspective)


if __name__ == '__main__':
    main()
