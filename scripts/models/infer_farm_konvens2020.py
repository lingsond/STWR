import os
import sys
import logging
from pathlib import Path
import pprint
import json

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
MODEL_DIR = BASE_DIR + "models/farm-ner-konvens2020_bert-hgcrw_direct"
# MODEL_DIR = BASE_DIR + "models/farm-ner-konvens2020_lmgot02_direct"
# MODEL_DIR = 'bert-base-german-cased'

def infer():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    # 9. Load it & harvest your fruits (Inference)
    # basic_texts = [
    #     {"text": "Schartau sagte dem Tagesspiegel, dass Fischer ein Idiot sei"},
    #     {"text": "Martin Müller spielt Handball in Berlin"},
    # ]
    model = Inferencer.load(MODEL_DIR)
    # result1 = model.inference_from_dicts(dicts=basic_texts)
    result = model.inference_from_file(DATA_DIR + 'test.txt')
    pprint.pprint(result[0:5])
    # with open("test_infer1.json", 'w') as fh:
    #    fh.write(pprint.pformat(result1, indent=2))
    with open("test_infer.json", 'w') as fh:
        json.dump(result, fh, indent=2)

    model.close_multiprocessing_pool()


if __name__ == "__main__":
    #task = sys.argv[1]
    #lang_model = sys.argv[2]
    # Parameter1 can be '', 'direct', 'indirect', 'reported'
    # Parameter2 can be 'lmgot01', 'lmgot02', 'bert-hgcrw', 'bert-gc'
    #ner(task, lang_model)
    infer()
