import os
import logging
from pathlib import Path
import pprint

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
from farm.eval import Evaluator

if os.name == 'nt':
    BASE_DIR = "C:/My Projects/Python/STWR/"
    use_cuda = False
else:
    BASE_DIR = "/home/stud/wangsadirdja/STWR/"
    use_cuda = True
DATA_DIR = BASE_DIR + "data/03_processed/GermEval2014/"
MODEL_DIR = BASE_DIR + "models/farm-ner-tutorial-germeval2014"


def evaluate_ner():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    ##########################
    ########## Settings
    ##########################
    set_all_seeds(seed=42, deterministic_cudnn=True)
    use_amp = None
    device, n_gpu = initialize_device_settings(use_cuda=use_cuda, use_amp=use_amp)
    batch_size = 100
    lang_model = "bert-base-german-cased"
    do_lower_case = False

    data_dir = Path(DATA_DIR)
    evaluation_filename = "test.txt"
    label_list = ["PER", "ORG", "LOC", "OTH"]
    metric = "f1_macro"

    # 1.Create a tokenizer
    print("# 1. Create a tokenizer")
    tokenizer = Tokenizer.load(
        pretrained_model_name_or_path=lang_model, do_lower_case=do_lower_case
    )

    # 2. Create a DataProcessor that handles all the conversion from raw text into a pytorch Dataset
    print("# 2. Create a DataProcessor")
    # See test/sample/ner/train-sample.txt for an example of the data format that is expected by the Processor
    # ner_labels = ["[PAD]", "X", "O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-OTH", "I-OTH"]
    ner_labels = ["[PAD]", "X", "O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-OTH", "I-OTH"]

    processor = NERProcessor(
        tokenizer=tokenizer,
        max_seq_len=128,
        data_dir=data_dir,
        delimiter="\t",
        metric=metric,
        label_list=ner_labels,
        train_filename=None,
        dev_filename=None,
        dev_split=0,
        test_filename=evaluation_filename,
    )

    # 3. Create a DataSilo that loads several datasets (train/dev/test), provides DataLoaders for them and calculates a few descriptive statistics of our datasets
    print("# 3. Create a DataSilo")
    data_loader_worker = 1
    data_silo = DataSilo(processor=processor, batch_size=batch_size, max_processes=data_loader_worker)

    # 4. Create an Evaluator
    evaluator = Evaluator(
        data_loader=data_silo.get_data_loader("test"),
        tasks=data_silo.processor.tasks,
        device=device
    )

    # 5. Load model
    model = AdaptiveModel.load(MODEL_DIR, device=device)
    # use "load" if you want to use a local model that was trained with FARM
    # model = AdaptiveModel.load(lang_model, device=device)
    model.connect_heads_with_processor(data_silo.processor.tasks, require_labels=True)

    # 6. Run the Evaluator
    results = evaluator.eval(model)
    f1_score = results[0]["f1_macro"]
    print("Macro-averaged F1-Score:", f1_score)
    pprint.pprint(results)


if __name__ == "__main__":
    evaluate_ner()