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
DATA_DIR = BASE_DIR + "data/03_processed/Konvens2020/"
MODEL_DIR = BASE_DIR + "models/farm-ner-konvens2020"


def ner(task: str, lm: str):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    ##########################
    ########## Settings
    ##########################
    set_all_seeds(seed=42, deterministic_cudnn=use_cuda)
    use_amp = None
    device, n_gpu = initialize_device_settings(use_cuda=use_cuda, use_amp=use_amp)
    n_epochs = 10
    batch_size = 32
    evaluate_every = 1000
    model_dir = MODEL_DIR
    if lm == 'bert-hgcrw':
        lang_model = "redewiedergabe/bert-base-historical-german-rw-cased"
        model_dir += '_bert-hgcrw'
    elif lm == 'lmgot01':
        lang_model = Path("/home/stud/wangsadirdja/pyfarmbert/models/lm/lmgot_01")
        model_dir += '_lmgot01'
    elif lm == 'lmgot02':
        lang_model = Path("/home/stud/wangsadirdja/pyfarmbert/models/lm/lmgot_02")
        model_dir += '_lmgot02'
    else:
        lang_model = "bert-base-german-cased"
    if task != 'all':
        model_dir += '_' + task
    do_lower_case = False

    # 1.Create a tokenizer
    tokenizer = Tokenizer.load(
        pretrained_model_name_or_path=lang_model, do_lower_case=do_lower_case
    )

    # 2. Create a DataProcessor that handles all the conversion from raw text into a pytorch Dataset
    # See test/sample/ner/train-sample.txt for an example of the data format that is expected by the Processor
    if task == 'direct':
        ner_labels = ["[PAD]", "X", "O", "B-DIR", "I-DIR"]
    elif task == 'indirect':
        ner_labels = ["[PAD]", "X", "O", "B-IND", "I-IND"]
    elif task == 'reported':
        ner_labels = ["[PAD]", "X", "O", "B-REP", "I-REP"]
    else:
        ner_labels = ["[PAD]", "X", "O", "B-DIR", "I-DIR", "B-IND", "I-IND", "B-REP", "I-REP"]

    data_dir = DATA_DIR
    if task != 'all':
        data_dir += task + '/'
    processor = NERProcessor(
        tokenizer=tokenizer, max_seq_len=64, data_dir=Path(data_dir), delimiter="\t", metric="seq_f1", label_list=ner_labels
    )

    # 3. Create a DataSilo that loads several datasets (train/dev/test), provides DataLoaders for them and calculates a few descriptive statistics of our datasets
    data_loader_worker = 1
    data_silo = DataSilo(processor=processor, batch_size=batch_size, max_processes=data_loader_worker)

    # 4. Create an AdaptiveModel
    # a) which consists of a pretrained language model as a basis
    language_model = LanguageModel.load(lang_model)
    # b) and a prediction head on top that is suited for our task => NER
    prediction_head = TokenClassificationHead(num_labels=len(ner_labels))

    model = AdaptiveModel(
        language_model=language_model,
        prediction_heads=[prediction_head],
        embeds_dropout_prob=0.1,
        lm_output_types=["per_token"],
        device=device,
    )

    # 5. Create an optimizer
    model, optimizer, lr_schedule = initialize_optimizer(
        model=model,
        learning_rate=1e-5,
        n_batches=len(data_silo.loaders["train"]),
        n_epochs=n_epochs,
        device=device,
    )

    # 6. Feed everything to the Trainer, which keeps care of growing our model into powerful plant and evaluates it from time to time
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        data_silo=data_silo,
        epochs=n_epochs,
        n_gpu=n_gpu,
        lr_schedule=lr_schedule,
        evaluate_every=evaluate_every,
        device=device,
    )

    # 7. Let it grow
    trainer.train()

    # 8. Hooray! You have a model. Store it:
    save_dir = model_dir
    model.save(save_dir)
    processor.save(save_dir)


def infer_conll():
    # 9. Load it & harvest your fruits (Inference)
    basic_texts = [
        {"text": "Schartau sagte dem Tagesspiegel, dass Fischer ein Idiot sei"},
        {"text": "Martin Müller spielt Handball in Berlin"},
    ]
    # model = Inferencer.load(MODEL_DIR)
    # result1 = model.inference_from_dicts(dicts=basic_texts)
    # result2 = model.inference_from_file(DATA_DIR + 'test.txt')
    # pprint.pprint(result1[0])
    # with open("test_infer1.json", 'w') as fh:
    #    fh.write(pprint.pformat(result1, indent=2))
    # with open("test_infer2.json", 'w') as fh:
    #     json.dump(result2, fh, indent=2)

    # model.close_multiprocessing_pool()


if __name__ == "__main__":
    task = sys.argv[1]
    lang_model = sys.argv[2]
    # Parameter1 can be '', 'direct', 'indirect', 'reported'
    # Parameter2 can be 'lmgot01', 'lmgot02', 'bert-hgcrw', 'bert-gc'
    ner(task, lang_model)
    # infer_conll()
