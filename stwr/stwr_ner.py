import os
import logging
from pathlib import Path
from decimal import Decimal

from farm.data_handler.data_silo import DataSilo
from farm.data_handler.processor import NERProcessor
from farm.modeling.optimization import initialize_optimizer
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


def stwr_tag(args):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    ##########################
    ########## Settings
    ##########################
    seed = int(args.seed)
    set_all_seeds(seed=seed, deterministic_cudnn=use_cuda)
    device, n_gpu = initialize_device_settings(use_cuda=use_cuda, use_amp=None)
    n_epochs = int(args.epochs)
    batch_size = int(args.batch_size)
    evaluate_every = int(args.evaluate_every)
    model_dir = BASE_DIR + args.model_base_dir + args.exp_id
    if args.lm == 'bert-hgcrw':
        lang_model = "redewiedergabe/bert-base-historical-german-rw-cased"
    elif args.lm == 'lmgot01':
        lang_model = Path("/home/stud/wangsadirdja/pyfarmbert/models/lm/lmgot_01")
    elif args.lm == 'lmgot02':
        lang_model = Path("/home/stud/wangsadirdja/pyfarmbert/models/lm/lmgot_02")
    elif args.lm == 'bert-gc':
        lang_model = "bert-base-german-cased"
    else:
        lang_model = Path(args.lm)

    do_lower_case = False

    # 1.Create a tokenizer
    tokenizer = Tokenizer.load(
        pretrained_model_name_or_path=lang_model, do_lower_case=do_lower_case
    )

    # 2. Create a DataProcessor that handles all the conversion from raw text into a pytorch Dataset
    # See test/sample/ner/train-sample.txt for an example of the data format that is expected by the Processor
    if args.task == 'direct':
        ner_labels = ["[PAD]", "X", "O", "B-DIR", "I-DIR"]
    elif args.task == 'indirect':
        ner_labels = ["[PAD]", "X", "O", "B-IND", "I-IND"]
    elif args.task == 'reported':
        ner_labels = ["[PAD]", "X", "O", "B-REP", "I-REP"]
    else:
        ner_labels = ["[PAD]", "X", "O", "B-DIR", "I-DIR", "B-IND", "I-IND", "B-REP", "I-REP"]

    data_dir = BASE_DIR + args.data_dir
    if args.task != 'all':
        data_dir += args.task + '/'
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

    dropout = float(args.dropout)
    model = AdaptiveModel(
        language_model=language_model,
        prediction_heads=[prediction_head],
        embeds_dropout_prob=dropout,
        lm_output_types=["per_token"],
        device=device,
    )

    # 5. Create an optimizer
    learning_rate = float(args.learning_rate)
    model, optimizer, lr_schedule = initialize_optimizer(
        model=model,
        learning_rate=learning_rate,
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
