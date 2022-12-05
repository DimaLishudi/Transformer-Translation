from enum import Enum
from pathlib import Path

from tokenizers import Tokenizer
from torch.utils.data import Dataset

import torch
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer


def process_training_file(input_path: Path, output_path: Path):
    """
    Processes raw training files ("train.tags.SRC-TGT.*"), saving the output as a sequence of unformatted examples
    (.txt file, one example per line).
    :param input_path: Path to the file with the input data (formatted examples)
    :param output_path: Path to the file with the output data (one example per line)
    """
    with open(input_path, "r") as input_file, open(output_path, "w") as output_file:
        for line in input_file:
            line = line.strip()
            if line.startswith("<") and line.endswith(">"):
                continue
            output_file.write(line + "\n")


def process_evaluation_file(input_path: Path, output_path: Path):
    """
    Processes raw validation and testing files ("IWSLT17.TED.{dev,test}2010.SRC-TGT.*.xml"),
    saving the output as a sequence of unformatted examples (.txt file, one example per line).
    :param input_path: Path to the file with the input data (formatted examples)
    :param output_path: Path to the file with the output data (one example per line)
    """
    with open(input_path, "r") as input_file, open(output_path, "w") as output_file:
        for line in input_file:
            line = line.strip()
            if line.startswith("<seg id="):
                start  = line.find(">")
                finish = line.rfind("<")
                output_file.write(line[start+1:finish].strip() + "\n")



def convert_files(base_path: Path, output_path: Path):
    """
    Given a directory containing all the dataset files, convert each one into the "one example per line" format.
    :param base_path: Path containing files with original data
    :param output_path: Path containing files with processed data
    """

    for language in "de", "en":
        process_training_file(
            base_path / f"train.tags.de-en.{language}",
            output_path / f"train.{language}.txt",
        )
        process_evaluation_file(
            base_path / f"IWSLT17.TED.dev2010.de-en.{language}.xml",
            output_path / f"val.{language}.txt",
        )
        process_evaluation_file(
            base_path / f"IWSLT17.TED.tst2010.de-en.{language}.xml",
            output_path / f"test.{language}.txt",
        )


class TranslationDataset(Dataset):
    def __init__(
        self,
        src_file_path,
        tgt_file_path,
        src_tokenizer: Tokenizer,
        tgt_tokenizer: Tokenizer,
        max_len=32,
        hold_texts=False,
    ):
        """
        Loads the training dataset and parses it into separate tokenized training examples.
        No padding should be applied at this stage
        :param src_file_path: Path to the source language training data
        :param tgt_file_path: Path to the target language training data
        :param src_tokenizer: Trained tokenizer for the source language
        :param tgt_tokenizer: Trained tokenizer for the target language
        :param max_len: Maximum length of source and target sentences for each example:
        if either of the parts contains more tokens, it needs to be filtered.
        """
        self.src_file_path = src_file_path
        self.tgt_file_path = tgt_file_path
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        # enable texts for debug/extra logging (but slower dataloading)
        self.hold_texts = hold_texts

        self.src_tokenizer.enable_truncation(max_length=max_len)
        self.tgt_tokenizer.enable_truncation(max_length=max_len)
        
        if self.hold_texts:
            self.src_lines = []
            self.tgt_lines = []

        self.src_ids = []

        with open(src_file_path, "r") as src_file:
            for line in src_file:
                if self.hold_texts:
                    self.src_lines.append(line.strip())
                self.src_ids.append(torch.asarray(self.src_tokenizer.encode(line).ids, dtype=torch.long))

        self.tgt_ids = []
        with open(tgt_file_path, "r") as tgt_file:
            for line in tgt_file:
                if self.hold_texts:
                    self.tgt_lines.append(line.strip())
                self.tgt_ids.append(torch.asarray(self.tgt_tokenizer.encode(line).ids, dtype=torch.long))

        # self.src_tokenizer.disable_truncation()
        # self.tgt_tokenizer.disable_truncation()

        

    def __len__(self):
        return len(self.src_ids)

    def __getitem__(self, i):
        if self.hold_texts:
            return self.src_ids[i], self.tgt_ids[i], self.src_lines[i], self.tgt_lines[i]
        return self.src_ids[i], self.tgt_ids[i]

    def collate_translation_data(self, batch):
        """
        Given a batch of examples with varying length, collate it into `source` and `target` tensors for the model.
        This method is meant to be used when instantiating the DataLoader class for training and validation datasets in your pipeline.
        """

        pad_tokens = {
            "src" : self.src_tokenizer.token_to_id("[PAD]"),
            "tgt" : self.tgt_tokenizer.token_to_id("[PAD]"),
        }

        ids_lists = {
            "src" : [],
            "tgt" : [],
        }
        src_texts = []
        tgt_texts = []

        for src_ids, tgt_ids, src_text, tgt_text in batch:
            ids_lists["src"].append(src_ids)
            ids_lists["tgt"].append(tgt_ids)
            if self.hold_texts:
                src_texts.append(src_text)
                tgt_texts.append(tgt_text)

        batch = {}

        for lang in ["src", "tgt"]:
            batch[lang] = torch.nn.utils.rnn.pad_sequence(
                sequences=ids_lists[lang],
                batch_first=True,
                padding_value=pad_tokens[lang]
            )

        if self.hold_texts:
            batch["src_text"] = src_texts
            batch["tgt_text"] = tgt_texts
    
        # for lang in ["src", "target"]:
        #     ids_list = batch[f"{lang}_ids"]
        #     batch_max_len = max([len(ids) for ids in ids_list])
        #     # initially ids_tensor is filled with "PAD"
        #     ids_tensor = torch.ones(len(ids_list), batch_max_len) * pad_tokens[lang]

        #     for i in range(len(ids_list)):
        #         ids = ids_list[i]
        #         ids_tensor[:len(ids)] = torch.asarray(ids, dtype=torch.long)
        #     batch[f"{lang}_ids"] = ids_tensor

        return batch
        

class SpecialTokens(Enum):
    UNKNOWN = "[UNK]"
    PADDING = "[PAD]"
    BEGINNING = "[BOS]"
    END = "[EOS]"


def train_tokenizers(base_dir: Path, save_dir: Path):
    """
    Trains tokenizers for source and target languages and saves them to `save_dir`.
    :param base_dir: Directory containing processed training and validation data (.txt files from `convert_files`)
    :param save_dir: Directory for storing trained tokenizer data (two files: `tokenizer_de.json` and `tokenizer_en.json`)
    """

    for language in ["de", "en"]:
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        src_files = [str(base_dir / f"train.{language}.txt"), str(base_dir / f"val.{language}.txt")]
        trainer = BpeTrainer(special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"], vocab_size=30000)
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer.post_processor = TemplateProcessing(
            single="[BOS] $A [EOS]",
            special_tokens=[
                ("[BOS]", 2),
                ("[EOS]", 3),
            ],
        )
        tokenizer.train(src_files, trainer)
        tokenizer.save(str(save_dir / f"tokenizer_{language}.json"))