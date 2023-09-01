import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, BinaryIO
import fire

import numpy as np
import torch
import random
import datasets
from dataclasses import dataclass

from transformers import LlamaTokenizer

logger = logging.getLogger(__name__)


Text_Format = (
    "###[Human]:{instruction}\n\n\n"
    "###[Assistant]:"
)

def process_dataset(batch, tokenizer, max_length):
    instruction = batch["instruction"]
    output = batch["output"]

    input = Text_Format.format(instruction=instruction)
    output = output + tokenizer.eos_token

    input_ids = tokenizer(input).input_ids
    output_ids = tokenizer(output).input_ids[1:] # remove bos token

    batch["input_ids"] = (input_ids + output_ids)[:max_length]
    batch["attention_mask"] = ([1] * (len(input_ids) + len(output_ids)))[:max_length]
    batch["labels"] = ([-100] * len(input_ids) + output_ids)[:max_length]

    return batch

def load_text_instruction_dataset(
    dataroot="",
    manifest_files="",
    max_length=384,
    tokenizer=None,
):
    if os.path.exists(os.path.join(dataroot, f"processed_{manifest_files}")):
        logger.warning("load processed dataset")
        dataset = datasets.load_from_disk(os.path.join(dataroot, f"processed_{manifest_files}"))
        return dataset
    
    logger.warning(f"load dataset from scratch from {dataroot}/{manifest_files}")
    
    manifest_files_list = manifest_files.split(",")

    raw_dataset = datasets.load_dataset(
        dataroot, data_files=manifest_files_list, split="train", streaming=False
    )

    dataset = raw_dataset.map(
        process_dataset,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_length": max_length,
        },
        remove_columns=raw_dataset.column_names,
        load_from_cache_file=False
    )

    dataset.save_to_disk(os.path.join(dataroot, f"processed_{manifest_files}"))

    return dataset


def collate_tokens(
        values: List[List[int]],
        pad_id: int
):
    size = max(len(v) for v in values)
    batch_size = len(values)
    res = torch.LongTensor(batch_size, size).fill_(pad_id)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(torch.LongTensor(v), res[i][: len(v)])

    return res


@dataclass
class TextInstructionDataCollator:
    """
    Data collator that will dynamically pad the inputs received.
    """
    pad_id: int = 0

    def __call__(self, samples: List[Dict]):
        input_ids = [sample["input_ids"] for sample in samples]
        attention_mask = [sample["attention_mask"] for sample in samples]
        labels = [sample["labels"] for sample in samples]

        input_ids = collate_tokens(input_ids, self.pad_id)
        attention_mask = collate_tokens(attention_mask, 0)
        labels = collate_tokens(labels, -100)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def offline_process(
    dataroot="",
    manifest_files="",
    max_length=384,
    lm_path="",
):
    text_tokenizer = LlamaTokenizer.from_pretrained(lm_path)

    dataset = load_text_instruction_dataset(
        dataroot,
        manifest_files,
        max_length,
        text_tokenizer,
    )
    for key in dataset[0].keys():
        print(key, len(dataset[0][key]))


if __name__ == "__main__":
    fire.Fire({
        "offline": offline_process,
    })