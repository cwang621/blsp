import logging
import os
import sys
import json
import fire
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Union, Tuple

import torch
from accelerate import Accelerator
from datasets import Dataset
from torch.utils.data.dataloader import DataLoader
from dataclasses import dataclass
from transformers import LlamaTokenizer, LlamaForCausalLM

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("continue writing")


Text_Format = (
    "###[Human]:Continue the following text in a coherent and engaging style with less than 40 words. {input}\n\n\n"
    "###[Assistant]:"
)


def get_shard_range(tot, nshard, rank):
    assert rank < nshard and rank >= 0, f"invaid rank/nshard {rank}/{nshard}"
    start = round(tot / nshard * rank)
    end = round(tot / nshard * (rank + 1))
    assert start < end, f"start={start}, end={end}"
    logger.info(
        f"rank {rank} of {nshard}, process {end-start} "
        f"({start}-{end}) out of {tot}"
    )
    return start, end


def get_dataset(manifest, nshard, rank):
    with open(manifest, "r") as f:
        lines = f.readlines()
        start, end = get_shard_range(len(lines), nshard, rank)
        lines = lines[start:end]
        lines = [json.loads(line.strip()) for line in lines]
    dataset = Dataset.from_list(lines)


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
        copy_tensor(torch.LongTensor(v), res[i][-len(v):])

    return res


@dataclass
class DataCollator:
    pad_id: int = 0

    def __call__(self, samples: List[Dict]):
        input_ids = [sample["input_ids"] for sample in samples]
        attention_mask = [sample["attention_mask"] for sample in samples]
        audio = [sample["audio"] for sample in samples]

        input_ids = collate_tokens(input_ids, self.pad_id)
        attention_mask = collate_tokens(attention_mask, 0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "audio": audio
        }
    

def continue_writing(
    llm_path,
    manifest,
    lab_dir,
    nshard=24,
    rank=0,
    batch_size=8
):
    accelerator = Accelerator()
    logger.info(accelerator.state)

    device = accelerator.device

    dataset = get_dataset(manifest, nshard, rank)
    tokenizer = LlamaTokenizer.from_pretrained(llm_path)

    def process_dataset(batch):
        batch["input_ids"] = tokenizer(Text_Format.format(input=batch["text"])).input_ids
        batch["attention_mask"] = [1] * len(batch["input_ids"])
        batch["audio"] = batch["audio"]
        batch["length"] = len(batch["input_ids"])
        return batch
    
    def is_in_length_range(length):
            return length > 0 and length < 192
    
    dataset = dataset.map(process_dataset)
    dataset = dataset.filter(is_in_length_range, input_columns=["length"])

    model = LlamaForCausalLM.from_pretrained(llm_path, torch_dtype=torch.float16)

    data_collator = DataCollator(tokenizer.pad_token_id)
    dataloader = DataLoader(
        dataset, 
        collate_fn=data_collator, 
        batch_size=batch_size
    )

    ### prepare everything
    model, dataloader = accelerator.prepare(
        model, dataloader
    )
    model.to(device)
    model.eval()

    split = os.path.splitext(os.path.basename(manifest))[0]
    lab_path = f"{lab_dir}/{split}_{rank}_{nshard}.jsonl"

    os.makedirs(lab_dir, exist_ok=True)

    progress_bar = tqdm(range(len(dataloader)), disable=not accelerator.is_local_main_process)
    with open(lab_path, "w") as f:
        for batch in dataloader:
            outputs = model.generate(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                max_new_tokens=64,
                do_sample=False,
                num_beams=1,
                top_p=0.75,
                temperature=0.1,
                num_return_sequences=1,
            )
            input_length = batch["input_ids"].shape[1]
            generated_tokens = outputs[:, input_length:]
            output_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            for audio, text in zip(batch["audio"], output_text):
                json_string = json.dumps(
                    {
                        "audio": audio,
                        "text": text
                    }
                )
                print(json_string, file=f)
            progress_bar.update(1)

    logger.info("finished successfully")
    



if __name__ == "__main__":
    fire.Fire({
        'continue_writing': continue_writing,
    })