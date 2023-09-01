import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, BinaryIO
import fire
import soundfile as sf

import numpy as np
import torch
import random
import datasets
from dataclasses import dataclass

from transformers import LlamaTokenizer, WhisperFeatureExtractor

logger = logging.getLogger(__name__)


def process_dataset(batch, tokenizer, instruction):
    input_ids = tokenizer(f"###[Human]:{instruction}").input_ids
    attention_mask = [1] * len(input_ids)
    labels = [-100] * len(input_ids)

    audio_path = batch["audio"]
    try:
        info = sf.info(audio_path)
        is_readable = True
    except:
        is_readable = False


    suffix_input_ids, suffix_attention_mask, suffix_labels = [], [], []
    ### \n\n\n###[Assistant]:
    new_input_ids = tokenizer("\n\n\n###[Assistant]:").input_ids[1:] # remove bos token
    suffix_input_ids += new_input_ids
    suffix_attention_mask += [1] * len(new_input_ids)
    suffix_labels += [-100] * len(new_input_ids)
    ### response
    new_input_ids = tokenizer(batch["text"]).input_ids[1:] + [tokenizer.eos_token_id]
    suffix_input_ids += new_input_ids
    suffix_attention_mask += [1] * len(new_input_ids)
    suffix_labels += new_input_ids
    

    batch["input_ids"] = input_ids
    batch["attention_mask"] = attention_mask
    batch["labels"] = labels
    batch["suffix_input_ids"] = suffix_input_ids
    batch["suffix_attention_mask"] = suffix_attention_mask
    batch["suffix_labels"] = suffix_labels
    batch["audio_path"] = audio_path
    batch["is_readable"] = is_readable

    return batch

def load_speech_text_paired_dataset(
    dataroot="",
    manifest_files="",
    tokenizer=None,
    instruction="",
    num_proc=8,
):
    if os.path.exists(os.path.join(dataroot, f"processed_{manifest_files}".replace("*", "all"))):
        logger.warning("load processed dataset")
        dataset = datasets.load_from_disk(os.path.join(dataroot, f"processed_{manifest_files}".replace("*", "all")))
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
            "instruction": instruction
        },
        remove_columns=raw_dataset.column_names,
        load_from_cache_file=False,
        num_proc=num_proc,
    )

    def is_readable(flag):
        return flag
    
    dataset = dataset.filter(
        is_readable,
        input_columns=["is_readable"]
    )

    dataset.save_to_disk(os.path.join(dataroot, f"processed_{manifest_files}".replace("*", "all")))

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

def get_waveform(
    path_or_fp: Union[str, BinaryIO],
    normalization=True,
    mono=True,
    frames=-1,
    start=0,
    always_2d=False,
    output_sample_rate=16000,
) -> Tuple[np.ndarray, int]:
    meta = path_or_fp.split(":")
    if len(meta) == 3 and (meta[0].endswith(".wav") or meta[0].endswith(".flac")):
        path_or_fp = meta[0]
        start = int(meta[1])
        frames = int(meta[2])
    else:
        path_or_fp = path_or_fp


    if isinstance(path_or_fp, str):
        ext = Path(path_or_fp).suffix
        if ext in [".wav", ".flac", ".ogg", ".mp3"]:
            pass
        else:
            raise ValueError(f"Unsupported audio format: {ext}")

    try:
        import soundfile as sf
    except ImportError:
        raise ImportError("Please install soundfile to load WAV/FLACC/OGG/MP3 audios")
    waveform, sample_rate = sf.read(
        path_or_fp, dtype="float32", always_2d=True, frames=frames, start=start
    )
    waveform = waveform.T

    waveform, sample_rate = convert_waveform(waveform, sample_rate, to_mono=mono, to_sample_rate=output_sample_rate)
    if not normalization:
        waveform *= 2 ** 15
    if not always_2d:
        waveform = waveform.squeeze(axis=0)
    return waveform

def convert_waveform(
    waveform: Union[np.ndarray, torch.Tensor],
    sample_rate: int,
    normalize_volume: bool = False,
    to_mono: bool = False,
    to_sample_rate: Optional[int] = None,
) -> Tuple[Union[np.ndarray, torch.Tensor], int]:
    """convert a waveform:
    - to a target sample rate
    - from multi-channel to mono channel
    - volume normalization
    Args:
        waveform (numpy.ndarray or torch.Tensor): 2D original waveform
            (channels x length)
        sample_rate (int): original sample rate
        normalize_volume (bool): perform volume normalization
        to_mono (bool): convert to mono channel if having multiple channels
        to_sample_rate (Optional[int]): target sample rate
    Returns:
        waveform (numpy.ndarray): converted 2D waveform (channels x length)
        sample_rate (float): target sample rate
    """
    try:
        import torchaudio.sox_effects as ta_sox
    except ImportError:
        raise ImportError("Please install torchaudio: pip install torchaudio")

    effects = []
    if normalize_volume:
        effects.append(["gain", "-n"])
    if to_sample_rate is not None and to_sample_rate != sample_rate:
        effects.append(["rate", f"{to_sample_rate}"])
    if to_mono and waveform.shape[0] > 1:
        effects.append(["channels", "1"])
    if len(effects) > 0:
        is_np_input = isinstance(waveform, np.ndarray)
        _waveform = torch.from_numpy(waveform) if is_np_input else waveform
        converted, converted_sample_rate = ta_sox.apply_effects_tensor(
            _waveform, sample_rate, effects
        )
        if is_np_input:
            converted = converted.numpy()
        return converted, converted_sample_rate
    return waveform, sample_rate


@dataclass
class SpeechTextPairedDataCollator:
    """
    Data collator that will dynamically pad the inputs received.
    """
    pad_id: int = 0
    sampling_rate: int = 16000
    extractor: WhisperFeatureExtractor = WhisperFeatureExtractor()

    def __call__(self, samples: List[Dict]):
        input_ids = [sample["input_ids"] for sample in samples]
        attention_mask = [sample["attention_mask"] for sample in samples]
        labels = [sample["labels"] for sample in samples]
        suffix_input_ids = [sample["suffix_input_ids"] for sample in samples]
        suffix_attention_mask = [sample["suffix_attention_mask"] for sample in samples]
        suffix_labels = [sample["suffix_labels"] for sample in samples]

        input_ids = collate_tokens(input_ids, self.pad_id)
        attention_mask = collate_tokens(attention_mask, 0)
        labels = collate_tokens(labels, -100)
        suffix_input_ids = collate_tokens(suffix_input_ids, self.pad_id)
        suffix_attention_mask = collate_tokens(suffix_attention_mask, 0)
        suffix_labels = collate_tokens(suffix_labels, -100)

        raw_speech = [
            get_waveform(sample["audio_path"], output_sample_rate=self.sampling_rate) for sample in samples
        ]
        speech_inputs = self.extractor(
            raw_speech, 
            sampling_rate=self.sampling_rate, 
            return_attention_mask=True,
            return_tensors="pt"
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "suffix_input_ids": suffix_input_ids,
            "suffix_attention_mask": suffix_attention_mask,
            "suffix_labels": suffix_labels,
            "speech_values": speech_inputs.input_features,
            "speech_attention_mask": speech_inputs.attention_mask
        }


def offline_process(
    dataroot="",
    manifest_files="",
    lm_path="",
    instruction="",
    num_proc=8,
):
    text_tokenizer = LlamaTokenizer.from_pretrained(lm_path)

    dataset = load_speech_text_paired_dataset(
        dataroot,
        manifest_files,
        text_tokenizer,
        instruction,
        num_proc
    )
    for key in dataset[0].keys():
        if key != "audio_path" and key != "is_readable":
            print(key, len(dataset[0][key]))
        else:
            print(key, dataset[0][key])
    print(len(dataset))


if __name__ == "__main__":
    fire.Fire({
        "offline": offline_process,
    })