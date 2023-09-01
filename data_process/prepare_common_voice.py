import json
import argparse
from tqdm import tqdm
import os
from  pathlib import Path

import csv
import pandas as pd

from torchaudio.datasets import LIBRISPEECH

from whisper.normalizers import EnglishTextNormalizer

normalizer = EnglishTextNormalizer()


def process(args):
    input_dir = Path(args.input_dir).absolute()
    cv_tsv_path = os.path.join(input_dir, f"{args.split}.tsv")
    cv_tsv = pd.read_csv(
        cv_tsv_path,
        sep="\t",
        header=0,
        encoding="utf-8",
        escapechar="\\",
        quoting=csv.QUOTE_NONE,
        na_filter=False
    )
    dataset = cv_tsv.to_dict(orient="index").items()
    dataset = [v for k,v in sorted(dataset, key=lambda x: x[0])]
    with open(args.output_file, "w") as fout:
        for data in tqdm(dataset):
            audio = os.path.join(input_dir, "clips", data["path"])
            text = normalizer(data["sentence"])
            json_string = json.dumps({
                "audio": audio,
                "text": text
            })
            fout.write(json_string + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", type=str, default=None,
                        help="Path to the input directory", required=True)
    parser.add_argument("--output_file", type=str, default=None,
                        help="Path to the output manifest file", required=True)
    parser.add_argument("--split", type=str, default="train",
                        help="choose from [train, dev, test]")
    
    args = parser.parse_args()

    process(args)