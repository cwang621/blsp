import pathlib
import json
import argparse
import tqdm
import os

from whisper.normalizers import EnglishTextNormalizer

normalizer = EnglishTextNormalizer()


def process(args):
    with open(args.output_file, "w") as fout:
        with open(os.path.join(args.input_dir, "audio_data", "GigaSpeech.json"), "r") as fin:
            data = json.load(fin)
            for audio in tqdm.tqdm(data["audios"]):
                root = "/".join(audio["path"].split("/")[:-1])
                for segment in audio["segments"]:
                    if args.split not in segment["subsets"]:
                        continue
                    sid = segment["sid"]
                    text = segment["text_tn"]
                    path = os.path.join(args.input_dir, "audio_data", root, sid+".wav")
                    text = text.replace("<COMMA>", ",").replace("<PERIOD>", ".")
                    text = text.replace("<QUESTIONMARK>", "?").replace("<EXCLAMATIONPOINT>", "!")
                    text = normalizer(text)
                    json_string = json.dumps({
                        "audio": path,
                        "text": text
                    })
                    fout.write(json_string + "\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", type=str, default=None,
                        help="Path to the input directory", required=True)
    parser.add_argument("--output_file", type=str, default=None,
                        help="Path to the output manifest file", required=True)
    parser.add_argument("--split", type=str, default="XL",
                        help="choose from [XS,S,M,L,XL]")
    
    args = parser.parse_args()

    args.split = "{" + args.split + "}"

    process(args)