import json
import argparse
from tqdm import tqdm
import os
import string
import re


def clean_text(text):
        text = text.replace("\u2019", "'").replace("\u00b0C", "  degrees centigrade").replace("\u00b0F", " degrees fahrenheit")
        printable = string.printable
        text = re.sub(f'[^{re.escape(printable)}]', '', text)
        return text

def process(args):
    with open(args.input_file, "r") as f:
        dataset = json.load(f)

    with open(args.output_file, "w") as f:
        for data in tqdm(dataset):
            instruction = data["instruction"]
            if not (instruction.endswith(".") or instruction.endswith("!") or instruction.endswith("?")):
                instruction += "."
            if data["input"] != "":
                instruction = instruction + " " + data["input"]
            instruction = clean_text(instruction)
            output = clean_text(data["output"])

            json_string = json.dumps(
                {
                    "instruction": instruction,
                    "output": output
                }
            )
            f.write(json_string + "\n")
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file", type=str, default=None,
                        help="Path to the input json file", required=True)
    parser.add_argument("--output_file", type=str, default=None,
                        help="Path to the output manifest file", required=True)
    
    args = parser.parse_args()

    process(args)