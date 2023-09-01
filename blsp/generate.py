import os
import argparse
import json
from tqdm import tqdm

from transformers import LlamaTokenizer, WhisperFeatureExtractor
from transformers import GenerationConfig
from src.modeling_blsp import BlspModel
from src.speech_text_paired_dataset import get_waveform

generation_config = GenerationConfig(
    max_new_tokens=128,
    min_new_tokens=1,
    do_sample=False,
    temperature=0.1,
    top_p=0.75,
    num_beams=1,
    num_return_sequences=1,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file", type=str, default=None,
        help="Path to the input file", required=True
    )
    parser.add_argument(
        "--output_file", type=str, default=None,
        help="Path to the output file", required=True
    )
    parser.add_argument(
        "--blsp_model", type=str, default=None,
        help="Path to the blsp model", required=True
    )
    parser.add_argument(
        "--instruction", type=str, default="",
        help="the general instruction for each example"
    )
    ### args for generation
    parser.add_argument(
        "--max_new_tokens", type=int, default=128,
        help="max new tokens for generation"
    )
    parser.add_argument(
        "--min_new_tokens", type=int, default=1,
        help="min new tokens for generation"
    )
    parser.add_argument(
        "--do_sample", action="store_true",
        help="whether do sample. For ST task, we will use greedy search to ensure stable output"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.1,
        help="temperature for generation"
    )
    parser.add_argument(
        "--top_p", type=float, default=0.75,
        help="top_p for generation"
    )
    args = parser.parse_args()


    tokenizer = LlamaTokenizer.from_pretrained(args.blsp_model)
    extractor = WhisperFeatureExtractor.from_pretrained(args.blsp_model)
    model = BlspModel.from_pretrained(args.blsp_model)

    generation_config.update(
        **{
            "max_new_tokens": args.max_new_tokens,
            "min_new_tokens": args.min_new_tokens,
            "do_sample": args.do_sample,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "pad_token_id": tokenizer.pad_token_id,
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id
        }
    )

    model = model.cuda()
    model.eval()
    with open(args.input_file, "r") as fin, open(args.output_file, "w") as fout:
        for line in tqdm(fin):
            data = json.loads(line.strip())

            instruction = data.get("instruction", args.instruction)
            input_ids = tokenizer(f"###[Human]:{instruction}", return_tensors="pt").input_ids.cuda()

            audio = data.get("audio", None)
            speech_values, speech_attention_mask = None, None
            if audio is not None:
                speech = get_waveform(audio, output_sample_rate=extractor.sampling_rate)
                speech_inputs = extractor(
                    speech,
                    sampling_rate=extractor.sampling_rate,
                    return_attention_mask=True,
                    return_tensors="pt"
                )
                speech_values = speech_inputs.input_features.cuda()
                speech_attention_mask = speech_inputs.attention_mask.cuda()
            
            suffix_input_ids = tokenizer("\n\n\n###[Assistant]:", return_tensors="pt").input_ids[:,1:].cuda()
            reference = data.get("reference", "")

            output = model.generate(
                input_ids=input_ids,
                suffix_input_ids=suffix_input_ids,
                speech_values=speech_values,
                speech_attention_mask=speech_attention_mask,
                generation_config=generation_config,
            )
            response = tokenizer.decode(output[0], skip_special_tokens=True)

            json_string = json.dumps(
                {
                    "response": response,
                    "reference": reference
                },
                ensure_ascii=False
            )
            fout.write(json_string + "\n")
            

if __name__ == "__main__":
    main()