import argparse
import os
import random
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import gradio as gr
from gradio import processing_utils

from transformers import LlamaTokenizer, WhisperFeatureExtractor
from transformers import GenerationConfig
from src.modeling_blsp import BlspModel
from src.speech_text_paired_dataset import get_waveform


generation_config = GenerationConfig(
    max_new_tokens=128,
    min_new_tokens=1,
    do_sample=True,
    temperature=0.9,
    top_p=0.75,
    num_beams=1,
    num_return_sequences=1,
)


class ChatHistory(object):
    def __init__(self, tokenizer, extractor):
        super().__init__()
        self.tokenizer = tokenizer
        self.extractor = extractor
        self.history = []
        self.audio_file = []
        self.audio_to_history = True

        ### add bos token
        self.add_bos()

    def add_bos(self):
        input_ids = tokenizer("", return_tensors="pt").input_ids.cuda()
        self.history.append(
            (input_ids,)
        )

    def add_text_history(self, text):
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids[:,1:].cuda()
        self.history.append(
            (input_ids,)
        )

    def add_audio(self, audio_file):
        self.audio_to_history = False
        self.audio_file.append(audio_file)

    def add_speech_history(self, speech):
        if self.audio_to_history:
            return
        self.audio_to_history = True
        speech = get_waveform(speech, output_sample_rate=self.extractor.sampling_rate)
        speech_inputs = self.extractor(
            speech,
            sampling_rate=self.extractor.sampling_rate,
            return_attention_mask=True,
            return_tensors="pt"
        )
        speech_values = speech_inputs.input_features.cuda()
        speech_attention_mask = speech_inputs.attention_mask.cuda()
        self.history.append(
            (speech_values, speech_attention_mask)
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Chat Demo")
    parser.add_argument(
        "--blsp_model", type=str, default=None,
        help="Path to the blsp model", required=True
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
        "--temperature", type=float, default=0.1,
        help="temperature for generation"
    )
    parser.add_argument(
        "--top_p", type=float, default=0.75,
        help="top_p for generation"
    )
    args = parser.parse_args()
    return args


# ========================================
#             Model Initialization
# ========================================

print('Initializing Chat')
args = parse_args()

tokenizer = LlamaTokenizer.from_pretrained(args.blsp_model)
extractor = WhisperFeatureExtractor.from_pretrained(args.blsp_model)
model = BlspModel.from_pretrained(args.blsp_model)

generation_config.update(
    **{
        "max_new_tokens": args.max_new_tokens,
        "min_new_tokens": args.min_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "pad_token_id": tokenizer.pad_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id
    }
)

model = model.cuda()
model.eval()
history = ChatHistory(tokenizer, extractor)
print('Initialization Finished')


# ========================================
#             Gradio Setting
# ========================================


def gradio_reset():
    history.history = []
    history.audio_file = []
    history.add_bos()
    return None, gr.update(value="", interactive=True), gr.update(value=None, interactive=True), gr.update(value=None, interactive=True)


def gradio_answer(chatbot, num_beams, temperature):
    generation_config.update(
        **{
            "num_beams": num_beams, 
            "temperature": temperature,
        }
    )

    output = model.chat(
        history=history.history,
        generation_config=generation_config,
    )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    history.add_text_history(response + "\n\n\n")
    chatbot[-1][1] = ""
    for character in response:
        chatbot[-1][1] += character
        yield chatbot


title = """<h1 align="center">Demo of BLSP</h1>"""
description = """<h3>This is the demo of BLSP. Upload your audios and start chatting!</h3>"""
article = """<p><a href='https://xxx.github.io'><img src='https://xxx'></a></p><p><a href='https://github.com/xxx'><img src='https://xxx'></a></p><p><a href='xxx'><img src='xxx'></a></p>
"""


#TODO show examples below


def add_text(chatbot, user_message):
    chatbot = chatbot + [(user_message, None)]
    history.add_text_history("###[Human]:")
    history.add_text_history(user_message)
    history.add_text_history("\n\n\n###[Assistant]:")
    return chatbot, gr.update(value="", interactive=False)


def add_file(chatbot, gr_audio):
    history.add_text_history("###[Human]:")
    history.add_audio(gr_audio.name)
    history.add_speech_history(history.audio_file[-1])
    chatbot = chatbot + [((gr_audio.name,), None)]
    history.add_text_history("\n\n\n###[Assistant]:")
    return chatbot


def add_micophone_file(chatbot, gr_audio_mic):
    if gr_audio_mic is not None:
        history.add_text_history("###[Human]:")
        audio = processing_utils.audio_from_file(gr_audio_mic)
        # audio_ = processing_utils.convert_to_16_bit_wav(audio[1])
        processing_utils.audio_to_file(audio[0], audio[1], gr_audio_mic + '.wav')
        # os.rename(gr_audio_mic, gr_audio_mic + '.wav')
        gr_audio_mic_wav = gr_audio_mic+".wav"
        history.add_audio(gr_audio_mic_wav)
        history.add_speech_history(history.audio_file[-1])
        chatbot = chatbot + [((gr_audio_mic_wav,), None)]
        history.add_text_history("\n\n\n###[Assistant]:")
    return chatbot, gr.update(value=None, interactive=True)


with gr.Blocks() as demo:
    gr.Markdown(title)
    gr.Markdown(description)
    # gr.Markdown(article)

    chatbot = gr.Chatbot([], elem_id="chatbot", height=750, avatar_images=(None, (os.path.join(os.path.dirname(__file__), "avatar.png"))))

    with gr.Row():
        with gr.Column(scale=0.2, min_width=0, max_width=400):
            num_beams = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=1,
                    step=1,
                    interactive=True,
                    label="beam",
                )
                
            temperature = gr.Slider(
                minimum=0.1,
                maximum=2.0,
                value=1.0,
                step=0.1,
                interactive=True,
                label="Temp",
                )
        with gr.Column(scale=0.08, min_width=0, max_width=10):
            clear = gr.Button("Restart")
        with gr.Column(scale=0.85):
            txt = gr.Textbox(
                show_label=False,
                placeholder="Enter text and press enter, or upload an image",
                container=False)
        with gr.Column(scale=0.08, min_width=0, max_width=10):
            btn = gr.UploadButton("📁", file_types=["video", "audio"])
        with gr.Column(scale=0.2, min_width=0, max_width=400):
            input_audio_mic = gr.Audio(
                label="🎤",
                type="filepath",
                source="microphone",
                visible=True,
            )

    txt_msg = txt.submit(add_text, [chatbot, txt], [chatbot, txt], queue=False).then(
        gradio_answer, [chatbot, num_beams, temperature], chatbot
    )
    txt_msg.then(lambda: gr.update(interactive=True), None, [txt], queue=False)

    file_msg = btn.upload(add_file, [chatbot, btn], [chatbot], queue=False).then(
        gradio_answer, [chatbot, num_beams, temperature], chatbot
    )

    input_audio_mic.change(add_micophone_file, [chatbot, input_audio_mic], [chatbot, input_audio_mic], queue=False).then(
        gradio_answer, [chatbot, num_beams, temperature], chatbot
    )
    clear.click(gradio_reset, [], [chatbot, txt, input_audio_mic, btn], queue=False)

demo.queue()
demo.launch(share=False, enable_queue=True)
