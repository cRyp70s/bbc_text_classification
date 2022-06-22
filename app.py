from typing import Union
import torch
import joblib
import gradio as gr
import typer
from model import Model

from utils import *

vocab = joblib.load("bin/vocab.bin")
le = joblib.load("bin/label_encoder.bin")
net = Model(100, 16, 1, len(vocab), le.classes_.shape[0])
net.load_state_dict(torch.load("bin/bbc_classification.pth"))

# Convert text to list of word indexes in the vocabulary.
# Padded or truncated if necessary.
word2paddedindex = lambda text: [
    vocab.get_index(word) for word in pad(text.split(), 200)
]


def predict(text: str) -> tuple:
    """
    Predict class for text.
    returns the class and confidence
    """
    text = preprocess(text)
    token_indexes = word2paddedindex(text)
    tensor = torch.tensor([token_indexes])
    preds = net(tensor)
    index = preds.argmax(axis=1)

    return le.inverse_transform(index)[0], float(torch.softmax(preds, 1)[0][index])


def start_gr(
    share: bool = False,
    inputs: Union[str, list] = "text",
    outputs: Union[str, list] = ["text", "number"],
):
    demo = gr.Interface(fn=predict, inputs=inputs, outputs=outputs)
    demo.launch(share=share)


def main(
    mode: Modes = typer.Argument(
        ...,
        help="Mode to run in. cli to run in command \
            line or gradio to launch a gradio app",
    ),
    share: bool = typer.Option(False, help="share option for gradio launch"),
    text: str = typer.Argument(
        None,
        help="Text to classify.\
        Can only be used in cli mode",
    ),
):
    if mode == Modes.CLI:
        typer.echo(predict(text))
    elif mode == Modes.GRADIO:
        start_gr(share)


if __name__ == "__main__":
    typer.run(main)
