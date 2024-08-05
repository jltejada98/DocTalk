from datasets import load_dataset
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
from datasets import Audio

ds = load_dataset("common_voice", "fr", split="test", streaming=True)
ds = ds.cast_column("audio", Audio(sampling_rate=16_000))