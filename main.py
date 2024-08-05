from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import Audio, load_dataset
import IPython.display as ipd

# load model and processor
processor = WhisperProcessor.from_pretrained("openai/whisper-base")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
forced_decoder_ids = processor.get_decoder_prompt_ids(language="french", task="translate")

# load streaming dataset and read first audio sample
ds = load_dataset("mozilla-foundation/common_voice_11_0", "fr", split="test", streaming=True)
ds = ds.cast_column("audio", Audio(sampling_rate=16_000))
input_speech = next(iter(ds))["audio"]

#Play Audio
ipd.display(ipd.Audio(data=input_speech["array"], autoplay=True, rate=input_speech["sampling_rate"]))

#Commence processing
input_features = processor(input_speech["array"], sampling_rate=input_speech["sampling_rate"], return_tensors="pt").input_features

# generate token ids
predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
# decode token ids to text
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
print(transcription)