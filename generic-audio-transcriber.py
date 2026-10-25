import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from audio_transcriber import transcribe

result = transcribe("D:/speak_and_check/Т2С5.mp3", language="en", model="base", beam_size=3)

print(result.text)
