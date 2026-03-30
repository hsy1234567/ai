import whisper

model = whisper.load_model("base")  # 可选 tiny, base, small, medium, large
result = model.transcribe("my_voice.mp4", language="zh")
print(result["text"])