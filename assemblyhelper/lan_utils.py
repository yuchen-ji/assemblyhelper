import whisper
whisper_model = whisper.load_model("large")
result = whisper_model.transcribe(r"/workspaces/assemblyhelper/eval/speech/16k16bit.mp3")
print(", ".join([i["text"] for i in result["segments"] if i is not None]))

# # Note: you need to be using OpenAI Python v0.27.0 for the code below to work
# import openai
# audio_file= open("/path/to/file/audio.mp3", "rb")
# transcript = openai.Audio.transcribe("whisper-1", audio_file)