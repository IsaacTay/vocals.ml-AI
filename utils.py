import librosa

def importAudio(path):
  audio, _ = librosa.load(path, sr=16000)
  return audio
