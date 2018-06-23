import librosa

def importAudio(path):
  audio, _ = librosa.load(path, sr=16000)
  return audio

def toULaw(audio):
  audio = np.floor(np.sign(audio)*(np.log(1+255*np.abs(audio))/np.log(256)) * 127.5 + 128)
  return audio

def toLin(output):
  output = output/127.5 - 1
  audio = np.sign(output) * (np.power(256,output) - 1) / 255
  return audio
