""" anonymization process via INANO
"""

from INANO import INANO


# load model
inano = INANO()
vocoder = None

# input original speech
orig_wav_path = "original_speech.wav"
orig_wav = None

# anonymize speech
ano_wav_path = "anonymized_speech.wav"
ano_mel = inano(orig_wav)
ano_wav = vocoder(ano_mel)
