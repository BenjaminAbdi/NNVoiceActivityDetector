import numpy as np
import SignalExceptions
from Signal import Signal
from scipy.io.wavfile import read

def getSignalFromFilename(filename):
    try:
        (samp_freq, data) = read(filename)
    except Exception as e :
        raise SignalExceptions.NotAvalidWavFileException(filename=filename)
    return Signal(np.array(data), samp_freq)

def shittyTestToDiscardBeforeHavingAshittyFile():
    signal = getSignalFromFilename("./processing-tools/dataTest/felicitations.wav")
    print(signal.feature_vectors.shape)
    print(signal.signal.shape)
    print(signal.getDuration())

shittyTestToDiscardBeforeHavingAshittyFile()
