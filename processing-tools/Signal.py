import numpy as np
from scipy import signal as scipySignal
import python_speech_features
from Timer import Timer
SAMPLE_FREQ_CONST = 16000
timer = Timer(["resample", "feat"])

class Signal:
    def __init__(self, np_array_of_values, sample_freq, ms_per_frame=25):
        is_np_array = isinstance(np_array_of_values, np.ndarray)
        assert(is_np_array and sample_freq > 0 and isinstance(sample_freq, int))
        self.signal = resample(np_array_of_values, initial_freq=sample_freq, new_freq=SAMPLE_FREQ_CONST)
        timer.storeTime("resample")
        self.sample_freq = SAMPLE_FREQ_CONST
        samples_per_ms = sample_freq / 1000
        self.samples_per_frame = np.int(np.floor(samples_per_ms * ms_per_frame))
        self.n_of_frames = np.int(np.ceil(len(self.signal) / self.samples_per_frame))
        #self.framed_signal = self.splitInFrames()
        self.feature_vectors = signalToFeatureVector(self.signal, SAMPLE_FREQ_CONST, ms_per_frame)
        timer.storeTime("feat")
        timer.showResults()


    def splitInFrames(self):
        n_zeros_to_add = (self.samples_per_frame - len(self.signal) % self.samples_per_frame) % self.samples_per_frame
        padded_signal = np.lib.pad(self.signal, (0, n_zeros_to_add), 'constant')
        return np.reshape( padded_signal, (self.n_of_frames, self.samples_per_frame) )

    def getDuration(self):
        return len(self.signal) / self.sample_freq


def signalToFeatureVector(signal, sample_rate, ms_per_frame):
    mfcc = python_speech_features.mfcc(signal, sample_rate, winlen=ms_per_frame/1000, winstep=0.025, nfft=2048)
    mfcc_delta = python_speech_features.delta(mfcc, 1)
    mfcc_delta_delta = python_speech_features.delta(mfcc, 2)
    return np.concatenate((mfcc, mfcc_delta, mfcc_delta_delta), axis=1)

def resample(signal, initial_freq, new_freq):
    if (initial_freq == new_freq):
        return signal
    n = signal.shape[0]
    prevExp = np.floor(np.log2(n))
    nextpow2  = np.int(np.power(2, prevExp + 1))
    signal  = np.pad(signal, ((0, nextpow2-n)), mode='constant')

    new_num_of_samples = np.int(np.ceil(len(signal) / initial_freq * new_freq))
    return scipySignal.resample(signal, new_num_of_samples)


def testResample():
    signal1 = [0, 1, 2]
    resample1 = resample(signal1, 44100, 44100)
    assert(resample1 == signal1)

    signal2 = [0, 1, 2, 3]
    resample2 = resample(signal2, 44100, 22050)
    assert(len(signal2) == len(resample2) * 2)

    signal3 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    resample3 = resample(signal3, 10000, 2000)
    assert(len(signal3) == len(resample3) * 5)

    signal4 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    resample4 = resample(signal4, 10000, 2001)
    assert(len(resample4) == 3)

#testResample()

def testSplitInFrame():
    def testWithNewSignal(duration):
        sample_freq = 10000 # Frame per second
        ms_per_frame = 100 # nb of millisecond per frame
        n_of_frames = np.int(duration * sample_freq)
        signal = Signal(np.zeros(n_of_frames), sample_freq, ms_per_frame)
        print(signal.framed_signal.shape[1], duration * 1000 / ms_per_frame, n_of_frames)
        assert( signal.framed_signal.shape[0] == np.ceil(duration * 1000 / ms_per_frame))
        assert( signal.framed_signal.shape[1] == (sample_freq * ms_per_frame/1000 ) )

    testWithNewSignal(0.9998)
    testWithNewSignal(0.9999)
    testWithNewSignal(1)
    testWithNewSignal(1.0001)
    testWithNewSignal(1.0002)
    testWithNewSignal(2)

    testSignal = Signal(np.array([1, 2, 3, 4, 5,6]), 2, 1000)
    print("Expected : [[1,2], [3,4], [5,6]] --- Value : {}".format(testSignal.framed_signal))
    testSignal = Signal(np.array([1, 2, 3, 4, 5,6]), 2, 800)
    print("Expected : [[1, 2, 3, 4], [5, 6, 0, 0]] --- Value : {}".format(testSignal.framed_signal))