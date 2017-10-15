from os import listdir
from os.path import isfile, join
import numpy as np
import math
import time
from FileToObject import getSignalFromFilename
DIR_TO_DATA = "data/"
class Corpus:
    def __init__(self, dirname=None, annotation=None, limit=math.inf):
        if (dirname == None):
            self.signals = np.array([])
            self.frames = np.ndarray((0, 39))
            return
        print("Importing corpus from directory: {}".format(dirname))
        self.signals = np.array([getSignalFromFilename(join(dirname,filename))
                        for index, filename in enumerate(listdir(dirname))
                        if (limit > index) and isWavFile(dirname, filename) ])
        self.frames = np.concatenate([signal.feature_vectors for signal in self.signals])

    def mergeWith(self, corpusList):
        corpusList.append(self)
        for corpus in corpusList:
            print(corpus)
        #print([(corpus.signals.shape, corpus.frames.shape) for corpus in corpusList])
        self.signals = np.concatenate( [corpus.signals for corpus in corpusList])
        self.frames = np.concatenate( [corpus.frames for corpus in corpusList])
        return self

    def getTotalDuration(self):
        return np.sum([signal.getDuration() for signal in self.signals])


def importAllCorpus(limit=math.inf):
    speech_corpus_list = ["data_de_morgane", "fables", "laurent_words", "sherman_words"]
    speech_dirs = [ join(DIR_TO_DATA, "speech", corpus_name) for corpus_name in speech_corpus_list]
    non_speech_corpus_list = ["ESC-50", "sound1_dataset"]
    non_speech_dirs = [ join(DIR_TO_DATA, "non-speech", corpus_name) for corpus_name in non_speech_corpus_list]

    speech_corpus = Corpus().mergeWith([Corpus(speech_dir, limit=limit) for speech_dir in speech_dirs])
    non_speech_corpus = Corpus().mergeWith([Corpus(non_speech_dir, limit=limit) for non_speech_dir in non_speech_dirs])
    return {"speech":speech_corpus, "non-speech":non_speech_corpus}


def testImportAllCorpus():
    tic = time.time()
    print(time.time() - tic)

    corpora = importAllCorpus()
    print(time.time() - tic)
    print(corpora["speech"].getTotalDuration(), corpora["non-speech"].getTotalDuration())

def isWavFile(dirname, filename):
    return isfile(join(dirname, filename)) and (".wav" in filename)


def testMergeCorpus():
    path_to_non_speech = "data/non-speech/sound1_dataset"
    testCorpus1 = Corpus(path_to_non_speech, limit=10)
    len_signal1 = testCorpus1.signals.shape
    len_frames1 = testCorpus1.frames.shape

    testCorpus2 = Corpus(path_to_non_speech, limit=10)

    testCorpus1.mergeWith([testCorpus2])
    print(testCorpus1.signals.shape[0], len_signal1, (len_signal1 +testCorpus2.signals.shape)  )
    print(testCorpus1.frames.shape[0], len_frames1, (len_frames1 +testCorpus2.frames.shape)  )

#testImportAllCorpus()

def testCorpus():
    path_to_non_speech = "data/non-speech/sound1_dataset"
    testCorpus = Corpus(path_to_non_speech)
    assert(testCorpus.frames.shape[0] ==  getTotalNbOfFrames(testCorpus))

def getTotalNbOfFrames(corpus):
    return np.sum(np.array([
            signal.feature_vectors.shape[0]
            for signal in corpus.signals]))


