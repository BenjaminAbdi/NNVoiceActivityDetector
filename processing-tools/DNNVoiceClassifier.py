import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout
import numpy as np
from Corpus import importAllCorpus
from Timer import Timer
import matplotlib.pyplot as plt

class DNNVoiceClassifier():
    def __init__(self, token, createNew=False, datasets=None):
        self.scores = []
        self.token = token
        self.datasets = datasets
        if (createNew):
            self.createModel()
        else:
            self.importModel()

    def importModel(self):
        self.model = load_model("{}.model".format(self.token))

    def saveModel(self):
        self.model.save("{}.model".format(self.token))

    def createModel(self):
        if (self.token == "model1"):
            self.model = Model1()
        if (self.token == "model2"):
            self.model = Model2()
        if (self.token == "model3"):
            self.model = Model3()
        if (self.token == "model4"):
            self.model = Model4()

    def train(self, n_iteration, batch_size):
        self.model.fit(self.datasets["train"]["frames"], self.datasets["train"]["is_voice"],
          epochs=n_iteration,
          batch_size=batch_size, verbose=0)
        score = self.model.evaluate(self.datasets["test"]["frames"], self.datasets["test"]["is_voice"], batch_size=128)
        self.scores.append(score[1])

def Model1():
    model = Sequential()
    model.add(Dense(200, input_dim=39, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(60, activation="relu"))
    model.add(Dense(20, activation="relu"))
    model.add(Dense(8, activation="relu"))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
            optimizer='rmsprop',
            metrics=['accuracy'])
    return model

def Model2():
    model = Sequential()
    model.add(Dense(200, input_dim=39, activation="relu"))
    model.add(Dense(60, activation="relu"))
    model.add(Dense(20, activation="relu"))
    model.add(Dense(8, activation="relu"))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
            optimizer='rmsprop',
            metrics=['accuracy'])
    return model

def Model3():
    model = Sequential()
    model.add(Dense(300, input_dim=39, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(80, activation="relu"))
    model.add(Dense(40, activation="relu"))
    model.add(Dense(12, activation="relu"))
    model.add(Dense(3, activation="relu"))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
            optimizer='rmsprop',
            metrics=['accuracy'])
    return model

def Model4():
    model = Sequential()
    model.add(Dense(60, input_dim=39, activation="relu"))
    model.add(Dense(20, activation="relu"))
    model.add(Dense(8, activation="relu"))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
            optimizer='rmsprop',
            metrics=['accuracy'])
    return model

def getShuffledArrays(corpus):
    non_shuffled_speech = [ (frame, 1) for frame in corpus["speech"].frames]
    non_shuffled_non_speech = [ (frame, 0) for frame in corpus["non-speech"].frames]
    datas = np.concatenate((non_shuffled_non_speech, non_shuffled_speech))
    np.random.shuffle(datas)
    return np.array([data[0] for data in datas]), np.array([data[1] for data in datas])

def corpusToDatasets(corpus):
    train_test_ratio = 0.9
    frames, labels = getShuffledArrays(corpus)
    data_length = frames.shape[0]
    train_size = np.int(np.floor(train_test_ratio * data_length))
    test_size = np.int(np.floor((1 - train_test_ratio) * data_length))

    train_set = {
        "is_voice": labels[:train_size],
        "frames": frames[:train_size]
    }
    test_set = {
        "is_voice": labels[-test_size:],
        "frames": frames[-test_size:]
    }

    return {"train": train_set, "test": test_set}

def createModelOnCorpus():
    corpus = importAllCorpus()
    datasets = corpusToDatasets(corpus)

    VAD1 = DNNVoiceClassifier("model1", datasets=datasets, createNew=True)
    VAD2 = DNNVoiceClassifier("model2", datasets=datasets, createNew=True)
    VAD3 = DNNVoiceClassifier("model3", datasets=datasets, createNew=True)
    VAD4 = DNNVoiceClassifier("model4", datasets=datasets, createNew=True)
    timer = Timer(["1", "2", "3", "4"])

    for n_train in range(0, 40):
        print("!!! Train VAD1 n°{} with Batch size 10000 !!!".format(n_train))
        VAD1.train(n_iteration=6, batch_size=10000)
        timer.storeTime("1")
        print("!!! Train VAD2 n°{} with Batch size 10000 !!!".format(n_train))
        VAD2.train(n_iteration=10, batch_size=10000)
        timer.storeTime("2")
        print("!!! Train VAD3 n°{} with Batch size 10000 !!!".format(n_train))
        VAD3.train(n_iteration=4, batch_size=10000)
        timer.storeTime("3")
        print("!!! Train VAD4 n°{} with Batch size 10000 !!!".format(n_train))
        VAD4.train(n_iteration=22, batch_size=10000)
        timer.storeTime("4")


    print("Results are:")
    plt1, = plt.plot(VAD1.scores, label="VAD1")
    plt2, = plt.plot(VAD2.scores, label="VAD2")
    plt3, = plt.plot(VAD3.scores, label="VAD3")
    plt4, = plt.plot(VAD4.scores, label="VAD4")
    plt.legend([plt1, plt2, plt3, plt4], ['VAD1', 'VAD2', 'VAD3', 'VAD4'])
    plt.show()
    print("VAD1: scores are {}".format(VAD1.scores))
    print("VAD2: scores are {}".format(VAD2.scores))
    print("VAD3: scores are {}".format(VAD3.scores))
    print("VAD4: scores are {}".format(VAD4.scores))
    timer.showResults()
    VAD1.saveModel()
    VAD2.saveModel()
    VAD3.saveModel()
    VAD4.saveModel()


createModelOnCorpus()