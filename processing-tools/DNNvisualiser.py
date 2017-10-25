from DNNVoiceClassifier import DNNVoiceClassifier
from Corpus import Corpus
from os.path import join
import numpy as np
import matplotlib.pyplot as plt

DIR_TO_DATA = "data"

def visualizeFalsy():
    currentCorpus = Corpus(join(DIR_TO_DATA, "non-speech", "sound1_dataset"))
    isSpeech = False
    VAD = DNNVoiceClassifier("model3")
    results = []
    for signal in currentCorpus.signals:
        expected_result = np.full((signal.feature_vectors.shape[0], 1), isSpeech)
        result = getNNPredictionResult( signal, expected_result, VAD=VAD)
        results.append(result)
    bad_results = [result for result in results if result["score"] < 0.9]
    print(np.sum([result["score"] for result in results])/len(results))
    for index,result in enumerate(bad_results):
        if index>5:
            break
        print(result["signal"].filename)
        fig=plt.figure()
        signal_data = result["signal"].plot(superimpose=True)
        frame_len = len(result["correctness"])
        signal_plot = fig.add_subplot(111, label="signal")
        correct_plot = fig.add_subplot(111, label="is_voice", ylim=(-1,1.5), frame_on=False)
        correct_plot.plot(np.linspace(0, result["signal"].getDuration(), frame_len), result["is_voice"], 'C2')
        signal_plot.plot(signal_data[0], signal_data[1], 'C1')
    plt.show()


def getNNPredictionResult(signal, expected_result, VAD):
        prediction = VAD.predict(signal.feature_vectors) > 0.5
        result = {
            "signal": signal,
            "is_voice": prediction,
            "correctness": np.equal(prediction, expected_result)
        }
        result["score"] = np.sum(result["correctness"]) / signal.feature_vectors.shape[0]
        return result

visualizeFalsy()