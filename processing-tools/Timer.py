import time as time
import numpy as np

class Timer():
    def __init__(self, listOfTokens):
        self.tracked_times = dict((token, 0) for token in listOfTokens)
        self.initial_time = time.time()
        self.tmp = time.time()

    def storeTime(self, token):
        if (token not in self.tracked_times):
            print("You provided the token '{}' which you did not declare when initializing the timer. Not taken into account.".format(token))
            self.tmp = time.time()
            return
        self.tracked_times[token] += time.time() - self.tmp
        self.tmp = time.time()

    def showResults(self):
        self.tracked_times["total_time"] = time.time() - self.initial_time
        for token in self.tracked_times:
            print("Step {} took {} seconds.".format(token, np.around(self.tracked_times[token], 3)))