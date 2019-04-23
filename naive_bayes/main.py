import io
import re
from collections import defaultdict
from math import log

class NaiveBayes:
    def __init__(self, storage):
        self.__punctuation = ' | , | . | ; | ! | ? '
        self.__articles = [' a ', 'an', ' the ', ' to ', ' in ', ' at ', ' on ', 'up', 'upon', 'under', 'above']
        self.__classes, self.__words = self.__fit(storage)


    def __fit(self, storage):
        classes, freq = defaultdict(lambda: 0), defaultdict(lambda: 0)
        for label, documents in storage:
            classes[label] += 1
            actual_words = self.__exclude(documents)
            for word in actual_words:
                freq[label, word] += 1

        for label, word in freq:
            freq[label, word] /= classes[label]
        for c in classes:
            classes[c] /= len(storage)

        return classes, freq

    def predict(self, words: list):
        prob = self.__words
        classes = self.__classes
        actual_words = self.__exclude(words)
        return min(classes.keys(),
            key=lambda cl: -log(classes[cl]) + \
                sum(-log(prob.get((cl, word), 10 ** (-7))) for word in actual_words))

    def __exclude(self, words):
        actual_words = [d for d in re.split(self.__punctuation, words) if d is not None]
        return [w for w in actual_words if w not in self.__articles]

storage = [line.split(' /// ') for line in io.open('storage.txt', mode='r', encoding='utf-8')]
bayes = NaiveBayes(storage)

texts = [line.split(' /// ')[1] for line in io.open('input.txt', mode='r', encoding='utf-8')]

for text in texts:
    tag = bayes.predict(text)
    print('[' + tag + '] ' + text)

