import RAKE
from nltk.stem.lancaster import LancasterStemmer


class TextAnalyser(object):
    def __init__(self):
        self.__rake = RAKE.Rake(RAKE.SmartStopList() + ['alt'])
        self.__stemmer = LancasterStemmer()
        pass

    def extract(self, text):
        words = self.__rake.run(text)
        stems = map(lambda x: self.__stemmer.stem(x[0]), words)

        # todo add lemmatization if needed

        return stems
