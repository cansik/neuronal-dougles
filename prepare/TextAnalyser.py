from rake_nltk import Rake
from nltk.stem.lancaster import LancasterStemmer


class TextAnalyser(object):
    def __init__(self):
        self.threshold = 0.9
        self.__rake = Rake()
        self.__stemmer = LancasterStemmer()
        self.__stopwords = ['alt']
        pass

    def extract(self, text):
        self.__rake.extract_keywords_from_text(text)
        scores = self.__rake.get_ranked_phrases_with_scores()
        keywords = self.unpack_keywords(scores)
        words = filter(lambda x: x[1] not in self.__stopwords, keywords)
        stems = map(lambda x: self.__stemmer.stem(x[1]), filter(lambda x: x[0] > self.threshold, words))

        # todo add lemmatization if needed

        return stems

    @staticmethod
    def unpack_keywords(keywords):
        words = []

        for k in keywords:
            for p in k[1].split(' '):
                words.append((k[0], p))

        return words
