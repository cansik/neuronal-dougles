from nltk import WordNetLemmatizer
from nltk.stem.lancaster import LancasterStemmer
from rake_nltk import Rake


class TextAnalyser(object):
    def __init__(self):
        self.threshold = 0.95
        self.__rake = Rake()
        self.__stemmer = LancasterStemmer()
        self.__lemma = WordNetLemmatizer()
        self.__stopwords = ['alt']
        pass

    def extract(self, text):
        self.__rake.extract_keywords_from_text(text)
        scores = self.__rake.get_ranked_phrases_with_scores()
        keywords = self.unpack_keywords(scores)
        words = filter(lambda x: x[1] not in self.__stopwords and x[1].isalnum(), keywords)

        filtered_words = map(lambda x: x[1], filter(lambda x: x[0] > self.threshold, words))

        lemms = map(lambda x: self.__lemma.lemmatize(x), filtered_words)
        stems = map(lambda x: self.__stemmer.stem(x), lemms)

        return stems

    @staticmethod
    def unpack_keywords(keywords):
        words = []

        for k in keywords:
            for p in k[1].split(' '):
                words.append((k[0], p))

        return words
