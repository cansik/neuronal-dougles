from nltk.corpus import cmudict


class SyllableAnalyser(object):
    def __init__(self):
        self.__cmu = cmudict.dict()
        pass

    @staticmethod
    def split_at(string, indices):
        parts = []
        counter = 0

        for index in indices:
            i = index + counter + 1
            token, string = (string[:i]).strip(), (string[i:]).strip()
            counter += index
            parts.append(token.lower())

        parts.append(string)
        return filter(None, parts)

    def split_to_syllable(self, word):
        try:
            lutes = self.__cmu[word.lower()][0]

            if len(lutes) != len(word):
                return []

            indices = [i for i, l in enumerate(lutes) if l[-1].isdigit()]
            return self.split_at(word, indices)  # map(lambda x: str(x), )
        except KeyError:
            return []
