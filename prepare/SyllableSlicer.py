import sylli


class SyllableSlicer(object):
    def __init__(self):
        self.syl = sylli.SylModule()
        self.syl.load_conf('data/sonority.txt')

    def slice(self, word):
        syllables = []

        sylls = self.syl.syllabify(word).split('.')
        counter = 0

        for s in sylls:

            syllables.append(word[counter:counter+len(s)])
            counter += len(s)

        return syllables
