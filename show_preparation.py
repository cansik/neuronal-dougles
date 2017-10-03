from util.MemoryTable import MemoryTable

syllablesTable = MemoryTable()
tokenTable = MemoryTable()
dictionaryTable = MemoryTable()

syllableIndex = dict()


def build_syllable_index():
    for syl, entry in syllablesTable.items():
        syllableIndex.update({entry['id']: syl})


def resolve_prediction(Y):
    syllables = list()
    for i in xrange(0, Y.shape[1]):
        confidence = Y[0, i]
        syllable = syllableIndex[i]

        syllables.append((i, syllable, confidence))

    return syllables


def load_tables():
    dictionaryTable.load('data/dictionary.json')
    tokenTable.load('data/tokens.json')
    syllablesTable.load('data/syllables.json')


def main():
    print('loading data...')
    load_tables()
    build_syllable_index()

    for syl, value in syllablesTable.items():
        if len(syl) == 1:
            print("%s: %s" % (value['id'], syl))


if __name__ == '__main__':
    main()
