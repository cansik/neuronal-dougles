import csv

from prepare.SyllableAnalyser import SyllableAnalyser
from prepare.TextAnalyser import TextAnalyser
from util.MemoryTable import MemoryTable

DICTIONARY_NAME = 'data/dictionary.csv'

syllablesTable = MemoryTable()
tokenTable = MemoryTable()
dictionaryTable = MemoryTable()

syllableAnalyser = SyllableAnalyser()
tokenAnalyser = TextAnalyser()


def analyseDictionary():
    with open(DICTIONARY_NAME) as f:
        read = csv.reader(f)
        for row in read:
            # filter out any one word and check if there are only alphas
            if len(row[0]) > 3 and row[0].isalpha():
                analyseRow(row)


def analyseRow(row):
    # analyse the row
    word = row[0]
    wordType = row[1]
    description = row[2]

    print('analysing %s' % word)

    tokens = tokenAnalyser.extract(description)
    syllables = syllableAnalyser.nsyl(word)

    # guard if no syllables are found
    if len(syllables) == 0:
        return

    # write information to tables
    tokenIndexes = map(lambda t: tokenTable.insert(t), tokens)
    syllableIndexes = map(lambda s: syllablesTable.insert(s), syllables)

    dictionaryTable.insert(word, {
        'wordType': wordType,
        'description': description,
        'tokens': tokenIndexes,
        'syllables': syllableIndexes
    })


def showTables():
    for k, v in tokenTable.items():
        print('%s: %s' % (k, v['id']))

    for k, v in syllablesTable.items():
        print('%s: %s' % (k, v['id']))


def saveTables():
    dictionaryTable.save('data/dictionary.json')
    tokenTable.save('data/tokens.json')
    syllablesTable.save('data/syllables.json')


def main():
    print('building word index...')

    analyseDictionary()

    # showTables()

    saveTables()

    print("Dictionary Entries: %s\tTokens: %s\tSyllables: %s" % (
        dictionaryTable.size(), tokenTable.size(), syllablesTable.size()))


if __name__ == '__main__':
    main()
