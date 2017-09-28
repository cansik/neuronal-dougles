from tinydb import TinyDB, Query

import csv
from prepare.SyllableAnalyser import SyllableAnalyser
from prepare.MemoryTable import MemoryTable
from prepare.TextAnalyser import TextAnalyser

DB_NAME = 'words.json'
DICTIONARY_NAME = 'data/dictionary.csv'

syllablesTable = MemoryTable()
tokenTable = MemoryTable()

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

    tokens = tokenAnalyser.extract(description)
    syllables = syllableAnalyser.nsyl(word)

    # guard if no syllables are found
    if len(syllables) == 0:
        return

    # write information to tables
    for t in tokens:
        tokenTable.insert(t)

    for s in syllables:
        syllablesTable.insert(s)

    print('analysing %s' % word)


def showTables():
    for k, v in tokenTable.items():
        print('%s: %s' % (k, v['id']))

    for k, v in syllablesTable.items():
        print('%s: %s' % (k, v['id']))


def main():
    print('building word index...')

    db = TinyDB(DB_NAME)
    db.purge()

    analyseDictionary()

    # showTables()

    print("Tokens: %s\tSyllables: %s" % (tokenTable.size(), syllablesTable.size()))


if __name__ == '__main__':
    main()
