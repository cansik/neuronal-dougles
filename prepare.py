import argparse
import csv

import sys

from prepare.Hyphenator import Hyphenator
from prepare.SyllableAnalyser import SyllableAnalyser
from prepare.SyllableSlicer import SyllableSlicer
from prepare.TextAnalyser import TextAnalyser
from util.MemoryTable import MemoryTable

thismodule = sys.modules[__name__]
thismodule.syllableAnalyser = SyllableSlicer()

DICTIONARY_NAME = 'data/dictionary.csv'

syllablesTable = MemoryTable()
tokenTable = MemoryTable()
dictionaryTable = MemoryTable()

tokenAnalyser = TextAnalyser()


def analyse_dictionary():
    with open(DICTIONARY_NAME) as f:
        read = csv.reader(f)
        for row in read:
            # filter out any one word and check if there are only alphas
            if len(row[0]) > 3 and row[0].isalpha():
                analyse_row(row)


def analyse_row(row):
    # analyse the row
    word = row[0]
    wordType = row[1]
    description = row[2]

    tokens = tokenAnalyser.extract(description)
    syllables = map(lambda x: x.lower(), thismodule.syllableAnalyser.slice(word))

    # guard if no syllables are found
    if len(syllables) == 0:
        return

    print('%s: %s (%s)' % (word, '-'.join(syllables), ', '.join(tokens)))

    # write information to tables
    tokenIndexes = map(lambda t: tokenTable.insert(t), tokens)
    syllableIndexes = map(lambda s: syllablesTable.insert(s), syllables)

    dictionaryTable.insert(word, {
        'wordType': wordType,
        'description': description,
        'tokens': tokenIndexes,
        'syllables': syllableIndexes
    })


def process_arguments():
    parser = argparse.ArgumentParser(description='Prepare dictionary to be trained into neural network.')
    parser.add_argument('--syllable', default='sonority', choices=['sonority', 'pronouncing', 'hyphenator'],
                        help='Syllable slicing algorithm.')

    return parser.parse_args()


def show_tables():
    for k, v in tokenTable.items():
        print('%s: %s' % (k, v['id']))

    for k, v in syllablesTable.items():
        print('%s: %s' % (k, v['id']))


def save_tables():
    dictionaryTable.save('data/dictionary.json')
    tokenTable.save('data/tokens.json')
    syllablesTable.save('data/syllables.json')


def main():
    args = process_arguments()

    if args.syllable == 'pronouncing':
        thismodule.syllableAnalyser = SyllableAnalyser()

    if args.syllable == 'hyphenator':
        print('using hyphenator')
        thismodule.syllableAnalyser = Hyphenator()

    print('building word index...')

    analyse_dictionary()

    # showTables()

    save_tables()

    print("Dictionary Entries: %s\tTokens: %s\tSyllables: %s" % (
        dictionaryTable.size(), tokenTable.size(), syllablesTable.size()))


if __name__ == '__main__':
    main()
