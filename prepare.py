import argparse
import csv

import sys

from prepare.Hyphenator import Hyphenator
from prepare.SyllableAnalyser import SyllableAnalyser
from prepare.SyllableSlicer import SyllableSlicer
from prepare.TextAnalyser import TextAnalyser
from util.MemoryTable import MemoryTable
from collections import defaultdict

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

        counter = 0

        for row in read:
            if counter % 1 == 0:
                # filter out any one word and check if there are only alphas
                if len(row[0]) > 3 and row[0].isalpha():
                    analyse_row(row)
            counter += 1


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

    # print('%s: %s (%s)' % (word, '-'.join(syllables), ', '.join(tokens)))

    dictionaryTable.insert(word, {
        'wordType': wordType,
        'description': description,
        'token_list': tokens,
        'syllable_list': syllables
    })


def create_index(store_raw):
    for i, (word, entry) in enumerate(dictionaryTable.items()):
        tokenIndexes = map(lambda t: tokenTable.insert(t), entry['token_list'])
        syllableIndexes = map(lambda s: syllablesTable.insert(s), entry['syllable_list'])

        entry.update({
            'syllables': syllableIndexes,
            'tokens': tokenIndexes
        })

        if not store_raw:
            entry.pop('syllable_list')
            entry.pop('token_list')


def filter_to_relevant(max_t_s, max_s_t):
    # s_t = 1*syllable has to be linked to n*tokens
    syllables = defaultdict(set)
    tokens = defaultdict(set)

    # prepare index
    print('create item index...')
    for i, (word, entry) in enumerate(dictionaryTable.items()):
        # count token to syllable
        for t in entry['token_list']:
            for s in entry['syllable_list']:
                tokens[t].add(s)

        # set output
        for s in entry['syllable_list']:
            for t in entry['token_list']:
                syllables[s].add(t)

    # analyse
    print('analyse items...')
    removable_tokens = set()
    removable_syllables = set()

    for t, s in tokens.items():
        l = len(s)
        if l > max_t_s:
            removable_tokens.add(t)

    for s, t in syllables.items():
        l = len(t)
        if l > max_s_t:
            removable_syllables.add(s)

    print("Removable Tokens: %s\tSyllables: %s" % (len(removable_tokens), len(removable_syllables)))

    # remove not relevant items
    print('remove items...')
    removable_words = []
    for i, (word, entry) in enumerate(dictionaryTable.items()):
        for t in entry['token_list']:
            if t in removable_tokens:
                entry['token_list'].remove(t)

        for s in entry['syllable_list']:
            if s in removable_syllables:
                entry['syllable_list'].remove(s)

        if len(entry['syllable_list']) == 0 or len(entry['token_list']) == 0:
            removable_words.append(word)

    for word in removable_words:
        dictionaryTable.remove(word)


def process_arguments():
    parser = argparse.ArgumentParser(description='Prepare dictionary to be trained into neural network.')
    parser.add_argument('--syllable', default='sonority', choices=['sonority', 'pronouncing', 'hyphenator'],
                        help='Syllable slicing algorithm.')
    parser.add_argument("-f", "--filter", action="store_true", help="Only use relevant data for preparation.")
    parser.add_argument("-d", "--debug", action="store_true", help="Store raw data to indexed data for debugging.")

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

    if args.filter:
        print('filtering by relation limit...')
        filter_to_relevant(2, 2)

    print('creating index...')
    create_index(store_raw=args.debug)

    # showTables()

    save_tables()

    print("Dictionary Entries: %s\tTokens: %s\tSyllables: %s" % (
        dictionaryTable.size(), tokenTable.size(), syllablesTable.size()))


if __name__ == '__main__':
    main()
