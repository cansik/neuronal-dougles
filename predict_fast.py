import argparse

import numpy as np

from prepare.TextAnalyser import TextAnalyser
from training.FastMLP import FastMLP
from util.MemoryTable import MemoryTable
from util.PickleUtil import load_object

from math import factorial

tokenAnalyser = TextAnalyser()

syllablesTable = MemoryTable()
tokenTable = MemoryTable()
dictionaryTable = MemoryTable()

syllableIndex = dict()


def process_arguments():
    parser = argparse.ArgumentParser(description='Neural network to create new words out of descriptions.')
    parser.add_argument('description', nargs='+', help='Description of the new word situation.')

    return parser.parse_args()


def create_data_model(length):
    tokens = list(tokenTable.items())
    tokenLength = len(tokens)

    # create data structure
    X = np.zeros((length, tokenLength))
    return X


def predict(text, index, nn, X):
    # extract tokens
    tokens = tokenAnalyser.extract(text)
    indexed_tokens = filter(lambda x: x in tokenTable, tokens)
    print('Extracted tokens: %s' % ', '.join(indexed_tokens))

    # map tokens to index
    indices = map(lambda x: tokenTable[x]['id'], indexed_tokens)

    # create one hot encoding
    # set input
    for t in indices:
        X[index, t] = 1

    return nn.predict(X)


def resolve_prediction(Y):
    syllables = list()
    for i in xrange(0, Y.shape[1]):
        confidence = Y[0, i]
        syllable = syllableIndex[i]

        syllables.append((i, syllable, confidence))

    return syllables


def get_top(syllables, max_percentage_difference):
    if len(syllables) == 0:
        return []

    tops = []
    last_confidence = syllables[0][2]
    index = 0

    while index < len(syllables) \
            and (last_confidence - syllables[index][2]) / last_confidence <= max_percentage_difference:
        tops.append(syllables[index])
        index += 1

    return tops


def permutate_words(syllables):
    return permutations(map(lambda x: x[1], syllables))


def permutations(l):
    perms = []
    length = len(l)
    for x in xrange(factorial(length)):
        available = list(l)
        newPermutation = []
        for radix in xrange(length, 0, -1):
            placeValue = factorial(radix - 1)
            index = x / placeValue
            newPermutation.append(available.pop(index))
            x -= index * placeValue
        perms.append(newPermutation)
    return perms


def build_syllable_index():
    for syl, entry in syllablesTable.items():
        syllableIndex.update({entry['id']: syl})


def load_tables():
    dictionaryTable.load('data/dictionary.json')
    tokenTable.load('data/tokens.json')
    syllablesTable.load('data/syllables.json')


def main():
    args = process_arguments()

    nn = FastMLP(layers=[], lazy=True)
    nn.load('data/fast_model.json', 'data/fast_weights.h5')

    load_tables()
    build_syllable_index()

    # create prediction for one model
    X = create_data_model(1)
    Y = predict(args.description[0], 0, nn, X)

    syllables = resolve_prediction(Y)
    syllables.sort(key=lambda x: x[2], reverse=True)

    tops = syllables[:3]  # get_top(syllables, 0.1)

    print('')
    print('Description:')
    print(args.description[0])

    print('')
    print('Syllables: ')
    for s in tops:
        print('%s: %s (%s)' % (s[0], s[1], s[2]))

    print('')
    print('Words:')

    perms = map(lambda x: ''.join(x), permutate_words(tops))
    for i, word in enumerate(perms):
        print('%s: %s' % (i, word))


if __name__ == '__main__':
    main()
