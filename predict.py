import argparse

import numpy as np

from prepare.TextAnalyser import TextAnalyser
from util.MemoryTable import MemoryTable
from util.PickleUtil import load_object

tokenAnalyser = TextAnalyser()

syllablesTable = MemoryTable()
tokenTable = MemoryTable()
dictionaryTable = MemoryTable()


def process_arguments():
    parser = argparse.ArgumentParser(description='Neural network to create new words out of descriptions.')
    parser.add_argument('description', nargs='+', help='Description of the new word situation.')

    parser.print_help()

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

    # map tokens to index
    indices = map(lambda x: tokenTable[x]['id'], filter(lambda x: x in tokenTable, tokens))

    # create one hot encoding
    # set input
    for t in indices:
        X[index, t] = 1

    return nn.predict_all(X)


def resolve_prediction(Y):
    # todo: one-hot to right syllables and create word list with permutation
    return []


def load_tables():
    dictionaryTable.load('data/dictionary.json')
    tokenTable.load('data/tokens.json')
    syllablesTable.load('data/syllables.json')


def main():
    args = process_arguments()

    nn = load_object('data/neural_network.pkl')
    load_tables()

    # create prediction for one model
    X = create_data_model(1)
    Y = predict(args.description[0], 0, nn, X)

    syllables = resolve_prediction(Y)


if __name__ == '__main__':
    main()
