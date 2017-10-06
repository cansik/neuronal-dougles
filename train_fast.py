from datetime import datetime

import numpy as np

from training.FastMLP import FastMLP
from training.NeuronalNetwork import NeuralNetwork
from util.MemoryTable import MemoryTable
from util.PickleUtil import save_object, load_object

syllablesTable = MemoryTable()
tokenTable = MemoryTable()
dictionaryTable = MemoryTable()


def create_data_model():
    syllables = list(syllablesTable.items())
    tokens = list(tokenTable.items())
    data = list(dictionaryTable.items())

    syllableLength = len(syllables)
    tokenLength = len(tokens)
    dataLength = len(data)

    # create data structure
    X = np.zeros((dataLength, tokenLength))
    Y = np.zeros((dataLength, syllableLength))

    # create one hot encoding
    for i, (word, entry) in enumerate(data):
        # set input
        for t in entry['tokens']:
            X[i, t] = 1

        # set output
        for s in entry['syllables']:
            Y[i, s] = 1

    return X, Y


def train(nn, X, Y):
    nn.fit(X, Y)


def test(nn, X):
    return nn.predict(X)


def load_tables():
    dictionaryTable.load('data/dictionary.json')
    tokenTable.load('data/tokens.json')
    syllablesTable.load('data/syllables.json')


def log(text):
    with open("log.txt", "a") as myfile:
        myfile.write(text)


def main():
    print('loading data...')
    load_tables()

    print('Dictionary Entries: %s\tTokens: %s\tSyllables: %s' % (
        dictionaryTable.size(), tokenTable.size(), syllablesTable.size()))

    # creating datamodel
    X, Y = create_data_model()

    # split model in percentage
    train_end = int(dictionaryTable.size() * 0.25)
    test_start = train_end + 1
    test_end = test_start + train_end

    X_train, Y_train = X[0:train_end], Y[0:train_end]
    X_test, Y_test = X[test_start:test_end], Y[test_start:test_end]

    # print sizes
    print('Train: X %s Y %s' % (X_train.shape, Y_train.shape))
    print('Test: X %s Y %s' % (X_test.shape, Y_test.shape))

    # params
    hidden_layers = [5000]
    epochs = 1
    learning_rate = 0.02

    # create neuronal network
    nn = FastMLP([X_train.shape[1]] + hidden_layers + [Y_train.shape[1]], epochs=epochs, learning_rate=learning_rate)

    print('train neural network...')
    train(nn, X_train, Y_train)

    nn.save('data/fast_model.json', 'data/fast_weights.h5')

    # load pre-learned neural network
    nn.load('data/fast_model.json', 'data/fast_weights.h5')

    print('testing neural network...')

    accuracy = nn.score(X_test, Y_test)
    log(','.join(map(str,
                     [datetime.now(), X_train.shape[1], Y_train.shape[1], '-'.join(map(str, hidden_layers)), epochs,
                      learning_rate, accuracy])))

    print('Accuracy: %s' % accuracy)


if __name__ == '__main__':
    main()
