import numpy as np

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
    return nn.predict_all(X)


def load_tables():
    dictionaryTable.load('data/dictionary.json')
    tokenTable.load('data/tokens.json')
    syllablesTable.load('data/syllables.json')


def main():
    print('loading data...')
    load_tables()

    print('Dictionary Entries: %s\tTokens: %s\tSyllables: %s' % (
        dictionaryTable.size(), tokenTable.size(), syllablesTable.size()))

    # creating datamodel
    X, Y = create_data_model()

    # split model in percentage train: 75% test: 25%
    train_end = int(dictionaryTable.size() * 0.75)
    test_start = train_end + 1

    X_train, Y_train = X[0:train_end], Y[0:train_end]
    X_test, Y_test = X[test_start:], Y[test_start:]

    # print sizes
    print('Train: X %s Y %s' % (X_train.shape, Y_train.shape))
    print('Test: X %s Y %s' % (X_test.shape, Y_test.shape))

    # create neuronal network
    nn = NeuralNetwork([X_train.shape[1], 100, 100, Y_train.shape[1]], epochs=5, learning_rate=0.2)

    print('train neural network...')
    train(nn, X_train, Y_train)
    save_object(nn, 'data/neural_network.pkl')

    # load pre-learned neural network
    nn = load_object('data/neural_network.pkl')

    print('testing neural network...')
    Y_predicted = test(nn, X_test)
    Y_binarized = nn.binarize_predicted(Y_predicted)

    print('Ones: %s' % np.count_nonzero(Y_binarized))

    accuracy = nn.score(Y_test, Y_binarized)

    print('Accuracy: %s' % accuracy)


if __name__ == '__main__':
    main()
