from training.NeuronalNetwork import NeuralNetwork
from util.MemoryTable import MemoryTable

syllablesTable = MemoryTable()
tokenTable = MemoryTable()
dictionaryTable = MemoryTable()

def prepare_train_set():
    pass

def prepare_test_set():
    pass

def train(nn, X, Y):
    pass

def test(nn, X):
    pass

def loadTables():
    dictionaryTable.load('data/dictionary.json')
    tokenTable.load('data/tokens.json')
    syllablesTable.load('data/syllables.json')


def main():
    print('loading data...')
    loadTables()

    print('Dictionary Entries: %s\tTokens: %s\tSyllables: %s' % (
        dictionaryTable.size(), tokenTable.size(), syllablesTable.size()))

    X_train, Y_train = prepare_train_set()
    X_test, Y_test = prepare_test_set()

    # create neuronal network
    nn = NeuralNetwork([X_train.shape[1], 50, Y_train.shape[1]])

    print('train neuronal network...')
    train(nn, X_train, Y_train)

    print('testing neuronal network...')
    Y_predicted = test(nn, X_test)

    accuracy = nn.score(Y_test, Y_predicted)

    print('Accuracy: %s' % accuracy)



if __name__ == '__main__':
    main()
