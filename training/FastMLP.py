from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop


class FastMLP(object):
    def __init__(self, layers, learning_rate=0.1, epochs=5, batch_size=128):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

        self.model = Sequential()

        self.model.add(Dense(512, activation='relu', input_shape=(layers[0],)))
        self.model.add(Dropout(0.2))

        for n_units in layers:
            self.model.add(Dense(n_units, activation='relu'))
            self.model.add(Dropout(0.2))

        self.model.add(Dense(layers[-1], activation='softmax'))

        self.model.summary()

        self.model.compile(loss='categorical_crossentropy',
                           optimizer=RMSprop(),
                           metrics=['accuracy'])

    def fit(self, x_train, y_train):
        return self.model.fit(x_train, y_train,
                              batch_size=self.batch_size,
                              epochs=self.epochs,
                              verbose=1)

    def score(self, x_test, y_test):
        score = self.model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        # print('Test accuracy:', score[1])

        return score[1]

    def predict(self, x_test):
        return self.model.predict(x_test, batch_size=self.batch_size)

    def save(self, model_name=None, weights_name=None):
        # serialize model to JSON
        if model_name is not None:
            model_json = self.model.to_json()
            with open(model_name, "w") as json_file:
                json_file.write(model_json)

        # serialize weights to HDF5
        if weights_name is not None:
            self.model.save_weights(weights_name)

    def load(self, model_name=None, weights_name=None):
        # load json and create model
        if model_name is not None:
            json_file = open(model_name, 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            self.model = model_from_json(loaded_model_json)

        # load weights into new model
        if weights_name is not None:
            self.model.load_weights(weights_name)
