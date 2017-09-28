import pickle


def save_object(obj, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_object(file_name):
    with open(file_name, 'r') as f:
        return pickle.load(f)
