import json


class MemoryTable(object):
    def __init__(self):
        self.__index_name = 'id'
        self.__index = 0
        self.__data = dict()
        pass

    def __setitem__(self, key, value):
        self.__data[key] = value

    def __getitem__(self, key):
        return self.__data[key]

    def __contains__(self, item):
        return item in self.__data

    def items(self):
        return self.__data.iteritems()

    def size(self):
        return self.__index

    def insert(self, key, value=None):
        if value is None:
            value = dict()

        if key in self.__data:
            return self.__data[key][self.__index_name]

        # update index
        index = self.__index
        self.__index += 1

        # add index to values
        value.update({self.__index_name: index})

        self.__data.update({key: value})
        return index

    def save(self, file_name):
        with open(file_name, 'w') as fp:
            json.dump(self.__data, fp, sort_keys=True, indent=4, separators=(',', ': '))

    def load(self, file_name):
        with open(file_name, 'r') as fp:
            self.__data = json.load(fp)

        self.__index = len(list(self.items()))
