import copy


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

    def items(self):
        return self.__data.iteritems()

    def insert(self, key, value=None):
        if value is None:
            value = dict()

        if key in self.__data:
            return self.__data[key][self.__index_name]

        # update index
        index = copy.copy(self.__index)
        self.__index += 1

        # add index to values
        value.update({self.__index_name: index})

        self.__data.update({key: value})
        return index
