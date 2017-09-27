
from tinydb import TinyDB, Query

DB_NAME = 'words.json'
DICTIONARY_NAME = 'data/dictionary.csv'

def analyseDictionary():
	with open(DICTIONARY_NAME) as f:
	    for line in f:
	       print line

def main():
	print('building word index...')

	db = TinyDB(DB_NAME)

	analyseDictionary()

if __name__ == '__main__':
	main()