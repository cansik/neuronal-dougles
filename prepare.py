
from tinydb import TinyDB, Query
from nltk.corpus import cmudict
import pandas as pd
import csv

DB_NAME = 'words.json'
DICTIONARY_NAME = 'data/dictionary.csv'

cmu = cmudict.dict()

def analyseDictionary():
	with open(DICTIONARY_NAME) as f:
	    read = csv.reader(f)
	    for row in read:
	    	# filter out any one word and check if there are only alphas
	    	if(len(row[0]) > 3 and row[0].isalpha()):
	       		analyseRow(row)

def analyseRow(row):
	# analyse the row
	word = row[0]
	wordType = row[1]
	description = row[2]
	syllables = map(lambda x: str(x), nsyl(word))

	# guard if no syllables are found		
	if(len(syllables) == 0):
		return

	print('%s => %s' % (word, '-'.join(syllables)))

def nsyl(word):
	try:
		lutes = cmu[word.lower()][0]

		if(len(lutes) != len(word)):
			return []

		indices = [i for i, l in enumerate(lutes) if l[-1].isdigit()]
		return splitAt(word, indices)
  	except KeyError:
  		return []

def splitAt(string, indices):
	parts = []
	counter = 0

	for index in indices:
		i = index + counter + 1
		token, string = (string[:i]).strip(),(string[i:]).strip()
		counter += index
		parts.append(token.lower())

	parts.append(string)
	return filter(None, parts)

def main():
	print('building word index...')

	db = TinyDB(DB_NAME)

	analyseDictionary()

if __name__ == '__main__':
	main()