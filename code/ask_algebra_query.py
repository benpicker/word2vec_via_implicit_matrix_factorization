import re
import os
import numpy as np
import pickle
from scipy.spatial import distance

folder_path = "./model_params/"
os.chdir(folder_path)
W = np.load("W.npy")
with open('vocab_to_idx.pickle', 'rb') as handle:
	vocab_to_idx = pickle.load(handle)
with open('vocab_idx_reverse.pickle', 'rb') as handle:
	vocab_idx_reverse = pickle.load(handle)
with open('vocabulary.pickle', 'rb') as handle:
	vocabulary = pickle.load(handle)

N_sp, k = W.shape

done = False
# instruction message 
print("------------------------\n")
print("This script allows you to query the word2vec model")
print("using algebraic expressions.  An example of such a")
print("query string is \n\n           'king + man - woman'\n\n")
print("Only letters, +,- and spaces are allowed.")
while done == False:
	# asks for the query string 
	print("\n------------------------")
	str = input("Enter your query below:\n")
	
	# verify query a string of letters, spaces, +, - 
	str_chars = set(str)
	letters = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
	whitelist = letters | set('+- ')
	assert str_chars <= whitelist, "Only letters, spaces, + and - allowed"


	# convert to lower case and eliminate spaces 
	str = str.lower().replace(" ","")

	# splits string into operations  
	query_words = {}
	w = ""
	op = "start"
	for i in range(len(str)):
		char = str[i]
		if char in letters: 
			w += char
		else: 
			query_words[w] = op
			op = char
			w = ""
		if i == (len(str)-1):
			query_words[w] = op

	for word in query_words.keys():
		assert word in vocabulary, f"{word} not in vocabulary. All words must be in vocabulary."

	for word,op in query_words.items():
		if op == "start":
			W_test = W[vocab_to_idx[word]].reshape((1,k))
		elif op == "+": 
			W_test += W[vocab_to_idx[word]]
		else: 
			W_test -= W[vocab_to_idx[word]]

	dist = distance.cdist(W_test,W, "cosine")
	closest_idx = dist.argsort()[-3:][::-1][0,0:5]
	closest = [vocab_idx_reverse[i] for i in closest_idx]

	print("\n\n\nResults: \n\n")


	print(closest)

	answer = input('\n\n\nWould you like to ask another query? [y]/[n]\n')
	while answer not in ["y","n"]:
		answer = input('\n\n\nInvalid reply. Please reply with [y] or [n]\n')

	if answer == "y":
		done = False
	else:
		done = True

