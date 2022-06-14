import re
import os
import numpy as np
import pickle
from scipy.spatial import distance

def print_overview_instructions():
	"""
	Prints instructions of script for user. 
	"""
	print("------------------------\n")
	print("This script allows you to query the word2vec model")
	print("using algebraic expressions.  An example of such a")
	print("query string is \n\n           'king + man - woman'\n\n")
	print("Only letters, +,- and spaces are allowed.")

def get_words_and_ops(query):
	"""
	Creates a dictionary with the words and their associated algebraic 
	operations. 
	"""
	letters = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
	words_and_ops = {}
	w = ""
	op = "start"
	for i in range(len(query)):
		char = query[i]
		if char in letters: 
			w += char
		else: 
			words_and_ops[w] = op
			op = char
			w = ""
		if i == (len(query)-1):
			words_and_ops[w] = op
	return words_and_ops

def get_query():
	"""
	Inputs the word query from user 
	"""
	print("\n------------------------")
	query = input("Enter your query below:\n")
	query = query.lower().replace(" ","")	
	return query

def check_for_new_query():
	"""
	Inputs whether user wants to continue querying or end script.
	"""
	is_new_query = input('\n\n\nWould you like to ask another query? [y]/[n]\n')
	while is_new_query not in ["y","n"]:
		is_new_query = input('\n\n\nInvalid reply. Please reply with [y] or [n]\n')
	return is_new_query

def print_closest_words(words_and_ops):
	"""
	Returns the closest words via word2vec. 
	"""
	for word,op in words_and_ops.items():
		if op == "start":
			W_test = W[vocab_to_idx[word]].reshape((1,k))
		elif op == "+": 
			W_test += W[vocab_to_idx[word]]
		else: 
			W_test -= W[vocab_to_idx[word]]

	dist = distance.cdist(W_test,W, "cosine")
	closest_idx = dist.argsort()[-3:][::-1][0,0:5]
	closest = [idx_to_vocab[i] for i in closest_idx]
	print("\n\n\nResults: \n\n")
	print(closest)

def check_valid_input(query):
	"""
	Verify query is only a string of letters, spaces, +, - 
	"""
	query_chars = set(query)
	letters = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
	whitelist = letters | set('+- ')
	assert query_chars <= whitelist, "Only letters, spaces, + and - allowed"	# convert to lower case and eliminate spaces 

def check_valid_words(words_and_ops):
	"""
	Verifies words are in vocabulary
	"""
	for word in words_and_ops.keys():
		assert word in vocabulary, f"{word} not in vocabulary. All words must be in vocabulary."


# load matrices and parameters from training 
folder_path = "./model_params/"
os.chdir(folder_path)
W = np.load("W.npy")
with open('vocab_to_idx.pickle', 'rb') as handle:
	vocab_to_idx = pickle.load(handle)
with open('idx_to_vocab.pickle', 'rb') as handle:
	idx_to_vocab = pickle.load(handle)
with open('vocabulary.pickle', 'rb') as handle:
	vocabulary = pickle.load(handle)
N_sp, k = W.shape


# script for user querying for closest words to user algebraic input
user_done = False
print_overview_instructions()
while user_done == False:
	query = get_query()
	check_valid_input(query)
	words_and_ops = get_words_and_ops(query)
	check_valid_words(words_and_ops)
	print_closest_words(words_and_ops)
	is_new_query = check_for_new_query()
	user_done = False if is_new_query == "y" else True