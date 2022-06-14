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
	print("to find the closes words.     An example of such a")
	print("query string is fish'.       Only letters, +,- and")
	print("spaces are allowed.")

def get_word():
	"""
	Inputs the word query from user 
	"""
	# asks for the query string 
	print("\n------------------------")
	word = input("Enter your query below:\n")
	word = word.lower().replace(" ","")	
	return word

def check_for_new_query():
	"""
	Inputs whether user wants to continue querying or end script.
	"""
	is_new_query = input('\n\n\nWould you like to ask another query? [y]/[n]\n')
	while is_new_query not in ["y","n"]:
		is_new_query = input('\n\n\nInvalid reply. Please reply with [y] or [n]\n')
	return is_new_query

def print_closest_words(word):
	"""
	Returns the closest words via word2vec. 
	"""
	target_idx = vocab_to_idx[word]
	dist = distance.cdist(W[[target_idx],:],np.delete(W,target_idx , axis=0),
	"cosine")
	closest_idx = dist.argsort()[-3:][::-1][0,0:5]
	closest = [idx_to_vocab[i] for i in closest_idx]
	print("\n\n\nResults: \n\n")
	print(closest)

def check_valid_input(word):
	"""
	Validates whether word is only 
	"""
	# verify query a string of letters
	word_chars = set(word)
	letters = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
	assert word_chars <= letters, "Only letters allowed"
	# convert to lower case and eliminate spaces 
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


# script for user querying for closest words to user input 
user_done = False
print_overview_instructions()
while user_done == False:
	word = get_word()
	check_valid_input(word)
	print_closest_words(word)
	is_new_query = check_for_new_query()
	user_done = False if is_new_query == "y" else True