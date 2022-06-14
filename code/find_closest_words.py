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
print("to find the closes words.     An example of such a")
print("query string is fish'.       Only letters, +,- and")
print("spaces are allowed.")
while done == False:
	# asks for the query string 
	print("\n------------------------")
	word = input("Enter your query below:\n")
	
	# verify query a string of letters, spaces, +, - 
	word_chars = set(word)
	letters = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
	whitelist = letters | set('+- ')
	assert word_chars <= whitelist, "Only letters, spaces, + and - allowed"


	# convert to lower case and eliminate spaces 
	word = word.lower().replace(" ","")
	
	assert word in vocabulary, f"{word} not in vocabulary. All words must be in vocabulary."


	target_idx = vocab_to_idx[word]
	dist = distance.cdist(W[[target_idx],:],np.delete(W,target_idx , axis=0),
	"cosine")
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

