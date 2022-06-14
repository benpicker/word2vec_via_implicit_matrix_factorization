from nltk.probability import FreqDist
from nltk.corpus import words
from nltk.corpus import stopwords
import nltk
from datasets import load_dataset
import re
import multiprocessing
import time
import scipy.sparse as ss
from scipy.spatial import distance
import numpy as np
import pickle
import os


# creates a folder to save the parameters
folder_path = "./model_params/"
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# import data set from hugging face
data = load_dataset("wikipedia", "20220301.simple")
data = data['train']

# obtains vocabulary and word counts 
nltk.download('words')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
whitelist = set('abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ')
text = ""
for i in range(len(data)-1):
    entry = data[i]['text']
    entry = ''.join(filter(whitelist.__contains__, entry))
    entry = entry.lower()
    text += " " + entry
text = text.split(" ")
text = [w for w in text if w not in stop_words and w != '']

freq_dist = FreqDist(word for word in text)
freq_dist_500 = list(filter(lambda x: x[1]>=500, freq_dist.items()))
vocabulary = set([word[0] for word in freq_dist_500])
data = [w for w in text if w in vocabulary]
word_counts = dict(freq_dist_500)


# identifies the context words for each center word within a given radius 
radius = 2 
n = len(data)
window_idx = set([k for k in range(-radius,radius+1) if k != 0])
pair_counts = {}
for i in range(n):
    for j in window_idx:    
        if i+j >=0 and i+j < n:
            if not frozenset({data[i], data[i+j]}) in pair_counts:
                pair_counts[frozenset({data[i], data[i+j]})] = 1
            else:
                pair_counts[frozenset({data[i], data[i+j]})] += 1


# size and dicts to convert back and forth 
vocab_size = len(vocabulary)
vocab_to_idx = {k: v for v,k in enumerate(vocabulary)}
idx_to_vocab = {v: k for v,k in enumerate(vocabulary)}
N_Sp = sum(pair_counts.values())

# builds the PMI matrix using csr 
M = ss.lil_matrix((vocab_size,vocab_size),dtype=int)
for k,N_pair in pair_counts.items():
    k = tuple(k)
    if len(k) == 1:
        i = vocab_to_idx[k[0]]
        j = vocab_to_idx[k[0]]
        N_w_i = word_counts[k[0]]
        N_w_j = word_counts[k[0]]
    else:
        i = vocab_to_idx[k[0]]
        j = vocab_to_idx[k[1]]
        N_w_i = word_counts[k[0]]
        N_w_j = word_counts[k[1]]
    M[i,j] = np.log((N_pair+1) * N_Sp / (N_w_i * N_w_j))
    M[j,i] = np.log((N_pair+1) * N_Sp / (N_w_i * N_w_j))
M = M.tocsr().asfptype()


# SVD outputs 
U, s, V = ss.linalg.svds(M, k=50)
# embeddings 
W = U @ np.sqrt(np.diag(s))


# saves all outputs 
os.chdir(folder_path)
with open('word_counts.pickle', 'wb') as handle:
	pickle.dump(word_counts, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('vocabulary.pickle', 'wb') as handle:
	pickle.dump(vocabulary, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('pair_counts.pickle', 'wb') as handle:
	pickle.dump(pair_counts, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('vocab_to_idx.pickle', 'wb') as handle:
	pickle.dump(vocab_to_idx, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('idx_to_vocab.pickle', 'wb') as handle:
	pickle.dump(idx_to_vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)
ss.save_npz("PMI_matrix.npz", M)
np.save("U.npy", U)
np.save("s.npy", s)
np.save("V.npy", V)
np.save("W.npy", W)