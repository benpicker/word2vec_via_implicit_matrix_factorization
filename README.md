# word2vec_via_implicit_matrix_factorization

This project implements a matrix factorization approach to the popular word embedding algorithm word2vec. Key points to note: 

There are three scripts: 
* `train.py` -- for training the word embeddings
* `closest_words_query.py` -- script that allows user to input a word and find the top 5 closest words in word embedding space. 
* `algebra_query.py` -- script that allows user to enter in word algebra query (e.g. `king + woman - man`) and get back the closest vectors in embedding sapce 

All scripts are designed to be run from the command line. The details of the project can be found in the file `project_summary.pdf` and the user is strongly encouraged to read it. 


The algorithm is run on the Wikipedia data sets from Hugging Face.

## Results 



<p align="center">
<img src="https://github.com/benpicker/word2vec_via_implicit_matrix_factorization/blob/main/markdown_files/closest_results.png" alt="Trulli" style="width:100%">
</p>

<p align="center">
<img src="https://github.com/benpicker/word2vec_via_implicit_matrix_factorization/blob/main/markdown_files/algebra_results.png" alt="Trulli" style="width:100%">
</p>