#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  5 13:32:56 2017

@author: squall
"""

import os
import numpy as np
import nltk

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

class Embeddings:
    
    def __init__(self, max_nb_words = 20000, max_sequence_length = 1000):
        
        self.GLOVE_DIR = "glove"
        self.EMBEDDING_DIM = 100
        
        self.max_number_of_words = max_nb_words
        self.max_sequence_length = max_sequence_length
        
        self.create_glove_dictionnary()
        
    def tokenize(self, text):
        
        return nltk.word_tokenize(text.lower())
    
    def create_glove_dictionnary(self):
        self.embeddings_index = {}
        f = open(os.path.join(self.GLOVE_DIR, 'glove.6B.100d.txt'), encoding="utf8")
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            self.embeddings_index[word] = coefs
        f.close()
        
        print('Found %s word vectors.' % len(self.embeddings_index))
    
    def tokens_to_glove_matrix(self, tokens, padded = False):
        
        rows_number = self.max_sequence_length if padded else len(tokens)
        self.embedding_matrix = np.zeros((rows_number, self.EMBEDDING_DIM))
        for i in range(min(len(tokens), rows_number)):
            word = tokens[i]
            embedding_vector = self.embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                self.embedding_matrix[i] = embedding_vector
                                     
        return self.embedding_matrix
    
    def create_embedding_matrix(self, text, padded = True):
        return self.tokens_to_glove_matrix(self.tokenize(text), padded)