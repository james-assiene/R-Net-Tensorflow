#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 29 15:35:50 2017

@author: squall
"""

import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import nltk
import numpy as np
import os

class CharacterLevelEmbedding:
    
    def __init__(self, batch_size = 10, max_sequence_length = 1000, max_word_length = 30):
        
        self.GLOVE_DIR = "glove"
        self.EMBEDDING_DIM = 300
        self.hidden_units = 50
        
        self.max_sequence_length = max_sequence_length
        self.batch_size = batch_size
        self.max_word_length = max_word_length
        
        self.create_glove_dictionnary()
        
    def tokenize(self, word):
        
        return list(word)
        
    def create_glove_dictionnary(self):
        self.embeddings_index = {}
        f = open(os.path.join(self.GLOVE_DIR, 'glove.840B.300d-char.txt'))
        for line in f:
            values = line.split()
            char = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            self.embeddings_index[char] = coefs
        f.close()
        
        print('Found %s char vectors.' % len(self.embeddings_index))
        
    def tokens_to_glove_matrix(self, tokens, padded = False):
        
        rows_number = self.max_word_length if padded else len(tokens)
        self.embedding_matrix = np.zeros((rows_number, self.EMBEDDING_DIM))
        for i in range(min(len(tokens), rows_number)):
            char = tokens[i]
            embedding_vector = self.embeddings_index.get(char)
            if embedding_vector is not None:
                # chars not found in embedding index will be all-zeros.
                self.embedding_matrix[i] = embedding_vector
                                     
        return self.embedding_matrix
    
    def create_embedding_matrix(self, word, padded = True):
        return self.tokens_to_glove_matrix(self.tokenize(word), padded=padded)
    
    def char_level_text_embedding(self, text):
        
        words = nltk.word_tokenize(text.lower())
        tensor = np.zeros((self.max_sequence_length, self.max_word_length, self.EMBEDDING_DIM))
        for idx in range(min(len(words), self.max_sequence_length)):
            tensor[idx] = self.create_embedding_matrix(words[idx])
            
        tensor = np.array(tensor)
        
        return tensor
            
    
    