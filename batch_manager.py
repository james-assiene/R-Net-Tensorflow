#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 13:35:59 2017

@author: squall
"""

import numpy as np
from embeddings import Embeddings
from character_level_embedding import CharacterLevelEmbedding
from numpy.random import randint

class BatchManager:
    
    def __init__(self, dataset, batch_size = 10, max_sequence_length = 1000):
        print("")
        max_nb_words = 2000
        self.used_items = []
        self.batch_size = batch_size
        self.embedder = Embeddings(max_nb_words, max_sequence_length)
        self.max_sequence_length = max_sequence_length
        self.char_level_embedder = CharacterLevelEmbedding(batch_size, max_sequence_length, 30)
        self.dataset = dataset
        self.the_indices = []
        self.batch_start_index = 0
        
        for document_index in range(len(self.dataset["answers"])):
            for question_index in range(len(self.dataset["answers"][document_index])):
                for answer_index in range(len(self.dataset["answers"][document_index][question_index])):
                    self.the_indices.append(str(document_index) + " " + str(question_index) + " " + str(answer_index))
                    
        np.random.shuffle(self.the_indices)
        
    def get_qapairs_number(self):
        return len(self.the_indices)
        
    def random_indices(self):
        
        if self.batch_start_index >= len(self.the_indices):
            self.batch_start_index = 0
            np.random.shuffle(self.the_indices)
            
        flag = True
            
        while flag:
            
            indices = self.the_indices[self.batch_start_index].split()
        
            document_index = int(indices[0])
            
            if len(self.dataset["documents"][document_index].split()) > self.max_sequence_length:
                self.batch_start_index+= 1
                
            else:
                flag = False
                
                question_index = int(indices[1])
                answer_index =int(indices[2])
        
        self.batch_start_index+= 1
        
        return (document_index, question_index, answer_index)
    
    def next_batch(self):
        
        indices_code = "jack"
        documents_batch = []
        questions_batch = []
        answers_batch = []
        char_level_documents_batch = []
        char_level_questions_batch = []
        answers_start_position_batch = []
        answers_end_position_batch = []
        
        for i in range(self.batch_size):
            
            document_index, question_index, answer_index = self.random_indices()
            
            documents_batch.append(self.embedder.create_embedding_matrix(self.dataset["documents"][document_index]))
            char_level_documents_batch.append(self.char_level_embedder.char_level_text_embedding(self.dataset["documents"][document_index]))
            questions_batch.append(self.embedder.create_embedding_matrix(self.dataset["questions"][document_index][question_index]))
            char_level_questions_batch.append(self.char_level_embedder.char_level_text_embedding(self.dataset["questions"][document_index][question_index]))
            
            start_index = self.dataset["answers"][document_index][question_index][answer_index]["answer_start"]
            end_index = self.dataset["answers"][document_index][question_index][answer_index]["answer_end"]
            
            answers_start_position_batch.append(start_index)
            answers_end_position_batch.append(end_index)
            
                
                    
        return documents_batch, questions_batch, char_level_documents_batch, char_level_questions_batch, answers_start_position_batch, answers_end_position_batch
