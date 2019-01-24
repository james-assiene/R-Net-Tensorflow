#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  5 11:27:41 2017

@author: squall
"""

import json


class SquadDatasetParser:
    
    def __init__(self, file_path = "datasets/train-v1.1.json"):
        
        self.documents = []
        self.questions = []
        self.answers = []
        
        file = open(file_path)
        file_content= json.load(file)
        self.file_content = file_content["data"]
        
    def get_documents_questions_answers(self):
        
        for data_item in self.file_content:
            paragraphs = data_item["paragraphs"]
            for paragraph in paragraphs:
                current_document_index = len(self.documents)
                self.questions.append([])
                self.answers.append([])
                self.documents.append(paragraph["context"])
                for question_answer_pair in paragraph["qas"]:
                    current_question_in_document_index = len(self.questions[current_document_index])
                    self.questions[current_document_index].append(question_answer_pair["question"])
                    self.answers[current_document_index].append([])
                    for possible_answer in question_answer_pair["answers"]:
                        start_ind = possible_answer["answer_start"]
                        self.answers[current_document_index][current_question_in_document_index].append(
                                {"answer_start": start_ind, "answer_end": start_ind + len(possible_answer["text"].split())})
                    
        return {"documents": self.documents, "questions": self.questions, "answers": self.answers}
    
    def get_all_questions(self):
        
        result = []
        
        for question in self.questions:
            result+= question
            
        return result
