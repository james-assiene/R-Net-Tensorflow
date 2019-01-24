#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 14:37:45 2017

@author: squall
"""

import tensorflow as tf
from tensorflow.contrib import rnn

class QuestionAndPassageEncoder:
    
    def __init__(self):
        print("encoder")
        self.hidden_units = 75
        self.dropout_keep_prob = 0.8
        
    def encode_question(self, E, pre_C):
        
        with tf.variable_scope("question_encoder", reuse=None) as varscope:
            
            dimensions = pre_C.get_shape().as_list()
            Cs = []
            
            for batch in range(dimensions[0]):
                
                with tf.variable_scope("char_encoder" + str(batch), reuse=None) as char_encode_scope:
            
                    gru_cell_fw = rnn.GRUCell(self.hidden_units)
                    gru_cell_bw = rnn.GRUCell(self.hidden_units)
                    sequence_length = [dimensions[2] for i in range(dimensions[1])]
                    
                    _, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(gru_cell_fw, gru_cell_bw, pre_C[batch,:,:,:], dtype=tf.float64, sequence_length=sequence_length)
                    
                    Cs.append(tf.concat([state_fw, state_bw], axis=1))
                
            C = tf.stack(Cs)
            
            dimensions = E.get_shape().as_list()
        
            gru_cell_fw_l1 = rnn.GRUCell(self.hidden_units)
            gru_cell_bw_l1 = rnn.GRUCell(self.hidden_units)
            
            gru_cell_fw_l2 = rnn.GRUCell(self.hidden_units)
            gru_cell_bw_l2 = rnn.GRUCell(self.hidden_units)
            
            gru_cell_fw_l3 = rnn.GRUCell(self.hidden_units)
            gru_cell_bw_l3 = rnn.GRUCell(self.hidden_units)
            
            sequence_length = [dimensions[1] for i in range(dimensions[0])]
            
            with tf.variable_scope("layer_1", reuse=None):
                (output_fw_l1, output_bw_l1), _ = tf.nn.bidirectional_dynamic_rnn(gru_cell_fw_l1, gru_cell_bw_l1, tf.concat([E, C], axis = 2), dtype=tf.float64, sequence_length=sequence_length)
                tf.nn.dropout(output_fw_l1, self.dropout_keep_prob)
                tf.nn.dropout(output_bw_l1, self.dropout_keep_prob)
                
            with tf.variable_scope("layer_2", reuse=None):
                (output_fw_l2, output_bw_l2), _ = tf.nn.bidirectional_dynamic_rnn(gru_cell_fw_l2, gru_cell_bw_l2, tf.concat([output_fw_l1, output_bw_l1], axis = 2), dtype=tf.float64, sequence_length=sequence_length)
                tf.nn.dropout(output_fw_l2, self.dropout_keep_prob)
                tf.nn.dropout(output_bw_l2, self.dropout_keep_prob)
            
            with tf.variable_scope("layer_3", reuse=None):
                (output_fw_l3, output_bw_l3), _ = tf.nn.bidirectional_dynamic_rnn(gru_cell_fw_l3, gru_cell_bw_l3, tf.concat([output_fw_l2, output_bw_l2], axis = 2), dtype=tf.float64, sequence_length=sequence_length)
                tf.nn.dropout(output_fw_l3, self.dropout_keep_prob)
                tf.nn.dropout(output_bw_l3, self.dropout_keep_prob)
            
            return tf.concat([output_fw_l3, output_bw_l3], axis=2)
        
    def encode_passage(self, E, pre_C):
        
        with tf.variable_scope("passage_encoder", reuse=None) as varscope:
            
            dimensions = pre_C.get_shape().as_list()
            Cs = []
            
            for batch in range(dimensions[0]):
                
                with tf.variable_scope("char_encoder" + str(batch), reuse=None) as char_encode_scope:
            
                    gru_cell_fw = rnn.GRUCell(self.hidden_units)
                    gru_cell_bw = rnn.GRUCell(self.hidden_units)
                    sequence_length = [dimensions[2] for i in range(dimensions[1])]
                    
                    _, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(gru_cell_fw, gru_cell_bw, pre_C[batch,:,:,:], dtype=tf.float64, sequence_length=sequence_length)
                    
                    Cs.append(tf.concat([state_fw, state_bw], axis=1))
                
            C = tf.stack(Cs)
            
            dimensions = E.get_shape().as_list()
        
            gru_cell_fw_l1 = rnn.GRUCell(self.hidden_units)
            gru_cell_bw_l1 = rnn.GRUCell(self.hidden_units)
            
            gru_cell_fw_l2 = rnn.GRUCell(self.hidden_units)
            gru_cell_bw_l2 = rnn.GRUCell(self.hidden_units)
            
            gru_cell_fw_l3 = rnn.GRUCell(self.hidden_units)
            gru_cell_bw_l3 = rnn.GRUCell(self.hidden_units)
            
            sequence_length = [dimensions[1] for i in range(dimensions[0])]
            
            with tf.variable_scope("layer_1", reuse=None):
                (output_fw_l1, output_bw_l1), _ = tf.nn.bidirectional_dynamic_rnn(gru_cell_fw_l1, gru_cell_bw_l1, tf.concat([E, C], axis = 2), dtype=tf.float64, sequence_length=sequence_length)
                output_fw_l1 = tf.nn.dropout(output_fw_l1, self.dropout_keep_prob)
                output_bw_l2 = tf.nn.dropout(output_bw_l1, self.dropout_keep_prob)
                
            with tf.variable_scope("layer_2", reuse=None):
                (output_fw_l2, output_bw_l2), _ = tf.nn.bidirectional_dynamic_rnn(gru_cell_fw_l2, gru_cell_bw_l2, tf.concat([output_fw_l1, output_bw_l1], axis = 2), dtype=tf.float64, sequence_length=sequence_length)
                output_fw_l2 = tf.nn.dropout(output_fw_l2, self.dropout_keep_prob)
                output_bw_l2 = tf.nn.dropout(output_bw_l2, self.dropout_keep_prob)
            
            with tf.variable_scope("layer_3", reuse=None):
                (output_fw_l3, output_bw_l3), _ = tf.nn.bidirectional_dynamic_rnn(gru_cell_fw_l3, gru_cell_bw_l3, tf.concat([output_fw_l2, output_bw_l2], axis = 2), dtype=tf.float64, sequence_length=sequence_length)
                output_fw_l3 = tf.nn.dropout(output_fw_l3, self.dropout_keep_prob)
                output_bw_l3 = tf.nn.dropout(output_bw_l3, self.dropout_keep_prob)
            
            return tf.concat([output_fw_l3, output_bw_l3], axis=2)