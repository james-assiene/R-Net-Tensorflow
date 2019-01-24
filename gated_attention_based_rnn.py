#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 18:37:46 2017

@author: squall
"""

import tensorflow as tf
from tensorflow.contrib import rnn

class GatedAttentionBasedRNN:
    
    def __init__(self):
        print("gated attention")
        self.hidden_state_units = 150 # 2 * 75, Bi-RNN => 2 * RNN size
        self.cell_size = 75
        self.dropout_keep_prob = 0.8
        
    def get_question_aware_passage_representation(self, U_P, U_Q):
        
        with tf.variable_scope("gated_attention_based_rnn", reuse=None) as varscope:
            
            ###  G ATED A TTENTION - BASED R ECURRENT N ETWORKS
            
            v_P = []
            
            passage_dimensions =  U_P.get_shape().as_list()
            question_dimensions =  U_Q.get_shape().as_list()
            
            batch_size = passage_dimensions[0]
            m = question_dimensions[1]
            n = passage_dimensions[1]
            self.hidden_state_units = passage_dimensions[2]
            
            W_Q_u = tf.Variable(tf.random_normal([batch_size, self.hidden_state_units, self.hidden_state_units], dtype=tf.float64))
            W_P_u = tf.Variable(tf.random_normal([batch_size, self.hidden_state_units, self.hidden_state_units], dtype=tf.float64))
            W_P_v = tf.Variable(tf.random_normal([batch_size, self.hidden_state_units, self.cell_size], dtype=tf.float64))
            v = tf.Variable(tf.random_normal([batch_size, self.hidden_state_units, 1], dtype=tf.float64))
            W_g = tf.Variable(tf.random_normal([batch_size, question_dimensions[2] + passage_dimensions[2], question_dimensions[2] + passage_dimensions[2]], dtype=tf.float64))
        
            
            rnn_cell = rnn.GRUCell(self.cell_size)
            v_P_t = rnn_cell.zero_state(batch_size, dtype=tf.float64) #v_t in the article
            
            W_Q_u_times_U_q_transpose = tf.matmul(W_Q_u, U_Q, transpose_b=True)
            
            for t in range(n):
                if t > 0:
                    varscope.reuse_variables()
                U_P_t = U_P[:,t,:]
                #U_Q_t = U_P[:,t,:]
                M_t = tf.tanh(W_Q_u_times_U_q_transpose + tf.matmul(W_P_u, tf.tile(tf.expand_dims(U_P_t, 2), [1,1,m])) + tf.matmul(W_P_v, tf.tile(tf.expand_dims(v_P_t, 2), [1,1,m])))
                a_t = tf.nn.softmax(tf.matmul(M_t, v, transpose_a = True))
                c_t = tf.matmul(U_Q, a_t, transpose_a = True)
                U_P_t_cat_c_t = tf.concat([tf.expand_dims(U_P_t, axis=2), c_t], axis=1)
                g_t = tf.sigmoid(tf.matmul(W_g, U_P_t_cat_c_t))
                U_P_t_cat_c_t_star = tf.multiply(g_t, U_P_t_cat_c_t)
                
                output_t, v_P_t = rnn_cell(tf.squeeze(U_P_t_cat_c_t_star), v_P_t)
                v_P_t = tf.nn.dropout(v_P_t, self.dropout_keep_prob)
                
                v_P.append(v_P_t)
                
            
            return tf.stack(v_P)
