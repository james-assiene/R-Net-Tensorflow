#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 16:04:33 2017

@author: squall
"""

import tensorflow as tf
from tensorflow.contrib import rnn

class RNETOutputLayer:
    
    def __init__(self):
        print("output")
        
        self.hidden_state_units = 150
        self.cell_size = 150
        self.dropout_keep_prob = 0.8
        
    def orig_get_predictions(self, H_P, U_Q):
        
        indices = []
        
        with tf.variable_scope("rnet_output_layer", reuse=None) as varscope:
            
            ###  G ATED A TTENTION - BASED R ECURRENT N ETWORKS
            
            passage_dimensions =  H_P.get_shape().as_list()
            question_dimensions = U_Q.get_shape().as_list()
            batch_size = passage_dimensions[1]
            n = passage_dimensions[0]
            m = question_dimensions[1]
            
            self.hidden_state_units = passage_dimensions[2]
            self.cell_size = self.hidden_state_units
            
            W_P_h = tf.Variable(tf.random_normal([batch_size, self.hidden_state_units, self.hidden_state_units], dtype=tf.float64))
            W_a_h = tf.Variable(tf.random_normal([batch_size, self.hidden_state_units, self.cell_size], dtype=tf.float64))
            W_Q_u = tf.Variable(tf.random_normal([batch_size, self.hidden_state_units, self.hidden_state_units], dtype=tf.float64))
            W_Q_v = tf.Variable(tf.random_normal([batch_size, self.hidden_state_units, self.hidden_state_units], dtype=tf.float64))
            V_Q_r = tf.Variable(tf.random_normal([batch_size, self.hidden_state_units, m], dtype=tf.float64))
            v = tf.Variable(tf.random_normal([batch_size, self.hidden_state_units, 1], dtype=tf.float64))
            
            rnn_cell = rnn.GRUCell(self.cell_size)
            h_a_t = rnn_cell.zero_state(batch_size, dtype=tf.float64) #v_t in the article
            
            H_P = tf.transpose(H_P, perm=[1, 2, 0])
            U_Q = tf.transpose(U_Q, perm=[0, 2, 1])
            
            W_P_h_times_h_P = tf.matmul(W_P_h, H_P)
                        
            s = tf.matmul(tf.tanh(tf.matmul(W_Q_u, U_Q) + tf.matmul(W_Q_v, V_Q_r)), v, transpose_a=True)
            a = tf.nn.softmax(s)
            r_Q = tf.matmul(U_Q, a)
            
            h_a_t = tf.squeeze(r_Q)
            
            result = []
            
            for t in range(2):
                if t > 0:
                    varscope.reuse_variables()
                    
                stacked_W_a_h_times_h_a_t = tf.tile(tf.matmul(W_a_h, tf.expand_dims(h_a_t, 2)), [1,1,n])
                M_t = tf.tanh(W_P_h_times_h_P + stacked_W_a_h_times_h_a_t)
                s_t = tf.matmul(M_t, v, transpose_a=True)
                
                a_t = tf.nn.softmax(s_t)
                
                p_t = tf.argmax(a_t, axis=1)
                
                print(p_t.get_shape())
                
                c_t = tf.matmul(H_P, a_t)
                
                _, h_a_t = rnn_cell(tf.squeeze(c_t), h_a_t)
                
                result.append(tf.one_hot(tf.squeeze(p_t), n))
            
            print(result[0].get_shape())
                
            return result
        
    def get_predictions(self, H_P, U_Q):
        
        indices = []
        
        with tf.variable_scope("rnet_output_layer", reuse=None) as varscope:
            
            ###  G ATED A TTENTION - BASED R ECURRENT N ETWORKS
            
            passage_dimensions =  H_P.get_shape().as_list()
            question_dimensions = U_Q.get_shape().as_list()
            batch_size = passage_dimensions[1]
            n = passage_dimensions[0]
            m = question_dimensions[1]
            
            self.hidden_state_units = passage_dimensions[2]
            self.cell_size = self.hidden_state_units
            
            W_P_h = tf.Variable(tf.random_normal([batch_size, self.hidden_state_units, self.hidden_state_units], dtype=tf.float64))
            W_a_h = tf.Variable(tf.random_normal([batch_size, self.hidden_state_units, self.cell_size], dtype=tf.float64))
            W_Q_u = tf.Variable(tf.random_normal([batch_size, self.hidden_state_units, self.hidden_state_units], dtype=tf.float64))
            W_Q_v = tf.Variable(tf.random_normal([batch_size, self.hidden_state_units, self.hidden_state_units], dtype=tf.float64))
            V_Q_r = tf.Variable(tf.random_normal([batch_size, self.hidden_state_units, m], dtype=tf.float64))
            v = tf.Variable(tf.random_normal([batch_size, self.hidden_state_units, 1], dtype=tf.float64))
            
            rnn_cell = rnn.GRUCell(self.cell_size)
            h_a_t = rnn_cell.zero_state(batch_size, dtype=tf.float64) #v_t in the article
            
            H_P = tf.transpose(H_P, perm=[1, 2, 0])
            U_Q = tf.transpose(U_Q, perm=[0, 2, 1])
            
            W_P_h_times_h_P = tf.matmul(W_P_h, H_P)
                        
            s = tf.matmul(tf.tanh(tf.matmul(W_Q_u, U_Q) + tf.matmul(W_Q_v, V_Q_r)), v, transpose_a=True)
            a = tf.nn.softmax(s)
            r_Q = tf.matmul(U_Q, a)
            
            h_a_t = tf.squeeze(r_Q)
            
            result = []
            deb = []
            
            for t in range(2):
                if t > 0:
                    varscope.reuse_variables()
                    
                stacked_W_a_h_times_h_a_t = tf.tile(tf.matmul(W_a_h, tf.expand_dims(h_a_t, 2)), [1,1,n])
                M_t = tf.tanh(W_P_h_times_h_P + stacked_W_a_h_times_h_a_t)
                s_t = tf.matmul(M_t, v, transpose_a=True)
                
                a_t = tf.nn.softmax(s_t,1)
                
                p_t = tf.argmax(a_t, axis=1)
                
                print(p_t.get_shape())
                
                c_t = tf.matmul(H_P, a_t)
                
                _, h_a_t = rnn_cell(tf.squeeze(c_t), h_a_t)
                h_a_t = tf.nn.dropout(h_a_t, self.dropout_keep_prob)
                
                result.append(tf.squeeze(a_t))
                deb.append(s_t)
            
            print(result[0].get_shape())
            tf.summary.histogram("s_t_1", deb[0][0])
            tf.summary.histogram("s_t_2", deb[1][0])
            tf.summary.histogram("a_t_1", result[0][0])
            tf.summary.histogram("a_t_2", result[1][0])
#            tf.summary.histogram("sm1_0", tf.nn.softmax(deb[0],0))
#            tf.summary.histogram("sm1_1", tf.nn.softmax(deb[0],1))
#            tf.summary.histogram("sm1_2", tf.nn.softmax(deb[0],2))
#            tf.summary.histogram("sm1_0s", tf.nn.softmax(tf.squeeze(deb[0]),0))
#            tf.summary.histogram("sm1_1s", tf.nn.softmax(tf.squeeze(deb[0]),1))
#            
#            tf.summary.histogram("sm2_0", tf.nn.softmax(deb[1],0))
#            tf.summary.histogram("sm2_1", tf.nn.softmax(deb[1],1))
#            tf.summary.histogram("sm2_2", tf.nn.softmax(deb[1],2))
#            tf.summary.histogram("sm2_0s", tf.nn.softmax(tf.squeeze(deb[1]),0))
#            tf.summary.histogram("sm2_1s", tf.nn.softmax(tf.squeeze(deb[1]),1))
                
            return result