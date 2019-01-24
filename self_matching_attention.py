#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 11:25:01 2017

@author: squall
"""

import tensorflow as tf
from tensorflow.contrib import rnn

class SelfMatchingAttention:
    
    def __init__(self):
        print("self matching")
        self.hidden_state_units = 150
        self.cell_size = 75
        self.dropout_keep_prob = 0.8
        
    def get_self_matching_question_aware_passage_representation(self, V_P):
        
        with tf.variable_scope("self_matching_attention", reuse=None) as varscope:
            
            ###  G ATED A TTENTION - BASED R ECURRENT N ETWORKS
            
            h_P = []
            
            passage_dimensions =  V_P.get_shape().as_list()
            batch_size = passage_dimensions[1]
            n = passage_dimensions[0]
            
            self.hidden_state_units = passage_dimensions[2]
            
            W_P_v = tf.Variable(tf.random_normal([batch_size, self.hidden_state_units, self.hidden_state_units], dtype=tf.float64))
            W_Ptilde_v = tf.Variable(tf.random_normal([batch_size, self.hidden_state_units, self.hidden_state_units], dtype=tf.float64))
            v = tf.Variable(tf.random_normal([batch_size, self.hidden_state_units, 1], dtype=tf.float64))
            W_g = tf.Variable(tf.random_normal([batch_size, passage_dimensions[2] + passage_dimensions[2], passage_dimensions[2] + passage_dimensions[2]], dtype=tf.float64))
        
            
            forward_rnn_cell = rnn.GRUCell(self.cell_size)
            backward_rnn_cell = rnn.GRUCell(self.cell_size)
            h_P_t_forward = forward_rnn_cell.zero_state(batch_size, dtype=tf.float64) 
            h_P_t_backward = backward_rnn_cell.zero_state(batch_size, dtype=tf.float64)
            
            V_P = tf.transpose(V_P, [1, 2, 0])
            
            W_P_v_times_V_P_transpose =  tf.transpose(tf.matmul(W_P_v, V_P), perm=[0,2,1])
            
            h_P_forward = []
            h_P_backward = []
            
            for t in range(n):
                if t > 0:
                    varscope.reuse_variables()
                
                with tf.variable_scope("forward_rnn", reuse=None) as forward_scope:
                    
                    if t > 0:
                        forward_scope.reuse_variables()
                    
                    V_P_t = V_P[:,:,t]
                    
                    c_t = self.compute_c_t(V_P_t, V_P, W_Ptilde_v, W_P_v, n, v, W_P_v_times_V_P_transpose)
                    
                    V_P_t_cat_c_t = tf.concat([tf.expand_dims(V_P_t, 2), c_t], axis=1)
                    
                    g_t = tf.sigmoid(tf.matmul(W_g, V_P_t_cat_c_t))
                    V_P_t_cat_c_t_star = tf.multiply(g_t, V_P_t_cat_c_t)
                    
                    _, h_P_t_forward = forward_rnn_cell(tf.squeeze(V_P_t_cat_c_t_star), h_P_t_forward)
                    h_P_t_forward = tf.nn.dropout(h_P_t_forward, self.dropout_keep_prob)
                    h_P_forward.append(h_P_t_forward)
                
                with tf.variable_scope("backward_rnn", reuse=None) as backward_scope:
                    
                    if t > 0:
                        backward_scope.reuse_variables()
                
                    V_P_t = V_P[:,:,n - 1 - t]
                    
                    c_t = self.compute_c_t(V_P_t, V_P, W_Ptilde_v, W_P_v, n, v, W_P_v_times_V_P_transpose)
                    
                    V_P_t_cat_c_t = tf.concat([tf.expand_dims(V_P_t, 2), c_t], axis=1)
                    
                    g_t = tf.sigmoid(tf.matmul(W_g, V_P_t_cat_c_t))
                    V_P_t_cat_c_t_star = tf.multiply(g_t, V_P_t_cat_c_t)
                    
                    _, h_P_t_backward = backward_rnn_cell(tf.squeeze(V_P_t_cat_c_t_star), h_P_t_backward)
                    h_P_t_backward = tf.nn.dropout(h_P_t_backward, self.dropout_keep_prob)
                    h_P_backward.append(h_P_t_backward)
                
            h_P = tf.concat([tf.stack(h_P_forward), tf.stack(h_P_backward)], axis=2)
            
            return h_P
                
    def compute_c_t(self, V_P_t, V_P, W_Ptilde_v, W_P_v, n, v, W_P_v_times_V_P_transpose):
        
        stacked_W_Ptilde_v_times_V_P_t_transpose = tf.transpose(tf.tile(tf.matmul(W_Ptilde_v, tf.expand_dims(V_P_t, 2)), [1,1,n]), perm=[0,2,1])
        
        s_t = tf.matmul(tf.tanh(W_P_v_times_V_P_transpose + stacked_W_Ptilde_v_times_V_P_t_transpose), v)
                
        a_t = tf.nn.softmax(s_t)
                
        c_t = tf.matmul(V_P, a_t)
        
        
        return c_t
            