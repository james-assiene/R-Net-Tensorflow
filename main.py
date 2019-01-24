#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 14:49:37 2017

@author: squall
"""

import tensorflow as tf
import numpy as np
from embeddings import Embeddings
from random import randint
from parse_dataset import SquadDatasetParser
from question_passage_encoder import QuestionAndPassageEncoder
from gated_attention_based_rnn import GatedAttentionBasedRNN
from self_matching_attention import SelfMatchingAttention
from r_net_output_layer import RNETOutputLayer
from batch_manager import BatchManager

#tf.reset_default_graph()

#with tf.variable_scope("root"):

max_number_of_words = 20000
max_sequence_length = 300
max_word_length = 30
word_embedding_dim = 300
char_embedding_dim = 300

dataset_parser = SquadDatasetParser()
train_dataset = dataset_parser.get_documents_questions_answers()
test_dataset_parser = SquadDatasetParser(file_path="datasets/dev-v1.1.json")
test_dataset = test_dataset_parser.get_documents_questions_answers()

batch_size = 32
batch_manager = BatchManager(train_dataset, batch_size, max_sequence_length)
test_batch_manager = BatchManager(test_dataset, batch_size, max_sequence_length)
train_epoch_size = batch_manager.get_qapairs_number() / batch_size
nb_epochs = 50

E_P = tf.placeholder(tf.float64, shape=(batch_size, max_sequence_length, word_embedding_dim), name="encoded_passage")
E_Q = tf.placeholder(tf.float64, shape=(batch_size, max_sequence_length, word_embedding_dim), name="encoded_question")
pre_C_P = tf.placeholder(tf.float64, shape=(batch_size, max_sequence_length, max_word_length, char_embedding_dim), name="pre_encoded_passage_char")
pre_C_Q = tf.placeholder(tf.float64, shape=(batch_size, max_sequence_length, max_word_length, char_embedding_dim), name="pre_encoded_question_char")

encoder = QuestionAndPassageEncoder()
U_Q = encoder.encode_question(E_Q, pre_C_Q)
U_P = encoder.encode_passage(E_P, pre_C_P)

gated_attention_based_rnn = GatedAttentionBasedRNN()
V_P = gated_attention_based_rnn.get_question_aware_passage_representation(U_P, U_Q)

self_matching_attention = SelfMatchingAttention()
H_P = self_matching_attention.get_self_matching_question_aware_passage_representation(V_P)

r_net_output_layer = RNETOutputLayer()
start_prediction, end_prediction = r_net_output_layer.get_predictions(H_P, U_Q)


si = tf.placeholder(tf.int64, shape=(batch_size), name="ground_truth_start_index")
ei = tf.placeholder(tf.int64, shape=(batch_size), name="ground_truth_end_index")

si_distribution = tf.one_hot(si, max_sequence_length, dtype=tf.float64)
ei_distribution = tf.one_hot(ei, max_sequence_length, dtype=tf.float64)

with tf.variable_scope("xent"):
#    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=si, logits=start_prediction) + tf.nn.sparse_softmax_cross_entropy_with_logits(labels=ei, logits=end_prediction)
#    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=si_distribution, logits=start_prediction) + tf.nn.softmax_cross_entropy_with_logits(labels=ei_distribution, logits=end_prediction)
    cross_entropy = tf.reduce_sum(tf.multiply(si_distribution, -tf.log(start_prediction)) + tf.multiply(ei_distribution, -tf.log(end_prediction)))
    loss = cross_entropy
#    
with tf.variable_scope("accuracy"):
    start_prediction_one_hot = tf.argmax(start_prediction, axis=1)
    end_prediction_one_hot = tf.argmax(end_prediction, axis=1)
    em_bool = tf.logical_and(tf.equal(start_prediction_one_hot, si), tf.equal(end_prediction_one_hot, ei))
    em = tf.reduce_mean(tf.cast(em_bool, tf.float64), axis=0)
    
    start_accuracy = tf.reduce_mean(tf.cast(tf.equal(start_prediction_one_hot, si), tf.float64), axis=0)
    end_accuracy = tf.reduce_mean(tf.cast(tf.equal(end_prediction_one_hot, ei), tf.float64), axis=0)
    
    tf.summary.scalar("exact_match", em)
    tf.summary.scalar("start_pred_acc", start_accuracy)
    tf.summary.scalar("end_pred_acc", end_accuracy)
    
    
    
lr = 1 # learning rate
rho = 0.95
epsilon = 1e-6
optimizer = tf.train.AdadeltaOptimizer(learning_rate=lr, rho=rho, epsilon=epsilon)
tf.summary.scalar("loss", loss)

print("optimizer start")

global_step = tf.Variable(0, dtype=tf.int64, name="global_step", trainable=False)
train_step = optimizer.minimize(loss, global_step=global_step)
print("optimizer done")


#sess = tf.InteractiveSession()
merged = tf.summary.merge_all()
#train_writer = tf.summary.FileWriter("./log/train")
#test_writer = tf.summary.FileWriter("./log/test")
#train_writer.add_graph(sess.graph)
#sess.run(tf.global_variables_initializer())

sv = tf.train.Supervisor(logdir="./training_supervisor",
                     summary_op=None, save_model_secs=600) # Do not run the summary service
with sv.managed_session() as sess:
    for step in range(0, round(nb_epochs * train_epoch_size)):
        print("feed : {}".format(step))
        e_p, e_q, pre_c_p, pre_c_q, s_i, e_i = batch_manager.next_batch()
        
        print("run")
        if sv.should_stop():
            break
        if step % 50 == 0:
            summary, _, current_loss, start_acc, end_acc, ex_match = sess.run([merged, train_step, loss, start_accuracy, end_accuracy, em], feed_dict={E_P: e_p, E_Q: e_q, pre_C_P: pre_c_p, pre_C_Q: pre_c_q, si: np.array(s_i, dtype=np.int64), ei: np.array(e_i, dtype=np.int64)})
            print("loss: : {}, sa : {}, ea : {}, em : {}".format(current_loss, start_acc, end_acc, ex_match))
            sv.summary_computed(sess, summary)
            
            t_e_p, t_e_q, t_pre_c_p, t_pre_c_q, t_s_i, t_e_i = test_batch_manager.next_batch()
            summary, test_loss, test_start_acc, test_end_acc, test_ex_match = sess.run([merged, loss, start_accuracy, end_accuracy, em], feed_dict={E_P: t_e_p, E_Q: t_e_q, pre_C_P: t_pre_c_p, pre_C_Q: t_pre_c_q, si: np.array(t_s_i, dtype=np.int64), ei: np.array(t_e_i, dtype=np.int64)})
            print("test_loss: : {}, test_sa : {}, test_ea : {}, test_em : {}".format(test_loss, test_start_acc, test_end_acc, test_ex_match))
            sv.summary_computed(sess, summary)
        else:
            _, current_loss, start_acc, end_acc, ex_match = sess.run([train_step, loss, start_accuracy, end_accuracy, em], feed_dict={E_P: e_p, E_Q: e_q, pre_C_P: pre_c_p, pre_C_Q: pre_c_q, si: np.array(s_i, dtype=np.int64), ei: np.array(e_i, dtype=np.int64)})
            print("loss: : {}, sa : {}, ea : {}, em : {}".format(current_loss, start_acc, end_acc, ex_match))
            
            
        
#with sess.as_default():
#    for i in range(round(30)):
#        print("feed : {}".format(i))
#        e_p, e_q, pre_c_p, pre_c_q, s_i, e_i = batch_manager.next_batch()
#        
#        print("run")
#        
#        summary, _, current_loss, start_acc, end_acc, ex_match = sess.run([merged, train_step, loss, start_accuracy, end_accuracy, em], feed_dict={E_P: e_p, E_Q: e_q, pre_C_P: pre_c_p, pre_C_Q: pre_c_q, si: np.array(s_i, dtype=np.int64), ei: np.array(e_i, dtype=np.int64)})
#        print("loss: : {}, sa : {}, ea : {}, em : {}".format(current_loss, start_acc, end_acc, ex_match))
#        train_writer.add_summary(summary, i)
#        
#        t_e_p, t_e_q, t_pre_c_p, t_pre_c_q, t_s_i, t_e_i = test_batch_manager.next_batch()
#        summary, test_loss, test_start_acc, test_end_acc, test_ex_match = sess.run([merged, loss, start_accuracy, end_accuracy, em], feed_dict={E_P: t_e_p, E_Q: t_e_q, pre_C_P: t_pre_c_p, pre_C_Q: t_pre_c_q, si: np.array(t_s_i, dtype=np.int64), ei: np.array(t_e_i, dtype=np.int64)})
#        print("test_loss: : {}, test_sa : {}, test_ea : {}, test_em : {}".format(test_loss, test_start_acc, test_end_acc, test_ex_match))
#        test_writer.add_summary(summary, i)
#        
#    sess.close()
