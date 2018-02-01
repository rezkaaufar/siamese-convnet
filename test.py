import tensorflow as tf
import numpy as np
import preprocess
import os
import BM25
import json

import siamese
import utils

# TODO
# Mean Average Precision
# Mean Recripocal Rank
# Precision @ K

tri2index = preprocess.load_letter_trigram() # dictionary to convert trigram to index (for one hot vector)
TRIGRAM_SIZE = len(tri2index)

# tf.session
sess = tf.InteractiveSession()

# setup siamese network
siamese = siamese.siamese(TRIGRAM_SIZE)
train_op = tf.train.AdamOptimizer(0.01).minimize(siamese.loss)

init = tf.global_variables_initializer()
sess.run(init)
saver = tf.train.Saver(tf.global_variables())

# test data
def generate_test_data():
  with open('./dataset/test_batch.dump', 'r', encoding='utf-8') as f:
    test_batch = json.load(f)
    return test_batch

checkpoint_path = './model/model.ckpt'
ckpt = tf.train.get_checkpoint_state('./model/')
saver.restore(sess, ckpt.model_checkpoint_path)
print("model loaded from {}".format(checkpoint_path))

# acc, pred_label = sess.run([siamese.acc, siamese.pred_label], feed_dict={
#       siamese.q1: qa_pairs_test[:,0,:],
#       siamese.q2: qa_pairs_test[:,1,:],
#       siamese.label: np.expand_dims(labels_test,1)})
# print("accuracy {}".format(acc / len(labels_test)))
# print(pred_label.T, labels_test)

def BM25_score():
  with open('./dataset/BM25_score.dump', 'r', encoding='utf-8') as f:
    test_batch = json.load(f)
    return test_batch

test_batch = generate_test_data()
bm_score = BM25_score()

maps = []
mrrs = []
patks = []
cmaps = []
cmrrs = []
cpatks = []

map = 0
mrr = 0
patk = 0
cmap = 0
cmrr = 0
cpatk = 0

for pl, scr in zip(test_batch, bm_score):
  # pl = test_batch[0]
  qa_pairs_test = np.asarray(pl[0])
  labels_test = np.asarray(pl[1])

  # ### YAHOO CODE ###
  # one_hot_batch = []
  # for elem in qa_pairs_test:
  #   q1_one_hot = np.zeros((TRIGRAM_SIZE))
  #   q2_one_hot = np.zeros((TRIGRAM_SIZE))
  #   q1_one_hot[elem[0]] = 1
  #   q2_one_hot[elem[1]] = 1
  #   one_hot_batch.append((q1_one_hot, q2_one_hot))
  # qa_pairs_test = np.asarray(one_hot_batch)
  #
  # ### END ###

  scr = np.expand_dims(np.asarray(scr), 1)
  ap, rr, pat1, cap, crr, cpat1 = sess.run([siamese.map, siamese.mrr, siamese.patk,
                                            siamese.cmap, siamese.cmrr, siamese.cpatk], feed_dict={
    siamese.q1: qa_pairs_test[:, 0, :],
    siamese.q2: qa_pairs_test[:, 1, :],
    siamese.label: np.expand_dims(labels_test, 1),
    siamese.test_len: qa_pairs_test.shape[0],
    siamese.bm_score: scr})
  map += ap
  mrr += rr[0][0]
  patk += pat1[0]
  cmap += cap
  cmrr += crr[0][0]
  cpatk += cpat1[0]
print("mean average precision {}".format(map / len(test_batch)))
print("mean reciprocal rank {}".format(mrr / len(test_batch)))
print("precision at 1 {}".format(patk / len(test_batch)))
print("combined mean average precision {}".format(cmap / len(test_batch)))
print("combined mean reciprocal rank {}".format(cmrr / len(test_batch)))
print("combined precision at 1 {}".format(cpatk / len(test_batch)))