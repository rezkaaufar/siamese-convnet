import tensorflow as tf
import numpy as np
import preprocess
import os
import json
import codecs

import siamese
import utils

tri2index = preprocess.load_letter_trigram()  # dictionary to convert trigram to index (for one hot vector)
TRIGRAM_SIZE = len(tri2index)

# prepare data and tf.session
Xs_train, Ys_train = utils.read_dataset()
QA_train = utils.Dataset(Xs_train, Ys_train)

# test data
def generate_test_data():
  with open('./dataset/test_batch.dump', 'r', encoding='utf-8') as f:
    test_batch = json.load(f)
    return test_batch

sess = tf.InteractiveSession()

# setup siamese network
siamese = siamese.siamese(TRIGRAM_SIZE)
train_op = tf.train.AdamOptimizer(0.01).minimize(siamese.loss)

init = tf.global_variables_initializer()
sess.run(init)
saver = tf.train.Saver(tf.global_variables())

def BM25_score():
  with open('./dataset/BM25_score.dump', 'r', encoding='utf-8') as f:
    test_batch = json.load(f)
    return test_batch

test_batch = generate_test_data()
bm_score = BM25_score()
TRAIN_STEP = 1001

maps = []
mrrs = []
patks = []
cmaps = []
cmrrs = []
cpatks = []

for step in range(TRAIN_STEP):
  qa_pairs, labels = QA_train.next_batch(128)
  _, loss_v = sess.run([train_op, siamese.loss], feed_dict={
                      siamese.q1: qa_pairs[:,0,:],
                      siamese.q2: qa_pairs[:,1,:],
                      siamese.label: np.expand_dims(labels,1)})

  if np.isnan(loss_v):
      print('Model diverged with loss = NaN')
      quit()

  if step % 100 == 0 and step != 0:
      print ('step %d: loss %.3f' % (step, loss_v))

  if step % 100 == 0 and step != 0:
    map = 0
    mrr = 0
    patk = 0
    cmap = 0
    cmrr = 0
    cpatk = 0
    for pl, scr in zip(test_batch, bm_score):
    #pl = test_batch[0]
      qa_pairs_test = np.asarray(pl[0])
      labels_test = np.asarray(pl[1])

      scr = np.expand_dims(np.asarray(scr),1)
      ap, rr, pat1, cap, crr, cpat1 = sess.run([siamese.map, siamese.mrr, siamese.patk,
                               siamese.cmap, siamese.cmrr, siamese.cpatk], feed_dict={
        siamese.q1: qa_pairs_test[:,0,:],
        siamese.q2: qa_pairs_test[:,1,:],
        siamese.label: np.expand_dims(labels_test,1),
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
    maps.append(map / len(test_batch))
    mrrs.append(mrr / len(test_batch))
    patks.append(patk / len(test_batch))
    cmaps.append(map / len(test_batch))
    cmrrs.append(mrr / len(test_batch))
    cpatks.append(patk / len(test_batch))

  if step % 500 == 0 and step != 0:
    checkpoint_path = './model/model.ckpt'
    saver.save(sess, checkpoint_path, global_step=step)
    print("model saved to {}".format(checkpoint_path))

mpp = "./dataset/maps.dump"
mrp = "./dataset/mrrs.dump"
pkp = "./dataset/patks.dump"
cmpp = "./dataset/cmaps.dump"
cmrp = "./dataset/cmrrs.dump"
cpkp = "./dataset/cpatks.dump"
json.dump(maps, codecs.open(mpp, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)
json.dump(mrrs, codecs.open(mrp, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)
json.dump(patks, codecs.open(pkp, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)
json.dump(cmaps, codecs.open(cmpp, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)
json.dump(cmrrs, codecs.open(cmrp, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)
json.dump(cpatks, codecs.open(cpkp, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)

