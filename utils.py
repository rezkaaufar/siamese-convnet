import csv
import preprocess as p
import numpy as np
import pickle
import json

# size of question and answer pair quora data : 404291
DATA_PATH = "dataset/quora_duplicate_questions.tsv"
tri2index = p.load_letter_trigram() # dictionary to convert trigram to index (for one hot vector)
TRIGRAM_SIZE = len(tri2index)

def split_dataset(test_amount, qap_path="./dataset/qa_pairs.dump", label_path="./dataset/labels.dump"):
  Xs = pickle.load(open(qap_path, "rb"))
  Ys = pickle.load(open(label_path, "rb"))
  Xs_test = Xs[:test_amount]
  Ys_test = Ys[:test_amount]
  Xs_train = Xs[test_amount:]
  Ys_train = Ys[test_amount:]
  return Xs_train, Ys_train, Xs_test, Ys_test

def read_dataset(qap_path="./dataset/qa_pairs.dump", label_path="./dataset/labels.dump"):
  Xs = pickle.load(open(qap_path, "rb"))
  Ys = pickle.load(open(label_path, "rb"))
  return Xs, Ys

def read_dataset_json(qap_path="./dataset/qa_pairs_train_red.dump", label_path="./dataset/labels_train_red.dump"):
  with open(qap_path, 'r', encoding='utf-8') as f:
    Xs = json.load(f)
  with open(label_path, 'r', encoding='utf-8') as f:
    Ys = json.load(f)
  Xs = np.asarray(Xs)
  Ys = np.asarray(Ys)
  return Xs, Ys

class Dataset(object):
  def __init__(self, qas, labels):
    """
    Builds dataset with qa pair and labels.
    Args:
      qas: Question and Answer data. (tuple of list)
      labels: Labels data
    """
    assert qas.shape[0] == labels.shape[0], (
      "images.shape: {0}, labels.shape: {1}".format(str(qas.shape), str(labels.shape)))

    self._num_examples = qas.shape[0]
    self._qas = qas
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def qas(self):
    return self.qas

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size):
    """
    Return the next `batch_size` examples from this data set.
    Args:
      batch_size: Batch size.
    """
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      self._epochs_completed += 1

      perm = np.arange(self._num_examples)
      np.random.shuffle(perm)
      self._qas = self._qas[perm]
      self._labels = self._labels[perm]

      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples

    end = self._index_in_epoch

    # cast batch to one hot representation
    one_hot_batch = []
    for qa_pair in self._qas[start:end]:
      q_one_hot = np.zeros((TRIGRAM_SIZE))
      a_one_hot = np.zeros((TRIGRAM_SIZE))
      q_one_hot[qa_pair[0]] = 1
      a_one_hot[qa_pair[1]] = 1
      one_hot_batch.append((q_one_hot, a_one_hot))
    return np.asarray(one_hot_batch), self._labels[start:end]

# testing
# print(TRIGRAM_SIZE)
# construct_dataset_yahoo()
# Xs_train, Ys_train, Xs_test, Ys_test = split_dataset(100)
# QA = DatasetSequence(Xs_test, Ys_test)
# for _ in range(12):
#   qa_pairs, labels = QA.next_batch(10)
#   print(labels)
#   print(qa_pairs[:,0,:].shape) # one-hot encoded question in batch
#   print(qa_pairs[:,1,:].shape) # one-hot encoded answer in batch