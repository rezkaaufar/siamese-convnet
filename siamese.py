import tensorflow as tf

class siamese:
# create model

  def __init__(self, trigram_size):
    self.q1 = tf.placeholder(tf.float32, [None, trigram_size]) # question input
    self.q2 = tf.placeholder(tf.float32, [None, trigram_size]) # training : answer, testing : question
    with tf.variable_scope("siamese") as scope:
      self.o1 = self.network(self.q1)
      scope.reuse_variables()
      self.o2 = self.network(self.q2)
    # with tf.variable_scope("siamese1"):
    #   self.o1 = self.network(self.q1)
    # with tf.variable_scope("siamese2"):
    #   self.o2 = self.network(self.q2)

    # Create loss
    self.label = tf.placeholder(tf.float32, [None, 1]) # change the input to
    self.bm_score = tf.placeholder(tf.float32, [None, 1])
    self.test_len = tf.placeholder(tf.int32)
    self.loss = self.similarity_loss()
    self.sim = self.cosine_sim()
    self.cs = self.combined_score()
    self.acc = self.accuracy()
    self.pred_label = self.get_predicted_label()
    self.map = self.mean_average_precision()
    self.mrr = self.mean_reciprocal_rank()
    self.patk = self.precision_at_1()
    self.cmap = self.comb_mean_average_precision()
    self.cmrr = self.comb_mean_reciprocal_rank()
    self.cpatk = self.comb_precision_at_1()

  def network(self, x):
    x = tf.expand_dims(x, 2) # adding image height
    x = tf.expand_dims(x, 3) # adding image channels
    with tf.name_scope("model") as scope:
      with tf.variable_scope("conv_layer") as scope:
        out = tf.contrib.layers.conv2d(inputs=x, num_outputs=10, kernel_size=[1, 10], activation_fn=tf.nn.relu, padding='SAME',
                                      weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),scope=scope)
        out = tf.contrib.layers.max_pool2d(out, [1, 100], padding='SAME')
      # Flatten the output
      with tf.variable_scope("flatten_layer") as scope:
        out = tf.contrib.layers.flatten(inputs=out)
      # Final FC Layer
      with tf.variable_scope("fully_connected") as scope:
        out = tf.contrib.layers.fully_connected(inputs=out, num_outputs= 128, activation_fn=tf.nn.relu,
                                         weights_initializer=tf.contrib.layers.xavier_initializer(), scope=scope)

    return out

  def similarity_loss(self):
    m = 0.5 # arbitrary
    cos_dist = tf.losses.cosine_distance(tf.nn.l2_normalize(self.o1, 1), tf.nn.l2_normalize(self.o2, 1), dim=1,
                                         reduction="none")
    pos = tf.subtract(1.0,cos_dist)
    neg = tf.maximum(0.0, tf.subtract(cos_dist,m))
    L = self.label * pos + (tf.subtract(1.0,self.label)) * neg
    loss = tf.reduce_mean(L)
    return loss

  def cosine_sim(self):
    cos_dist = tf.losses.cosine_distance(tf.nn.l2_normalize(self.o1, 1), tf.nn.l2_normalize(self.o2, 1), dim=1,
                                         reduction="none")
    return tf.subtract(1.0,cos_dist)
    #return cos_dist

  def combined_score(self):
    alpha = 0.2
    cos_dist = tf.losses.cosine_distance(tf.nn.l2_normalize(self.o1, 1), tf.nn.l2_normalize(self.o2, 1), dim=1,
                                         reduction="none")
    cos_dist = tf.subtract(1.0,cos_dist)
    return tf.multiply(alpha, self.bm_score) + tf.multiply(1 - alpha, cos_dist)

  def accuracy(self):
    val = tf.cast(tf.less_equal(self.sim,  0.5), tf.int32) # self.sim is distance, not similarity
    val = tf.cast(val, tf.float32)
    return tf.reduce_sum(tf.cast(tf.equal(val, self.label), tf.int32))

  def get_predicted_label(self):
    val = tf.cast(tf.less_equal(self.sim,  0.5), tf.int32) # self.sim is distance, not similarity
    return tf.cast(val, tf.float32)

  # source : https://medium.com/@pds.bangalore/mean-average-precision-abd77d0b9a7e
  def mean_average_precision(self):
    sim_score = self.sim
    _, indices = tf.nn.top_k(tf.transpose(sim_score),k=self.test_len) # get the indices sorted by the similarity score
    indices = tf.transpose(indices)
    relevant_labels = tf.gather(self.label, indices=indices) # sort the labels by the sorted indices
    precision_at_k = tf.cumsum(relevant_labels)
    index_rel_labels = tf.expand_dims(tf.range(1, self.test_len+1),1) # get the index of the labels
    average_precision = tf.divide(tf.cast(precision_at_k, tf.float32), tf.cast(index_rel_labels, tf.float32)) # compute AP
    return tf.reduce_mean(average_precision)

  # source : https://machinelearning.wtf/terms/mean-reciprocal-rank-mrr/
  def mean_reciprocal_rank(self):
    sim_score = self.sim
    _, indices = tf.nn.top_k(tf.transpose(sim_score), k=self.test_len) # get the indices sorted by the similarity score
    indices = tf.transpose(indices)
    relevant_labels = tf.gather(self.label, indices=indices) # sort the labels by the sorted indices
    rank = tf.argmax(relevant_labels) # get the position of the highest predicted labels
    mrr = tf.divide(1,tf.add(rank,1))
    return mrr

  def precision_at_1(self):
    sim_score = self.sim
    _, indices = tf.nn.top_k(tf.transpose(sim_score), k=self.test_len) # get the indices sorted by the similarity score
    indices = tf.transpose(indices)
    relevant_labels = tf.gather(self.label, indices=indices) # sort the labels by the sorted indices
    precision1 = tf.gather(tf.squeeze(relevant_labels), indices=[0])
    return precision1

    # source : https://medium.com/@pds.bangalore/mean-average-precision-abd77d0b9a7e
  def comb_mean_average_precision(self):
    sim_score = self.cs
    _, indices = tf.nn.top_k(tf.transpose(sim_score),
                             k=self.test_len)  # get the indices sorted by the similarity score
    indices = tf.transpose(indices)
    relevant_labels = tf.gather(self.label, indices=indices)  # sort the labels by the sorted indices
    precision_at_k = tf.cumsum(relevant_labels)
    index_rel_labels = tf.expand_dims(tf.range(1, self.test_len + 1), 1)  # get the index of the labels
    average_precision = tf.divide(tf.cast(precision_at_k, tf.float32),
                                  tf.cast(index_rel_labels, tf.float32))  # compute AP
    return tf.reduce_mean(average_precision)

  # source : https://machinelearning.wtf/terms/mean-reciprocal-rank-mrr/
  def comb_mean_reciprocal_rank(self):
    sim_score = self.cs
    _, indices = tf.nn.top_k(tf.transpose(sim_score),
                             k=self.test_len)  # get the indices sorted by the similarity score
    indices = tf.transpose(indices)
    relevant_labels = tf.gather(self.label, indices=indices)  # sort the labels by the sorted indices
    rank = tf.argmax(relevant_labels)  # get the position of the highest predicted labels
    mrr = tf.divide(1, tf.add(rank, 1))
    return mrr

  def comb_precision_at_1(self):
    sim_score = self.cs
    _, indices = tf.nn.top_k(tf.transpose(sim_score),
                             k=self.test_len)  # get the indices sorted by the similarity score
    indices = tf.transpose(indices)
    relevant_labels = tf.gather(self.label, indices=indices)  # sort the labels by the sorted indices
    precision1 = tf.gather(tf.squeeze(relevant_labels), indices=[0])
    return precision1
