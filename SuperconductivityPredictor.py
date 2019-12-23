import tensorflow as tf
import numpy as np


class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        self.vocab_size = 100 #num elements

        # recurrent layer
        self.GRU = layers.GRU(
            100, return_sequences=True, return_state=True, dtype="float32",
        )
        self.embeddings = tf.Variable(tf.random.truncated_normal((self.vocab_size,100), dtype=tf.float32, stddev=.1))

        self.P = Sequential()
        self.P.add(layers.Dense(100,activation="relu", dtype="float32"))
        self.P.add(layers.Dense(100,activation="relu", dtype="float32",use_bias=False))
        self.P.add(layers.Dense(1,activation="relu", dtype="float32",use_bias=False))

    def call(self, inputs):
        inputs = tf.nn.embedding_lookup(self.embeddings,inputs)
        inputs, *next_state = self.GRU(inputs, initial_state=hidden)
        return inputs

    def loss(self, inputs, labels):
        predictions = self.call(inputs)
        return
