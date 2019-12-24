import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from os import path
from tensorflow.compat.v2.keras import layers

from preprocessing import *




class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

        self.batch_size = 100

        # recurrent layer
        self.GRU = layers.GRU(
            600, return_sequences=False, return_state=True, dtype="float32",
        )
        self.activation = layers.LeakyReLU()

        self.dense_1 = layers.Dense(1000,activation=self.activation, dtype="float32")
        self.dense_2 = layers.Dense(1000,activation=self.activation, dtype="float32")
        self.dense_3 = layers.Dense(500,activation=self.activation, dtype="float32")
        self.dense_4 = layers.Dense(1,activation="relu", dtype="float32")

    def call(self, inputs, initial_state):
        inputs, *next_state = self.GRU(inputs, initial_state=initial_state)
        inputs = self.dense_1(inputs)
        inputs = self.dense_2(inputs)
        inputs = self.dense_3(inputs)
        inputs = self.dense_4(inputs)
        return inputs, next_state

    def loss(self, predictions, labels):
        return tf.keras.losses.MSE(labels,predictions)


def conv(x):
    return tf.convert_to_tensor([x], dtype=tf.float32)

def train(model,train_inputs,train_labels):
    batch_start = 0
    while batch_start < len(train_inputs) - 1:
        batch_inputs = train_inputs[batch_start:batch_start+model.batch_size]
        batch_labels = train_labels[batch_start:batch_start+model.batch_size]

        loss = 0
        for i in range(len(batch_inputs)):
            initial_state = None
            with tf.GradientTape() as tape:
                try:
                    predictions, _ = model.call(conv(batch_inputs[i]),initial_state)
                except:
                    predictions, _ = model.call(conv(batch_inputs[i]),initial_state)
                loss += tf.reduce_sum(model.loss(predictions,conv(batch_labels[i])))

        gradients = tape.gradient(loss,model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients,model.trainable_variables))

        batch_start += model.batch_size


def test(model,test_inputs,test_labels):
    batch_start = 0

    loss = []

    while batch_start < len(test_labels) - 1:
        batch_inputs = test_inputs[batch_start:batch_start+model.batch_size]
        batch_labels = test_labels[batch_start:batch_start+model.batch_size]

        loss_batch = 0
        for i in range(len(batch_inputs)):
            initial_state = None
            predictions, _ = model.call(conv(batch_inputs[i]),initial_state)
            loss_batch += model.loss(predictions,conv(batch_labels[i]))

        loss_batch /= len(batch_inputs)
        loss.append(loss_batch)

        batch_start += model.batch_size
    return tf.reduce_mean(loss)


def prepare_training_testing_data(test_fraction = .1):

    if path.exists('./Data/data_inputs.npy'):
        data_inputs = np.load('./Data/data_inputs.npy', allow_pickle=True)
        data_labels = np.load('./Data/data_labels.npy', allow_pickle=True)
    else:
        data_inputs, data_labels = load_data('./Data/Supercon_data.csv')
        np.save('./Data/data_inputs.npy',data_inputs)
        np.save('./Data/data_labels.npy',data_labels)

    shuffle = np.array(np.random.permutation(len(data_inputs)))
    data_inputs = data_inputs[shuffle]
    data_labels = data_labels[shuffle]

    split = int(len(data_inputs)*test_fraction)

    train_inputs = data_inputs[split:]
    train_labels = data_labels[split:]

    test_inputs = data_inputs[:split]
    test_labels = data_labels[:split]

    return train_inputs, train_labels, test_inputs, test_labels

loss_list = []

def main():
    # setup plot
    plt.ion()
    plt.show()


    train_inputs, train_labels, test_inputs, test_labels = prepare_training_testing_data()
    model = Model()
    for epoch in range(1000):
        print("EPOCH: ",epoch)
        train(model,train_inputs,train_labels)
        if epoch % 10 == 0:
            loss = test(model,test_inputs,test_labels)
            loss_list.append(loss)
            plt.plot(loss_list)
            plt.draw()
            plt.pause(.001)
            plt.cla()



    while True:
        try:
            comp = input("Compound: ")
            split = split_compound_string(comp)
            arr = compound_to_array(split, 100)
            pred, _ = model.call(conv(arr),None)
            print(pred)
        except:
            print("Something Broke")


if __name__ == '__main__':
    main()
