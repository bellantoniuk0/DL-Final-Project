import tensorflow as tf

#sixth fully connected layer

class my_fc6(tf.keras.layers.Layer):
    def __init__(self):
        super(my_fc6, self).__init__()
        self.fc = tf.keras.layers.Dense(4096)
        self.relu = tf.keras.layers.ReLU()

    def call(self, x):
        x = self.fc(x)
        x = self.relu(x)
        return x