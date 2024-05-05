import tensorflow as tf
import numpy as np
# shape 300
class my_fc6(tf.keras.Model):
    def __init__(self, num_classes=300):
        super(my_fc6, self).__init__()
        self.fc = tf.keras.layers.Dense(num_classes, activation='softmax')
        # either add another dense layer or just a softmax
        #self.relu = tf.keras.layers.ReLU()

    def call(self, x):
        print("Before Dense layer, shape of x:", x.shape)
        x = self.fc(x)
        # x = self.relu(x)
        return x


def save_fc6_model_weights(model, save_path):
    """Save the weights of the Dense layer to a file."""
    model.save_weights(save_path, save_format="tf")
