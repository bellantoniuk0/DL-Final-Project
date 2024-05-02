import tensorflow as tf
import numpy as np

class my_fc6(tf.keras.layers.Layer):
    def __init__(self):
        super(my_fc6, self).__init__()
        self.fc = tf.keras.layers.Dense(4096)
        self.relu = tf.keras.layers.ReLU()

    def call(self, x):
        x = self.fc(x)
        x = self.relu(x)
        return x

    def load_weights(self, weights_path):
        """Load the weights of the Dense layer from a file."""
        dense_weights = np.load(weights_path)['weights']
        self.fc.set_weights(dense_weights)
        print(f"Dense layer weights loaded from {weights_path}")

def save_fc6_model_weights(model, save_path):
    """Save the weights of the Dense layer to a file."""
    dense_weights = model.fc.get_weights()
    np.savez(save_path, weights=dense_weights)
    print(f"Dense layer weights saved to {save_path}")