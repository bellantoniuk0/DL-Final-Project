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

def save_fc6_model_weights(model, save_path):
    """Save the C3D model weights to a file."""
    model.save_weights(save_path, save_format="tf")
    print(f"Model weights saved to {save_path}")
   