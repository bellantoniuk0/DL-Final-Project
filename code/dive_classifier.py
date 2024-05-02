import tensorflow as tf

class DiveClassifier(tf.keras.Model):
    def __init__(self):
        super(DiveClassifier, self).__init__()
        self.fc_position = tf.keras.layers.Dense(3)
        self.fc_armstand = tf.keras.layers.Dense(2)
        self.fc_rot_type = tf.keras.layers.Dense(4)
        self.fc_ss_no = tf.keras.layers.Dense(10)
        self.fc_tw_no = tf.keras.layers.Dense(8)

    def call(self, x):
        position = self.fc_position(x)
        armstand = self.fc_armstand(x)
        rot_type = self.fc_rot_type(x)
        ss_no = self.fc_ss_no(x)
        tw_no = self.fc_tw_no(x)
        return position, armstand, rot_type, ss_no, tw_no

# Create an instance of the TensorFlow model
tf_model = DiveClassifier()

def save_dive_classifier_model_weights(model, save_path):
    """Save the C3D model weights to a file."""
    model.save_weights(save_path, save_format="tf")
    print(f"Model weights saved to {save_path}")


