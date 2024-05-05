import tensorflow as tf

class ScoreRegressor(tf.keras.Model):
    def __init__(self, num_classes=300):
        super(ScoreRegressor, self).__init__()
        self.fc_final_score = tf.keras.layers.Dense(num_classes)

    def call(self, x):
        final_score = self.fc_final_score(x)
        return final_score

def save_score_reg_model_weights(model, save_path):
    """Save the C3D model weights to a file."""
    model.save_weights(save_path, save_format="tf")
    # print(f"Model weights saved to {save_path}")
