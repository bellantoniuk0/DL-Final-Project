import tensorflow as tf

class ScoreRegressor(tf.keras.Model):
    def __init__(self):
        super(ScoreRegressor, self).__init__()
        self.fc_final_score = tf.keras.layers.Dense(1)

    def call(self, x):
        final_score = self.fc_final_score(x)
        return final_score