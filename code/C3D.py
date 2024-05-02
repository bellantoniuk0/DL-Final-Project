import tensorflow as tf

class C3D(tf.keras.Model):

    def __init__(self):
        super(C3D, self).__init__()

        self.conv1 = tf.keras.layers.Conv3D(64, kernel_size=(3, 3, 3), padding='same', activation='relu')
        self.pool1 = tf.keras.layers.MaxPool3D(pool_size=(1, 2, 2), strides=(1, 2, 2), padding='valid')

        self.conv2 = tf.keras.layers.Conv3D(128, kernel_size=(3, 3, 3), padding='same', activation='relu')
        self.pool2 = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid')

        self.conv3a = tf.keras.layers.Conv3D(256, kernel_size=(3, 3, 3), padding='same', activation='relu')
        self.conv3b = tf.keras.layers.Conv3D(256, kernel_size=(3, 3, 3), padding='same', activation='relu')
        self.pool3 = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid')

        self.conv4a = tf.keras.layers.Conv3D(512, kernel_size=(3, 3, 3), padding='same', activation='relu')
        self.conv4b = tf.keras.layers.Conv3D(512, kernel_size=(3, 3, 3), padding='same', activation='relu')
        self.pool4 = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid')

        self.conv5a = tf.keras.layers.Conv3D(512, kernel_size=(3, 3, 3), padding='same', activation='relu')
        self.conv5b = tf.keras.layers.Conv3D(512, kernel_size=(3, 3, 3), padding='same', activation='relu')
        self.pool5 = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid')

        self.flatten = tf.keras.layers.Flatten()
        self.fc6 = tf.keras.layers.Dense(4096, activation='relu')
        self.fc7 = tf.keras.layers.Dense(4096, activation='relu')
        self.fc8 = tf.keras.layers.Dense(487, activation=None)

        self.dropout = tf.keras.layers.Dropout(0.5)
        self.softmax = tf.keras.layers.Softmax()

    def call(self, x):
        x = x * 255
        h = self.conv1(x)
        h = self.pool1(h)

        h = self.conv2(h)
        h = self.pool2(h)

        h = self.conv3a(h)
        h = self.conv3b(h)
        h = self.pool3(h)

        h = self.conv4a(h)
        h = self.conv4b(h)
        h = self.pool4(h)

        h = self.conv5a(h)
        h = self.conv5b(h)
        h = self.pool5(h)

        h = self.flatten(h)
        h = self.fc6(h)
        h = self.dropout(h)
        h = self.fc7(h)
        h = self.dropout(h)

        logits = self.fc8(h)
        probs = self.softmax(logits)

        return probs

def save_c3d_model_weights(model, save_path):
    """Save the C3D model weights to a file."""
    model.save_weights(save_path, save_format="tf")
    # print(f"Model weights saved to {save_path}")
   
