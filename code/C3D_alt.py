import tensorflow as tf

class C3D_altered(tf.keras.Model):

    def __init__(self, num_classes=299):
        super(C3D_altered, self).__init__()

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
        #self.relu = tf.keras.layers.ReLU()
        self.fc = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, x):
        h = self.conv1(x)
        # h = self.relu(h)
        h = self.pool1(h)

        h = self.conv2(h)
        # h = self.relu(h)
        h = self.pool2(h)

        h = self.conv3a(h)
        # h = self.relu(h)
        h = self.conv3b(h)
        # h = self.relu(h)
        h = self.pool3(h)

        h = self.conv4a(h)
        # h = self.relu(h)
        h = self.conv4b(h)
        # h = self.relu(h)
        h = self.pool4(h)

        h = self.conv5a(h)
        # h = self.relu(h)
        h = self.conv5b(h)
        # h = self.relu(h)
        h = self.pool5(h)

        h = self.flatten(h)
        h = self.fc(h)

        return h

def save_c3d_alt_model_weights(model, save_path):
    """Save the C3D model weights to a file."""
    model.save_weights(save_path, save_format="tf")
    # print(f"Model weights saved to {save_path}")