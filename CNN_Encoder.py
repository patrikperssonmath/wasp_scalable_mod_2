import tensorflow as tf


class CNN_Encoder(tf.keras.Model):
    # Since you have already extracted the features and dumped it using pickle
    # This encoder passes those features through a Fully connected layer
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        # shape after fc == (batch_size, 64, embedding_dim)
        self.fc = tf.keras.layers.Dense(embedding_dim)
        self.flatten = tf.keras.layers.Flatten()

    def call(self, x):

        x = self.flatten(x)

        x = self.fc(x)
        x = tf.nn.elu(x)

        x = tf.expand_dims(x, 1)

        return x
