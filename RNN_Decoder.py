import tensorflow as tf
from BahdanauAttention import BahdanauAttention


class RNN_Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size):
        super(RNN_Decoder, self).__init__()
        self.units = units

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.lstm = tf.keras.layers.LSTM(
            self.units, return_state=True, return_sequences=True, name="encoder")

        self.fc1 = tf.keras.layers.Dense(self.units)
        self.fc2 = tf.keras.layers.Dense(vocab_size)

    def embed(self, x):
        x = self.embedding(x)

        return x

    def call(self, x, hidden):

        # passing the concatenated vector to the GRU
        output, state_h, state_c = self.lstm(x, initial_state=hidden)

        state = [state_h, state_c]

        # shape == (batch_size, max_length, hidden_size)
        x = self.fc1(output)

        # x shape == (batch_size * max_length, hidden_size)
        x = tf.reshape(x, (-1, x.shape[2]))

        # output shape == (batch_size * max_length, vocab)
        x = self.fc2(x)

        return x, state

    def reset_state(self, batch_size):
        return [tf.zeros((batch_size, self.units)), tf.zeros((batch_size, self.units))]
