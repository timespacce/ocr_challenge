import tensorflow as tf


class OCRModel(tf.keras.Model):
    vocab_size = None
    filter_count = None

    ##
    def __init__(self, label_length, vocab_size, filter_size, num_layers, hidden_size):
        super(OCRModel, self).__init__()
        self.label_length = label_length
        self.vocab_size = vocab_size
        self.filter_size = filter_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        ##
        self.conv1 = tf.keras.layers.Conv2D(self.filter_size, kernel_size=(4, 4), strides=(1, 1), padding='same')
        self.conv_block_1 = self.convolution_block()
        self.conv_block_2 = self.convolution_block()
        self.conv_block_3 = self.convolution_block()
        ##
        self.conv2 = tf.keras.layers.Conv2D(self.label_length, kernel_size=(4, 4), strides=(1, 1), padding='same')  # (B, X, Y, LABEL)
        ##
        self.dense = tf.keras.layers.Dense(self.hidden_size)  # (B, LABEL, HIDDEN_SIZE)
        self.classification = tf.keras.layers.Dense(self.vocab_size)  # (B, LABEL, VOCAB_SIZE)
        ##
        self.sf = tf.keras.layers.Softmax(axis=-1)

    def convolution_block(self):
        seq = tf.keras.Sequential([
            tf.keras.layers.Conv2D(self.filter_size, kernel_size=(4, 4), strides=(1, 1), padding='same'),
            tf.keras.layers.Conv2D(self.filter_size, kernel_size=(4, 4), strides=(1, 1), padding='same'),
            tf.keras.layers.Conv2D(self.filter_size, kernel_size=(2, 2), strides=(1, 1), padding='valid'),
            tf.keras.layers.BatchNormalization(axis=-1),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dropout(rate=1e-1)
        ])
        return seq

    def call(self, inputs, **kwargs):
        ffn = inputs
        batch_size, _, _, _ = ffn.shape
        ##
        ffn = self.conv1(ffn)
        ffn = self.conv_block_1(ffn)
        ffn = self.conv_block_2(ffn)
        ffn = self.conv_block_3(ffn)
        ##
        ffn = self.conv2(ffn)  # (B, X, Y, LABEL)
        ffn = tf.transpose(ffn, perm=[0, 3, 1, 2])  # (B, LABEL, X, Y)
        ffn = tf.reshape(ffn, shape=(batch_size, self.label_length, -1))  # (B, LABEL, X * Y)
        ##
        ffn = self.dense(ffn)  # (B, LABEL, HIDDEN_SIZE)
        ffn = self.classification(ffn)  # (B, LABEL, VOCAB_SIZE)
        ffn = self.sf(ffn)
        ##
        return ffn
