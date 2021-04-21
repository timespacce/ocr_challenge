import tensorflow as tf


class OCRModel(tf.keras.Model):
    vocab_size = None
    filter_count = None

    ##
    def __init__(self, label_length, vocab_size):
        super(OCRModel, self).__init__()
        self.label_length = label_length
        self.vocab_size = vocab_size
        self.filter_count = 32
        ##
        self.conv_1 = tf.keras.layers.Conv2D(self.filter_count, kernel_size=(2, 2), strides=(2, 2))
        self.bn_1 = tf.keras.layers.BatchNormalization(axis=-1)
        self.relu_1 = tf.keras.layers.LeakyReLU()
        self.conv_2 = tf.keras.layers.Conv2D(self.filter_count, kernel_size=(2, 2), strides=(2, 2))
        self.bn_2 = tf.keras.layers.BatchNormalization(axis=-1)
        self.relu_2 = tf.keras.layers.LeakyReLU()
        self.conv_3 = tf.keras.layers.Conv2D(self.filter_count, kernel_size=(2, 2), strides=(2, 2))
        self.bn_3 = tf.keras.layers.BatchNormalization(axis=-1)
        self.relu_3 = tf.keras.layers.LeakyReLU()
        ##
        self.conv4 = tf.keras.layers.Conv2D(self.label_length, kernel_size=(2, 2), padding='same')  # (B, X, Y, LABEL)
        ##
        self.dense = tf.keras.layers.Dense(128)  # (B, LABEL, HIDDEN_SIZE)
        self.classification = tf.keras.layers.Dense(self.vocab_size)  # (B, LABEL, VOCAB_SIZE)
        ##
        self.sf = tf.keras.layers.Softmax(axis=-1)

    def call(self, inputs, **kwargs):
        ffn = inputs
        batch_size, _, _, _ = ffn.shape
        ##
        ffn = self.conv_1(ffn)
        ffn = self.bn_1(ffn)
        ffn = self.relu_1(ffn)
        ffn = self.conv_2(ffn)
        ffn = self.bn_2(ffn)
        ffn = self.relu_2(ffn)
        ffn = self.conv_3(ffn)
        ffn = self.bn_3(ffn)
        ffn = self.relu_3(ffn)
        ##
        ffn = self.conv4(ffn)  # (B, X, Y, LABEL)
        ffn = tf.transpose(ffn, perm=[0, 3, 1, 2])  # (B, LABEL, X, Y)
        ffn = tf.reshape(ffn, shape=(batch_size, self.label_length, -1))  # (B, LABEL, X * Y)
        ##
        ffn = self.dense(ffn)
        ffn = self.classification(ffn)
        ffn = self.sf(ffn)
        ##
        return ffn
