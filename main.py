import string
import sys
import time
import zipfile

import numpy as np
import tensorflow as tf
from model import OCRModel
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

##

model: tf.keras.Model
cat_cross_ent: tf.keras.losses.Loss
optimizer: tf.keras.optimizers.Optimizer
checkpoint: tf.train.Checkpoint
checkpoint_manager: tf.train.CheckpointManager

##
data_dir = None
checkpoint_dir = None
##
h, w = 64, 128
train_buffer_size = 34800
test_buffer_size = 5509
blocks = 4
batch_size = 128
label_length = 6
vocab_size = 37
epochs = 0
steps = None
alpha = 1e-4
eager = False

vocabulary = []
token_to_id = {}
id_to_token = {}


##


def run():
    global data_dir, checkpoint_dir, eager
    ##
    data_dir, checkpoint_dir = sys.argv[1], sys.argv[2]
    ##
    if eager:
        tf.config.run_functions_eagerly(True)
        tf.executing_eagerly()
    ##
    build_model()
    train_model()
    return


def load_data():
    global data_dir, train_buffer_size, test_buffer_size, steps, vocabulary, vocab_size, label_length, token_to_id, id_to_token
    ##
    zip_file = zipfile.ZipFile(data_dir, "r")
    namelist = zip_file.namelist()
    train_count, test_count, label_length = 0, 0, 0
    for file in namelist:
        if not file.endswith('.png'):
            continue
        ##
        label, img_id = file.split("/")[-1].split("_")
        label_length = len(label)
        ##
        if "train" in file:
            train_count += 1
        else:
            test_count += 1
    ##
    train_buffer_size = min(train_buffer_size, train_count)
    test_buffer_size = min(test_buffer_size, test_count)
    train_buffer_size = int(train_buffer_size // batch_size) * batch_size
    test_buffer_size = int(test_buffer_size // batch_size) * batch_size
    steps = train_buffer_size // batch_size
    train_inputs, train_labels = np.zeros((train_buffer_size, h, w, 3), dtype=np.float32), np.zeros((train_buffer_size, label_length), dtype=np.int32)
    test_inputs, test_labels = np.zeros((test_buffer_size, h, w, 3), dtype=np.float32), np.zeros((test_buffer_size, label_length), dtype=np.int32)
    ##
    vocabulary = list(string.ascii_uppercase) + list(map(str, range(0, 10))) + ["-"]
    for i, token in enumerate(vocabulary):
        token_to_id[token] = i
        id_to_token[i] = token
    vocab_size = len(vocabulary)
    ##
    train_count, test_count = 0, 0
    for i, file in enumerate(namelist):
        if not file.endswith('.png'):
            continue
        train = "train" in file
        ##
        if train_count == train_buffer_size and train:
            continue
        if test_count == test_buffer_size and not train:
            continue
        ##
        label, img_id = file.split("/")[-1].split("_")
        image_file = zip_file.open(file)
        image = Image.open(image_file)
        image_resized = image.resize((w, h))
        image_normalized = np.array(image_resized) / 255
        image_normalized = image_normalized.reshape((h, w, 3))
        label_tokenized = [token_to_id[i] for i in label]
        if train:
            train_inputs[train_count] = image_normalized
            train_labels[train_count] = label_tokenized
            train_count += 1
        else:
            test_inputs[test_count] = image_normalized
            test_labels[test_count] = label_tokenized
            test_count += 1
        print("\r", end="")
        print("{:>3.1f}% IMAGES Loaded".format((i / len(namelist)) * 1e2), end="", flush=True)
    ##
    print("")
    print("TRAIN = {} AND TEST = {}".format(train_inputs.shape, test_inputs.shape))
    ##
    return (train_inputs, train_labels), (test_inputs, test_labels)


def load_and_shuffle(xs, ys):
    global batch_size
    ##
    count, _, _, _ = xs.shape
    indices = np.arange(count)
    np.random.shuffle(indices)
    xs = xs[indices]
    ys = ys[indices]
    ##
    tf_xs = tf.data.Dataset.from_tensor_slices(xs)
    tf_ys = tf.data.Dataset.from_tensor_slices(ys)
    tf_data_set = tf.data.Dataset.zip((tf_xs, tf_ys))
    tf_data_set = tf_data_set.shuffle(buffer_size=count, reshuffle_each_iteration=True)
    tf_data_set = tf_data_set.batch(batch_size=batch_size)
    return tf_data_set


def build_model():
    global checkpoint_dir, vocab_size, label_length, model, cat_cross_ent, optimizer, batch_size, alpha, checkpoint, checkpoint_manager
    ##
    model = OCRModel(label_length=label_length, vocab_size=vocab_size, filter_size=32, num_layers=3, hidden_size=512)
    model.build(input_shape=(batch_size, h, w, 3))
    model.summary()
    ##
    cat_cross_ent = tf.keras.losses.CategoricalCrossentropy(reduction='none')
    optimizer = tf.keras.optimizers.Adam(learning_rate=alpha)
    ##
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_dir, max_to_keep=5)
    if checkpoint_manager.latest_checkpoint:
        selected_checkpoint = checkpoint_manager.checkpoints[1]
        status = checkpoint.restore(selected_checkpoint)
        status.assert_consumed()
        print("Restored from {}".format(selected_checkpoint))
    else:
        print("Initializing from scratch.")
    ##
    return


def loss_function(y_hat, y):
    global cat_cross_ent, vocab_size
    ##
    y_label = tf.one_hot(y, depth=vocab_size, axis=-1)
    pro_sample = tf.reduce_sum(cat_cross_ent(y_label, y_hat), axis=-1)
    err = tf.reduce_mean(pro_sample)
    return err


def train_model():
    global model, epochs, blocks, batch_size, steps, id_to_token, checkpoint, checkpoint_manager
    ##
    (train_inputs, train_labels), (test_inputs, test_labels) = load_data()

    ##
    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            y_hat = model(x)
            loss = loss_function(y_hat, y)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    ##
    @tf.function
    def inference_step(x):
        y_hat = model(x)
        return y_hat

    def run_inference(inputs, labels, mode):
        count, _, _, _ = inputs.shape
        index, all_correct = 0, 0
        validations = []
        row_format = "{0} : {1} : {2:.3f}\n"
        block_size = count // blocks
        for i in range(1, blocks + 1):
            data_set = load_and_shuffle(xs=inputs[(i - 1) * block_size:i * block_size],
                                        ys=labels[(i - 1) * block_size:i * block_size])
            for x, y in data_set:
                y_hat = inference_step(x)
                y_hat_labels = tf.argmax(y_hat, axis=-1)
                for x_i, y_i in zip(y_hat_labels, y):
                    x_i, y_i = x_i.numpy(), y_i.numpy()
                    diff = tf.abs(tf.cast(y_i, tf.int64) - x_i)
                    diff_sum = tf.reduce_sum(diff).numpy()
                    index = index + 1
                    x_i, y_i = [id_to_token[j] for j in x_i], [id_to_token[j] for j in y_i]
                    validations.append((x_i, y_i, diff_sum))
                ##
                print("\r", end="")
                print("{:>5} INFERENCE : {:>5} / {:>5} ".format(mode, index, count), end="", flush=True)
        ##
        s = open("validation_{}.txt".format(mode), 'w')
        validations.sort(key=lambda x: x[2])
        for x_i, y_i, acc in validations:
            row = row_format.format(''.join(x_i), ''.join(y_i), acc)
            s.write(row)
        s.close()
        ##
        validations = np.array(validations, dtype=np.object)
        recall_1 = (validations[:, 2] == 0).sum()
        recall_2 = recall_1 + (validations[:, 2] == 1).sum()
        recall_3 = recall_2 + (validations[:, 2] == 2).sum()
        ##
        percent_1, percent_2, percent_3 = (recall_1 / index) * 1e2, (recall_2 / index) * 1e2, (recall_3 / index) * 1e2
        print("{:>5} : samples = {:>5} / {:>5} : correct = {:.1f} % recall@1 = {:.1f} % recall@2 = {:.1f} %".format(mode, recall_1, index,
                                                                                                                    percent_1, percent_2, percent_3))

    ##
    for e in range(epochs):
        acc_l1, batch = 0, 0
        block_size = train_buffer_size // blocks
        for i in range(1, blocks + 1):
            sub_data_set = load_and_shuffle(xs=train_inputs[(i - 1) * block_size:i * block_size],
                                            ys=train_labels[(i - 1) * block_size:i * block_size])
            ##
            for x, y in sub_data_set:
                l1 = train_step(x, y)
                acc_l1, batch = acc_l1 + l1, batch + 1
                avg_l1 = acc_l1 / batch
                print("\r", end="")
                print("E = {} : {:.1f}% : {:.1f} ".format(e, (batch / steps) * 1e2, avg_l1), end="", flush=True)
        avg_l1 = acc_l1 / batch
        print(" L1={1:3f}".format(e, avg_l1))
        ##
        if (e + 1) % 5 == 0:
            saved_checkpoint = checkpoint_manager.save()
            print("checkpoint saved at {}".format(saved_checkpoint))
        ##
        if (e + 1) % 5 == 0:
            run_inference(test_inputs, test_labels, "TEST")

    ##
    run_inference(train_inputs, train_labels, "TRAIN")
    run_inference(test_inputs, test_labels, "TEST")

    return


if __name__ == "__main__":
    begin = time.time()
    run()
    runtime = time.time() - begin
    print("OCR in {} s".format(runtime))
