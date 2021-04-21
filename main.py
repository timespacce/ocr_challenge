import string
import time
import zipfile

import numpy as np
import tensorflow as tf
from model import OCRModel
import os
import matplotlib.pyplot as plt
from PIL import Image

##

model: tf.keras.Model
cat_cross_ent: tf.keras.losses.Loss
optimizer: tf.keras.optimizers.Optimizer

##

buffer_size = 34318
batch_size = 512
label_length = 6
vocab_size = 37
epochs = 25
steps = None
alpha = 1e-3

vocabulary = []
token_to_id = {}
id_to_token = {}


##


def run():
    ##
    tf.config.experimental_run_functions_eagerly(True)
    tf.executing_eagerly()
    ##
    build_model()
    train_model()
    return


def load_data(target):
    global buffer_size, steps, vocabulary, vocab_size, label_length, token_to_id, id_to_token
    ##
    zip_file = zipfile.ZipFile(target, "r")
    namelist = zip_file.namelist()
    train_count, test_count, label_length = 0, 0, 0
    h, w = 128, 64
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
    buffer_size = int(buffer_size // batch_size) * batch_size
    steps = buffer_size // batch_size
    train_count = min(buffer_size, train_count)
    test_count = min(buffer_size, test_count)
    train_inputs, train_labels = np.zeros((train_count, h, w, 3), dtype=np.float32), np.zeros(
        (train_count, label_length), dtype=np.int32)
    test_inputs, test_labels = np.zeros((test_count, h, w, 3), dtype=np.float32), np.zeros((test_count, label_length), dtype=np.int32)
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
        if train_count == buffer_size and train:
            continue
        if test_count == buffer_size and not train:
            continue
        ##
        label, img_id = file.split("/")[-1].split("_")
        image_file = zip_file.open(file)
        image = Image.open(image_file)
        image_resized = image.resize((w, h))
        image_normalized = np.array(image_resized) / 255
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
    global buffer_size, batch_size
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
    tf_data_set = tf_data_set.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True)
    tf_data_set = tf_data_set.batch(batch_size=batch_size)
    return tf_data_set


def build_model():
    global vocab_size, label_length, model, cat_cross_ent, optimizer, alpha
    ##
    model = OCRModel(label_length=label_length, vocab_size=vocab_size)
    cat_cross_ent = tf.keras.losses.CategoricalCrossentropy(reduction='none')
    optimizer = tf.keras.optimizers.Adam(learning_rate=alpha, beta_1=0.9, beta_2=0.98, epsilon=1e-6)
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
    global model, epochs, steps, id_to_token
    ##
    (train_inputs, train_labels), (test_inputs, test_labels) = load_data("data/synthetic_dataset.zip")

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

    ##
    for e in range(epochs):
        acc_l1, batch = 0, 0
        sub_data_set = load_and_shuffle(xs=train_inputs, ys=train_labels)
        ##
        for x, y in sub_data_set:
            l1 = train_step(x, y)
            acc_l1, batch = acc_l1 + l1, batch + 1
            print("\r", end="")
            print("E = {} : {:.1f}% ".format(e, (batch / steps) * 1e2), end="", flush=True)
        avg_l1 = acc_l1 / batch
        print(" L1={1:3f}".format(e, avg_l1))

    ##

    def run_inference(inputs, labels, mode):
        data_set = load_and_shuffle(xs=inputs, ys=labels)
        count, all_correct = 0, 0
        validations = []
        row_format = "{0} : {1} : {2:.3f}\n"
        for x, y in data_set:
            y_hat = inference_step(x)
            y_hat_labels = tf.argmax(y_hat, axis=-1)
            ##
            for x_i, y_i in zip(y_hat_labels, y):
                x_i, y_i = x_i.numpy(), y_i.numpy()
                diff = tf.abs(tf.cast(y_i, tf.int64) - x_i)
                diff_sum = tf.reduce_sum(diff).numpy()
                correct = int(diff_sum == 0)
                count, all_correct = count + 1, all_correct + correct
                validations.append((x_i, y_i, diff_sum))
        ##
        s = open("validation_{}.txt".format(mode), 'w')
        validations.sort(key=lambda x: x[2])
        for x_i, y_i, acc in validations:
            row = row_format.format(x_i, y_i, acc)
            s.write(row)
        s.close()
        percent = (all_correct / count) * 1e2
        print("{} : {} / {} or {:.3f} %".format(mode, all_correct, count, percent))

    ##
    run_inference(train_inputs, train_labels, "TRAIN")
    run_inference(test_inputs, test_labels, "TEST")

    return


if __name__ == "__main__":
    begin = time.time()
    run()
    runtime = time.time() - begin
    print("OCR in {} s".format(runtime))
