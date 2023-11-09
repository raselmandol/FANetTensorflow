import os
import time
import datetime
import random
import numpy as np
import tensorflow as tf
from glob import glob
import albumentations as A
import cv2
from tensorflow.keras import layers
from tensorflow.data import Dataset
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import MeanIoU
from tensorflow.image import resize
from utils import (
    seeding, shuffling, create_dir, init_mask,
    epoch_time, rle_encode, rle_decode, print_and_save, load_data
)
from model import FANet
from loss import DiceBCELoss

class DATASET:
    def __init__(self, images_path, masks_path, size, transform=None):
        self.images_path = images_path
        self.masks_path = masks_path
        self.transform = transform
        self.n_samples = len(images_path)
        self.size = size

    def generator(self):
        for index in range(self.n_samples):
            image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
            mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)

            if self.transform is not None:
                augmentations = self.transform(image=image, mask=mask)
                image = augmentations["image"]
                mask = augmentations["mask"]

            image = resize(image, self.size, method='bilinear')
            image = tf.transpose(image, (2, 0, 1))
            image = image / 255.0
            image = tf.convert_to_tensor(image, dtype=tf.float32)

            mask = resize(mask, self.size, method='bilinear')
            mask = tf.expand_dims(mask, axis=0)
            mask = mask / 255.0
            mask = tf.convert_to_tensor(mask, dtype=tf.float32)

            yield image, mask

def train(model, loader, mask, optimizer, loss_fn, device):
    epoch_loss = 0
    return_mask = []

    @tf.function
    def rle_decode_tensor(data, size):
        # Implement your RLE decode logic using TensorFlow operations
        # ...
        return decoded_data

    model.train()
    for i, (x, y) in enumerate(loader.generator()):
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        y = tf.convert_to_tensor(y, dtype=tf.float32)

        b, c, h, w = y.shape
        m = []

        for edata in mask[i * b : i * b + b]:
            edata = " ".join(str(d) for d in edata)
            edata = tf.strings.as_string(edata)
            edata = rle_decode_tensor(edata, size)
            edata = tf.expand_dims(edata, axis=0)
            m.append(edata)

        m = tf.convert_to_tensor(m, dtype=tf.float32)

        optimizer.zero_grad()
        y_pred = model([x, m])
        loss = loss_fn(y_pred, y)
        optimizer.apply_gradients(zip(optimizer.compute_gradients(loss, model.trainable_variables), model.trainable_variables))

        y_pred = tf.math.sigmoid(y_pred)
        y_pred = y_pred.numpy()

        for py in y_pred:
            py = tf.squeeze(py, axis=0)
            py = py > 0.5
            py = tf.dtypes.cast(py, tf.uint8)
            py = rle_encode(py)
            return_mask.append(py)

        epoch_loss += loss.numpy()

    epoch_loss = epoch_loss / len(loader)
    return epoch_loss, return_mask

# Rest of the code remains the same
