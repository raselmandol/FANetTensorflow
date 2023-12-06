import tensorflow as tf
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras import backend as K

class DiceLoss(tf.keras.losses.Loss):
    def __init__(self, name='dice_loss', smooth=1):
        super(DiceLoss, self).__init__(name=name)
        self.smooth = smooth

    def call(self, y_true, y_pred):
        y_pred = K.sigmoid(y_pred)

        y_true = K.flatten(y_true)
        y_pred = K.flatten(y_pred)

        intersection = K.sum(y_true * y_pred)
        dice = (2. * intersection + self.smooth) / (K.sum(y_true) + K.sum(y_pred) + self.smooth)

        return 1 - dice


class DiceBCELoss(tf.keras.losses.Loss):
    def __init__(self, name='dice_bce_loss', smooth=1):
        super(DiceBCELoss, self).__init__(name=name)
        self.smooth = smooth

    def call(self, y_true, y_pred):
        y_pred = K.sigmoid(y_pred)

        y_true = K.flatten(y_true)
        y_pred = K.flatten(y_pred)

        intersection = K.sum(y_true * y_pred)
        dice_loss = 1 - (2. * intersection + self.smooth) / (K.sum(y_true) + K.sum(y_pred) + self.smooth)
        BCE = binary_crossentropy(y_true, y_pred)
        Dice_BCE = (0.5 * BCE) + (0.5 * dice_loss)

        return Dice_BCE
