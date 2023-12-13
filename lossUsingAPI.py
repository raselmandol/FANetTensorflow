import tensorflow as tf
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras import backend as K

def dice_coefficient(y_true, y_pred, smooth=1):
    y_pred = K.sigmoid(y_pred)
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred) #y_pred
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return dice

def dice_loss(y_true, y_pred, smooth=1):
    return 1 - dice_coefficient(y_true, y_pred, smooth)

def dice_bce_loss(y_true, y_pred, smooth=1):
    y_pred = K.sigmoid(y_pred)
    dice = dice_coefficient(y_true, y_pred, smooth)
    BCE = binary_crossentropy(y_true, y_pred)
    dice_bce = (0.5 * BCE) + (0.5 * (1 - dice))
    return dice_bce #dice
