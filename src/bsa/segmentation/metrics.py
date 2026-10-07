import tensorflow as tf


def iou(y_true, y_pred, smooth=1):
    intersection = tf.reduce_sum(tf.abs(y_true * y_pred))
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    iou_score = (intersection + smooth) / (union + smooth)
    return iou_score

def dice_coef(y_true, y_pred, smooth=1):
    intersection = tf.reduce_sum(y_true * y_pred)
    dice_score = (2.0 * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)
    return dice_score

def dice_loss(y_true, y_pred):
    loss = 1.0 - dice_coef(y_true, y_pred)
    return loss
