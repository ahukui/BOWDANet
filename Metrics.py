#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 18:37:33 2019

@author: labadmin
"""

from keras import backend as K
import tensorflow as tf 


smooth = 1.0

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)


def Edge_Extracted(y_pred):
    #Edge extracted by sobel filter  
    min_x = tf.constant(0, tf.float32)
    max_x = tf.constant(1, tf.float32)
    sobel_x = tf.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], tf.float32)
#    sobel_x /= factor 
    sobel_x_filter = tf.reshape(sobel_x, [1,3, 3, 1, 1])      
    sobel_y_filter = tf.transpose(sobel_x_filter, [0, 2, 1, 3, 4])
    
    filters_x = tf.nn.conv3d(y_pred , sobel_x_filter,
                              strides=[1, 1, 1, 1,1],data_format="NCDHW", padding='SAME')
    
    filters_y = tf.nn.conv3d(y_pred , sobel_y_filter,
                              strides=[1, 1, 1, 1, 1],data_format="NCDHW", padding='SAME')

    edge = tf.sqrt(filters_x * filters_x + filters_y * filters_y+1e-16)

    edge = tf.clip_by_value(edge, min_x, max_x)    
    
    return edge

def Dist_Loss(y_true, y_pred):
    edge = Edge_Extracted(y_pred)
    edge = K.flatten(edge)
    y_true_f = K.flatten(y_true)    
    edge_loss = K.sum(edge * y_true_f)
    return edge_loss