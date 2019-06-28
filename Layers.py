# -*- coding: utf-8 -*-
"""
@author: kui
"""
from keras.layers import merge,Conv3D,ZeroPadding3D,Input,BatchNormalization,Activation,MaxPooling3D,UpSampling3D,Deconv3D
from keras.regularizers import l2
from keras import backend as K
from keras.layers.core import Dropout
from keras.layers.pooling import AveragePooling3D
from keras.layers.merge import concatenate,add
from keras.layers.advanced_activations import LeakyReLU

def __conv_block(ip, nb_filter, bottleneck, dropout_rate, weight_decay,train_flage,name_flage):
    ''' Apply BatchNorm, Relu, 3x3 Conv2D, optional bottleneck block and dropout
    Args:
        ip: Input keras tensor
        nb_filter: number of filters
        bottleneck: add bottleneck block
        dropout_rate: dropout rate
        weight_decay: weight decay factor
        ......
    Returns: keras tensor with batch_norm, relu and convolution2d added (optional bottleneck)
    '''
    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = BatchNormalization(axis=concat_axis, epsilon=0,trainable=train_flage,name=name_flage+'CBN')(ip)
    x = Activation('relu')(x)


    if bottleneck:
        inter_channel = nb_filter*4  # Obtained from https://github.com/liuzhuang13/DenseNet/blob/master/densenet.lua

        x = Conv3D(inter_channel, (1,1,1), kernel_initializer='he_normal', padding='same', use_bias=False,
                   kernel_regularizer=l2(weight_decay),trainable=train_flage,name=name_flage)(x)
        x = BatchNormalization(axis=concat_axis,trainable=train_flage,name=name_flage+'BN')(x)
        x = Activation('relu')(x)

    x = Conv3D(nb_filter, (3,3, 3), kernel_initializer='he_normal', padding='same', use_bias=False, 
               kernel_regularizer=l2(weight_decay),trainable=train_flage,name=name_flage+'_DB')(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x


def __dense_block(x, nb_layers, nb_filter, growth_rate, bottleneck, dropout_rate, weight_decay,train_flage,name_flage,
                  grow_nb_filters=True, return_concat_list=False):
    ''' Build a dense_block where the output of each conv_block is fed to subsequent ones
    Args:
        x: keras tensor
        nb_layers: the number of layers of conv_block to append to the model.
        nb_filter: number of filters
        growth_rate: growth rate
        bottleneck: bottleneck block
        dropout_rate: dropout rate
        weight_decay: weight decay factor
        grow_nb_filters: flag to decide to allow number of filters to grow
        return_concat_list: return the list of feature maps along with the actual output
    Returns: keras tensor with nb_layers of conv_block appended
    '''
    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x_list = [x]

    for i in range(nb_layers):
        cb = __conv_block(x, growth_rate, bottleneck, dropout_rate, weight_decay,train_flage,name_flage+'Conv_'+str(i))
        x_list.append(cb)

        x = concatenate([x, cb], axis=concat_axis)

        if grow_nb_filters:
            nb_filter += growth_rate

    if return_concat_list:
        return x, nb_filter, x_list
    else:
        return x, nb_filter


def transition_block(ip, nb_filter,weight_decay,train_flage,name_flage):
    ''' Apply BatchNorm, Relu 1x1, Conv2D, optional compression, dropout and Maxpooling2D
    Args:
        ip: keras tensor
        nb_filter: number of filters
        compression: calculated as 1 - reduction. Reduces the number of feature maps
                    in the transition block.
        dropout_rate: dropout rate
        weight_decay: weight decay factor
    Returns: keras tensor, after applying batch_norm, relu-conv, dropout, maxpool
    '''
    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = BatchNormalization(axis=concat_axis,trainable=train_flage,name=name_flage+'BN')(ip)
    x = Activation('relu')(x)
    x = Conv3D(nb_filter, (1,1,1), kernel_initializer='he_normal', padding='same', use_bias=False,
               kernel_regularizer=l2(weight_decay),trainable=train_flage,name=name_flage+'Conv')(x)
#    x = AveragePooling3D((2,2,2), strides=(2,2,2))(x)
    return x

       
def down_stage(ip,nb_layers,nb_filter,growth_rate,dropout_rate,weight_decay,compression,train_flage,name_flage,pooling=True):
    
    x0, nb_filter = __dense_block(ip,nb_layers , nb_filter, growth_rate,bottleneck=True,dropout_rate=dropout_rate,
                                     weight_decay=weight_decay,train_flage=train_flage,name_flage=name_flage+'_DB_')
    x1=transition_block(x0, nb_filter,weight_decay=weight_decay,train_flage=train_flage,name_flage=name_flage+'_TB0_')
    x =transition_block(ip, nb_filter,weight_decay=weight_decay,train_flage=train_flage,name_flage=name_flage+'_TB1_')
    addx=add([x,x1],name=name_flage+'ADD')
    if pooling:    
        out= AveragePooling3D(strides=(2,2,2))(addx)
        return addx,out
    return addx


def up_stage(ip,nb_layers,nb_filter,growth_rate,dropout_rate,weight_decay,compression,train_flage,name_flage):
    
    x0, nb_filter = __dense_block(ip,nb_layers , nb_filter, growth_rate,bottleneck=True,dropout_rate=dropout_rate,
                                     weight_decay=weight_decay,train_flage=train_flage,name_flage=name_flage+'_DB_')
    x1 =transition_block(x0, nb_filter,weight_decay=weight_decay,train_flage=train_flage,name_flage=name_flage+'_TB0_')
    addx=add([ip,x1],name=name_flage+'ADD')                                 
    return addx
 
 
def Decon_stage(x,nb_filters,kernel_size,strides,weight_decay,train_flage,name_flage):
    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1
    x = BatchNormalization(axis=concat_axis,trainable=train_flage,name=name_flage+'BN')(x)
    x = Activation('relu')(x)
    x = Deconv3D(nb_filters,kernel_size,strides=strides, activation='relu', padding='same',data_format='channels_first',
                        kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay),trainable=train_flage,name=name_flage)(x)
#    x = UpSampling3D()(x)    
    return x



def output(x,weight_decay,train_flage,name_flage):
    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1
    x = BatchNormalization(axis=concat_axis,trainable=train_flage,name=name_flage+'BN')(x)
    x = Activation('relu')(x)
    x = Conv3D(1, (1,1, 1), kernel_initializer='he_normal', padding='same', use_bias=False,
               kernel_regularizer=l2(weight_decay),trainable=train_flage,name=name_flage)(x)
    x = Activation('sigmoid')(x)
    return x


def side_out(x,up_size,weight_decay,train_flage,name_flage):
    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1
    x = BatchNormalization(axis=concat_axis,trainable=train_flage,name=name_flage+'BN')(x)
    x = Activation('relu')(x)
    x = Conv3D(1, (1,1, 1), kernel_initializer='he_normal', padding='same', use_bias=False,
               kernel_regularizer=l2(weight_decay),trainable=train_flage,name=name_flage)(x)
    x = Activation('sigmoid')(x)
    up1=UpSampling3D((up_size,up_size,up_size))(x)
    return up1


def DisOutput(x,weight_decay=5e-4):
    x = Conv3D(1, (1,1,1), kernel_initializer='he_normal', padding='same', use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)
#    x = BatchNormalization(axis=concat_axis)(x)               
    x = Activation('sigmoid',name = 'DisOutput')(x)
    return x    


def conv_block(ip, nb_filter,dropout_rate=0.3, weight_decay=5e-4):
    ''' Apply BatchNorm, Relu, 3x3 Conv2D, optional bottleneck block and dropout
    Args:
        ip: Input keras tensor
        nb_filter: number of filters
        bottleneck: add bottleneck block
        dropout_rate: dropout rate
        weight_decay: weight decay factor
    Returns: keras tensor with batch_norm, relu and convolution2d added (optional bottleneck)
    '''
    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = Conv3D(nb_filter, (3,3,3), strides=1, kernel_initializer='he_normal', padding='same', use_bias=False, kernel_regularizer=l2(weight_decay))(ip)
    x = BatchNormalization(axis=concat_axis)(x)
    x = LeakyReLU(alpha=0.2)(x)      
    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x


def FinalOut(ip, nb_filter=1, weight_decay=5e-4):
    ''' Apply BatchNorm, Relu, 3x3 Conv2D, optional bottleneck block and dropout
    Args:
        ip: Input keras tensor
        nb_filter: number of filters
        bottleneck: add bottleneck block
        dropout_rate: dropout rate
        weight_decay: weight decay factor
    Returns: keras tensor with batch_norm, relu and convolution2d added (optional bottleneck)
    '''
    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = Conv3D(nb_filter, (4,4,4), kernel_initializer='he_normal', padding='same', use_bias=False, kernel_regularizer=l2(weight_decay))(ip)
    x = BatchNormalization(axis=concat_axis)(x)
    x = Activation('sigmoid')(x)

    return x