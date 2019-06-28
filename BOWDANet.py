# -*- coding: utf-8 -*-
"""
@author: kui
"""
from keras.layers import Conv3D,Input,BatchNormalization,Activation,UpSampling3D
from keras.models import Model
import numpy as np
from keras.optimizers import SGD,Adam
from keras import backend as K
from keras.layers.merge import add
from keras.layers.pooling import AveragePooling3D
import tensorflow as tf 
from Metrics import dice_coef, Dist_Loss
from Layers import * 
from keras.regularizers import l2
K.set_image_data_format('channels_first')

img_rows = 96
img_cols = 96
chan =16
same_num=4
def SNet(reduction=0.5, dropout_rate=0.3, weight_decay=5e-4):

    if  K.image_data_format() == 'channels_last':
      img_input = Input(shape=(img_rows,img_cols,chan,1))
      concat_axis=-1
    else:
      img_input = Input(shape=(1,chan, img_rows, img_cols))
      concat_axis=1


    if reduction != 0.0:
        assert reduction <= 1.0 and reduction > 0.0, 'reduction value must lie between 0.0 and 1.0'

    # compute compression factor
    compression = 1.0 - reduction
    nb_layers=[4,8,16,8,4,2]
    growth_rate=32

    #stage0    
    x1 = Conv3D(64, (3,3,3),strides=(1,1,1),kernel_initializer='he_normal',padding='same',
               use_bias=False, kernel_regularizer=l2(weight_decay),trainable=True,name='x1')(img_input)
    x = BatchNormalization(axis=concat_axis,trainable=True,name='x1_BN')(x1)
    x = Activation('relu')(x)
    x = AveragePooling3D((2,2,2))(x)

    #stage1
    s1_x0,s1_x = down_stage(x,nb_layers[0],0,growth_rate,dropout_rate,weight_decay,compression,train_flage=True,name_flage='s1')  
    
    #stage2
    s2_x0,s2_x = down_stage(s1_x,nb_layers[1],0,growth_rate,dropout_rate,weight_decay,compression,train_flage=True,name_flage='s2')    
 
    
    #stage3  
    s3_x0 = down_stage(s2_x,nb_layers[2],0,growth_rate,dropout_rate,weight_decay,compression,train_flage=True,name_flage='s3',pooling=False)
 
    #stage4
    D1 = Decon_stage(s3_x0,256,kernel_size=(3,3,3),strides=(2,2,2),weight_decay=weight_decay,train_flage=True,name_flage='D1')
    con1 = add([D1,s2_x0])     
    s4_x = up_stage(con1,nb_layers[3],0,growth_rate,dropout_rate,weight_decay,compression,train_flage=True,name_flage='s4')

    
    #stage5    
    D2 = Decon_stage(s4_x,128,kernel_size=(3,3,3),strides=(2,2,2),weight_decay=weight_decay,train_flage=True,name_flage='D2')
    con2 = add([D2,s1_x0])#Apply_Attention(D2,s1_x0) #add([D2,s1_x0])      
    s5_x = up_stage(con2,nb_layers[4],0,growth_rate,dropout_rate,weight_decay,compression,train_flage=True,name_flage='s5')    
  
   
    #stage6  
    D3 = Decon_stage(s5_x,64,kernel_size=(3,3,3),strides=(2,2,2),weight_decay=weight_decay,train_flage=True,name_flage='D3')
    con3 = add([D3,x1])
    s6_x = up_stage(con3,nb_layers[5],0,growth_rate,dropout_rate,weight_decay,compression,train_flage=True,name_flage='s6')       

        
    main_out = output(s6_x,weight_decay,train_flage=True,name_flage='out')  
    model = Model(img_input,main_out)
    sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='binary_crossentropy',metrics=[dice_coef])#'binary_crossentropy'loss_weights=[0.3,0.6,1.0]
    return model  
       



def Discriminator_model(): 
    
    if  K.image_data_format() == 'channels_last':
      img_input1 = Input(shape=(int(img_rows/4),int(img_cols/4),int(chan/4,256)))
      img_input2 = Input(shape=(int(img_rows/2),int(img_cols/2),int(chan/2,128)))
      img_input3 = Input(shape=(img_rows,img_cols,chan,64))
    else:
      img_input1 = Input(shape=(256,int(chan/4), int(img_rows/4),int(img_cols/4)))
      img_input2 = Input(shape=(128,int(chan/2), int(img_rows/2),int(img_cols/2)))
      img_input3 = Input(shape=(64,chan, img_rows, img_cols))    
      

    x = conv_block(img_input1,128)
    x = add([UpSampling3D((2,2,2))(x),img_input2])
    x = conv_block(x,64)
    x = add([UpSampling3D((2,2,2))(x),img_input3])    
    x = conv_block(x,32)
    out = DisOutput(x) 
    optimizer = Adam(1e-5)
    model = Model([img_input1,img_input2,img_input3],out)    
    model.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'] )

    return model
    
   
def Gen_Dis_model(trable_flage=True,reduction=0.5, dropout_rate=0.3, weight_decay=5e-4):


    if  K.image_data_format() == 'channels_last':
      img_input = Input(shape=(img_rows,img_cols,chan,1))
      concat_axis=-1
    else:
      img_input = Input(shape=(1,chan, img_rows, img_cols))
      concat_axis=1


    if reduction != 0.0:
        assert reduction <= 1.0 and reduction > 0.0, 'reduction value must lie between 0.0 and 1.0'

    # compute compression factor
    compression = 1.0 - reduction
    nb_layers=[4,8,16,8,4,2]
    growth_rate=32
    # Initial convolution

    #stage1
    x1 = Conv3D(64, (3,3,3),strides=(1,1,1),kernel_initializer='he_normal',padding='same',
               use_bias=False, kernel_regularizer=l2(weight_decay),trainable=True,name='x1')(img_input)
    x = BatchNormalization(axis=concat_axis,trainable=True,name='x1_BN')(x1)
    x = Activation('relu')(x)
    x = AveragePooling3D((2,2,2))(x)

    #stage1
    s1_x0,s1_x = down_stage(x,nb_layers[0],0,growth_rate,dropout_rate,weight_decay,compression,train_flage=True,name_flage='s1')
  
    #stage2
    s2_x0,s2_x = down_stage(s1_x,nb_layers[1],0,growth_rate,dropout_rate,weight_decay,compression,train_flage=True,name_flage='s2')    

    #stage3 
    s3_x0 = down_stage(s2_x,nb_layers[2],0,growth_rate,dropout_rate,weight_decay,compression,train_flage=True,name_flage='s3',pooling=False)

    #stage4
    D1 = Decon_stage(s3_x0,256,kernel_size=(3,3,3),strides=(2,2,2),weight_decay=weight_decay,train_flage=True,name_flage='D1')
    con1 = add([D1,s2_x0])
    s4_x = up_stage(con1,nb_layers[3],0,growth_rate,dropout_rate,weight_decay,compression,train_flage=True,name_flage='s4')
 
    #stage5    
    D2 = Decon_stage(s4_x,128,kernel_size=(3,3,3),strides=(2,2,2),weight_decay=weight_decay,train_flage=True,name_flage='D2')
    con2 =add([D2,s1_x0])   
    s5_x = up_stage(con2,nb_layers[4],0,growth_rate,dropout_rate,weight_decay,compression,train_flage=True,name_flage='s5')    
   
    #stage6  
    D3 = Decon_stage(s5_x,64,kernel_size=(3,3,3),strides=(2,2,2),weight_decay=weight_decay,train_flage=True,name_flage='D3')

    con3 = add([D3,x1])
    s6_x = up_stage(con3,nb_layers[5],0,growth_rate,dropout_rate,weight_decay,compression,train_flage=True,name_flage='s6')       
    main_out = output(s6_x,weight_decay,train_flage=True,name_flage='out')  

    Discriminator_model.trainable=False
    GAN_loss = Discriminator_model()([s4_x,s5_x,s6_x])    
    model = Model(img_input, [main_out,main_out,GAN_loss]) 
    
    sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss=[Dist_Loss,'binary_crossentropy','binary_crossentropy' ], loss_weights=[0.1,1.0,1.0], metrics=[dice_coef] )#
    return model 
     
    
    
