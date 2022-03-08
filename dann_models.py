import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import initializers
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from keras_unet.metrics import iou, iou_thresholded

def conv_block(x, filter_size, size, dropout, batch_norm=False):
    
    conv = Conv2D(size, (filter_size, filter_size), padding="same")(x)
    if batch_norm is True:
        conv = BatchNormalization(axis=3)(conv)
    conv = Activation("relu")(conv)

    conv = Conv2D(size, (filter_size, filter_size), padding="same")(conv)
    if batch_norm is True:
        conv = BatchNormalization(axis=3)(conv)
    conv = Activation("relu")(conv)
    
    if dropout > 0:
        conv = Dropout(dropout)(conv)

    return conv

def repeat_elem(tensor, rep):
    # lambda function to repeat Repeats the elements of a tensor along an axis
    #by a factor of rep.
    # If tensor has shape (None, 256,256,3), lambda will return a tensor of shape 
    #(None, 256,256,6), if specified axis=3 and rep=2.

     return Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3),
                          arguments={'repnum': rep})(tensor)

def res_conv_block(x, filter_size, size, dropout, batch_norm=False, name='default'):
    '''
    Residual convolutional layer.
    Two variants....
    Either put activation function before the addition with shortcut
    or after the addition (which would be as proposed in the original resNet).
    
    1. conv - BN - Activation - conv - BN - Activation 
                                          - shortcut  - BN - shortcut+BN
                                          
    2. conv - BN - Activation - conv - BN   
                                     - shortcut  - BN - shortcut+BN - Activation                                     
    
    Check fig 4 in https://arxiv.org/ftp/arxiv/papers/1802/1802.06955.pdf
    '''

    conv = Conv2D(size, (filter_size, filter_size), padding='same')(x)
    if batch_norm is True:
        conv = BatchNormalization(axis=3)(conv)
    conv = Activation('relu')(conv)
    
    conv = Conv2D(size, (filter_size, filter_size), padding='same')(conv)
    if batch_norm is True:
        conv = BatchNormalization(axis=3)(conv)
    #conv = Activation('relu')(conv)    #Activation before addition with shortcut
    if dropout > 0:
        conv = Dropout(dropout)(conv)

    shortcut = Conv2D(size, kernel_size=(1, 1), padding='same')(x)
    if batch_norm is True:
        shortcut = BatchNormalization(axis=3)(shortcut)

    res_path = add([shortcut, conv])
    
    if name != 'default':
        res_path = Activation('relu', name=name)(res_path)    #Activation after addition with shortcut (Original residual block)
    else:
        res_path = Activation('relu')(res_path)    #Activation after addition with shortcut (Original residual block)

    return res_path

def res_custom_conv_block(custom_conv, filter_size, size, dropout, batch_norm=False):
    '''
    Residual convolutional layer.
    Two variants....
    Either put activation function before the addition with shortcut
    or after the addition (which would be as proposed in the original resNet).
    
    1. custom_conv - BN - Activation - conv - BN - Activation 
                                            - shortcut  - BN - shortcut+BN
                                          
    2. custom_conv - BN - Activation - conv - BN   
                                            - shortcut  - BN - shortcut+BN - Activation                                     
    
    Check fig 4 in https://arxiv.org/ftp/arxiv/papers/1802/1802.06955.pdf
    '''

    conv = custom_conv
    if batch_norm is True:
        conv = BatchNormalization(axis=3)(conv)
    conv = Activation('relu')(conv)
    
    conv = Conv2D(size, (filter_size, filter_size), padding='same')(conv)
    if batch_norm is True:
        conv = BatchNormalization(axis=3)(conv)
    #conv = Activation('relu')(conv)    #Activation before addition with shortcut
    if dropout > 0:
        conv = Dropout(dropout)(conv)
    shortcut = Conv2D(size, (filter_size, filter_size), padding='same')(custom_conv)
    if batch_norm is True:
        shortcut = BatchNormalization(axis=3)(shortcut)

    res_path = add([shortcut, conv])
    res_path = Activation('relu')(res_path)    #Activation after addition with shortcut (Original residual block)
    return res_path

def gating_signal(input, out_size, batch_norm=False):
    """
    resize the down layer feature map into the same dimension as the up layer feature map
    using 1x1 conv
    :return: the gating feature map with the same dimension of the up layer feature map
    """
    x = Conv2D(out_size, (1, 1), padding='same')(input)
    if batch_norm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def attention_block(x, gating, inter_shape):
    shape_x = K.int_shape(x)
    shape_g = K.int_shape(gating)

# Getting the x signal to the same shape as the gating signal
    theta_x = Conv2D(inter_shape, (2, 2), strides=(2, 2), padding='same')(x)  # 16
    shape_theta_x = K.int_shape(theta_x)

# Getting the gating signal to the same number of filters as the inter_shape
    phi_g = Conv2D(inter_shape, (1, 1), padding='same')(gating)
    upsample_g = Conv2DTranspose(inter_shape, (3, 3),
                                 strides=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]),
                                 padding='same')(phi_g)  # 16

    concat_xg = add([upsample_g, theta_x])
    act_xg = Activation('relu')(concat_xg)
    psi = Conv2D(1, (1, 1), padding='same')(act_xg)
    sigmoid_xg = Activation('sigmoid')(psi)
    shape_sigmoid = K.int_shape(sigmoid_xg)
    upsample_psi = UpSampling2D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]))(sigmoid_xg)  # 32

    upsample_psi = repeat_elem(upsample_psi, shape_x[3])

    y = multiply([upsample_psi, x])

    result = Conv2D(shape_x[3], (1, 1), padding='same')(y)
    result_bn = BatchNormalization()(result)
    return result_bn

@tf.custom_gradient
def grad_reverse(x,l):
    y = tf.identity(x)
    def custom_grad(dy):
        return -dy * l, None
    return y, custom_grad

class GradReverse(Layer):
    #def __init__(self, l, **kwargs):
    def __init__(self, l):
        super().__init__()
        self.l = tf.Variable(l, trainable=False)

    def call(self, x):
        return grad_reverse(x,self.l)

    def get_config(self):
        return {'l': self.l.numpy()}

def DANN_Attention_Residual_Unet(input_shape, num_classes=1, dropout_rate = 0.0, batch_norm = True, filter_num=64):
    # https://github.com/zhou912/Learning-from-unlabeled-real-seismic-data-fault-detection-based-on-transfer-learning
    # Implement code as 2D with Attention Residual U-Net
    '''
    Residual UNet, with attention 

    '''
    # network structure  
    FILTER_NUM = 32 # number of basic filters for the first layer
    FILTER_SIZE = 3 # size of the convolutional filter
    UP_SAMP_SIZE = 2 # size of upsampling filters
    # input data
    # dimension of the image depth
    input_img = Input(input_shape, dtype=tf.float32)
    axis = 3

    # Downsampling layers
    # DownRes 1, double residual convolution + pooling
    conv_128 = res_conv_block(input_img, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    pool_64 = MaxPooling2D(pool_size=(2,2))(conv_128)
    # DownRes 2
    conv_64 = res_conv_block(pool_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    pool_32 = MaxPooling2D(pool_size=(2,2))(conv_64)
    # DownRes 3
    conv_32 = res_conv_block(pool_32, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    pool_16 = MaxPooling2D(pool_size=(2,2))(conv_32)
    # DownRes 4
    conv_16 = res_conv_block(pool_16, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)
    pool_8 = MaxPooling2D(pool_size=(2,2))(conv_16)
    # DownRes 5, convolution only
    conv_8 = res_conv_block(pool_8, FILTER_SIZE, 16*FILTER_NUM, dropout_rate, batch_norm)

    # Upsampling layers
    # UpRes 6, attention gated concatenation + upsampling + double residual convolution
    gating_16 = gating_signal(conv_8, 8*FILTER_NUM, batch_norm)
    att_16 = attention_block(conv_16, gating_16, 8*FILTER_NUM)
    up_16 = UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(conv_8)
    up_16 = concatenate([up_16, att_16], axis=axis)
    up_conv_16 = res_conv_block(up_16, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 7
    gating_32 = gating_signal(up_conv_16, 4*FILTER_NUM, batch_norm)
    att_32 = attention_block(conv_32, gating_32, 4*FILTER_NUM)
    up_32 = UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_16)
    up_32 = concatenate([up_32, att_32], axis=axis)
    up_conv_32 = res_conv_block(up_32, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 8
    gating_64 = gating_signal(up_conv_32, 2*FILTER_NUM, batch_norm)
    att_64 = attention_block(conv_64, gating_64, 2*FILTER_NUM)
    up_64 = UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_32)
    up_64 = concatenate([up_64, att_64], axis=axis)
    up_conv_64 = res_conv_block(up_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm, name="class_1")
    # UpRes 9
    gating_128 = gating_signal(up_conv_64, FILTER_NUM, batch_norm)
    att_128 = attention_block(conv_128, gating_128, FILTER_NUM)
    up_128 = UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_64)
    up_128 = concatenate([up_128, att_128], axis=axis)
    up_conv_128 = res_conv_block(up_128, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm, name="class_2")

    # 1*1 convolutional layers

    class_output = Conv2D(num_classes, kernel_size=(1,1))(up_conv_128)
    class_output = BatchNormalization(axis=axis)(class_output)
    
    if num_classes == 1:
        class_output = Activation('sigmoid', name="class_output")(class_output)
    else: 
        class_output = Activation('softmax', name="class_output")(class_output)

    conv_d = Conv2D(1, (1, 1), activation='relu', padding='same',name = 'dis_0')(up_conv_32)
    discriminator_feature = Flatten(name = 'dis_3')(conv_d)
    #domain classifier
    #grl_layer = GradientReversal(-1.0)
    #dann classifier
    grl_layer = GradReverse(-1.0)
    discriminator_feature = grl_layer(discriminator_feature)
    domain_1 = Dense(256,activation='relu',name = 'dis_1')(discriminator_feature)
    domain_2 = Dropout(0.5,name = 'dis_2')(domain_1)

    discriminator_output = Dense(2,activation="softmax",name="discriminator_output")(domain_2)

    discriminator_model = Model(inputs=[input_img], outputs=[discriminator_output])
    # discriminator_model.compile(optimizer=LazyAdam(lr = 1e-3), 
    #                         loss='categorical_crossentropy', 
    #                         metrics=['accuracy'])

    model = Model(inputs = [input_img],outputs = [class_output,discriminator_output])
    # model.compile(optimizer=LazyAdam(lr = 1e-3), 
    #           loss={'class_output':dice_loss, 
    #                 'discriminator_output':'categorical_crossentropy'}, 
    #           loss_weights = {'class_output':1,'discriminator_output':0}, 

    source_classification_model = Model(inputs=[input_img], outputs=[class_output])
    # source_classification_model.compile(optimizer=LazyAdam(lr = 1e-3), 
    #                                 loss=dice_loss, 
    #                                 metrics=metrics)

    return discriminator_model, source_classification_model, model

def DANN_v2_Attention_Residual_Unet(input_shape, num_classes=1, dropout_rate = 0.0, batch_norm = True, filter_num=64):
    # https://github.com/zhou912/Learning-from-unlabeled-real-seismic-data-fault-detection-based-on-transfer-learning
    # Implement code as 2D with Attention Residual U-Net
    '''
    Residual UNet, with attention 

    '''
    # network structure  
    FILTER_NUM = 32 # number of basic filters for the first layer
    FILTER_SIZE = 3 # size of the convolutional filter
    UP_SAMP_SIZE = 2 # size of upsampling filters
    # input data
    # dimension of the image depth
    input_img = Input(input_shape, dtype=tf.float32)
    axis = 3

    # Downsampling layers
    # DownRes 1, double residual convolution + pooling
    conv_128 = res_conv_block(input_img, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    pool_64 = MaxPooling2D(pool_size=(2,2))(conv_128)
    # DownRes 2
    conv_64 = res_conv_block(pool_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    pool_32 = MaxPooling2D(pool_size=(2,2))(conv_64)
    # DownRes 3
    conv_32 = res_conv_block(pool_32, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    pool_16 = MaxPooling2D(pool_size=(2,2))(conv_32)
    # DownRes 4
    conv_16 = res_conv_block(pool_16, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)
    pool_8 = MaxPooling2D(pool_size=(2,2))(conv_16)
    # DownRes 5, convolution only
    conv_8 = res_conv_block(pool_8, FILTER_SIZE, 16*FILTER_NUM, dropout_rate, batch_norm)

    # Upsampling layers
    # UpRes 6, attention gated concatenation + upsampling + double residual convolution
    gating_16 = gating_signal(conv_8, 8*FILTER_NUM, batch_norm)
    att_16 = attention_block(conv_16, gating_16, 8*FILTER_NUM)
    up_16 = UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(conv_8)
    up_16 = concatenate([up_16, att_16], axis=axis)
    up_conv_16 = res_conv_block(up_16, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 7
    gating_32 = gating_signal(up_conv_16, 4*FILTER_NUM, batch_norm)
    att_32 = attention_block(conv_32, gating_32, 4*FILTER_NUM)
    up_32 = UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_16)
    up_32 = concatenate([up_32, att_32], axis=axis)
    up_conv_32 = res_conv_block(up_32, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 8
    gating_64 = gating_signal(up_conv_32, 2*FILTER_NUM, batch_norm)
    att_64 = attention_block(conv_64, gating_64, 2*FILTER_NUM)
    up_64 = UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_32)
    up_64 = concatenate([up_64, att_64], axis=axis)
    up_conv_64 = res_conv_block(up_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm, name="class_1")
    # UpRes 9
    gating_128 = gating_signal(up_conv_64, FILTER_NUM, batch_norm)
    att_128 = attention_block(conv_128, gating_128, FILTER_NUM)
    up_128 = UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_64)
    up_128 = concatenate([up_128, att_128], axis=axis)
    up_conv_128 = res_conv_block(up_128, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm, name="class_2")

    # 1*1 convolutional layers

    class_output = Conv2D(num_classes, kernel_size=(1,1))(up_conv_128)
    class_output = BatchNormalization(axis=axis)(class_output)
    
    if num_classes == 1:
        class_output = Activation('sigmoid', name="class_output")(class_output)
    else: 
        class_output = Activation('softmax', name="class_output")(class_output)

    conv_d = Conv2D(1, (1, 1), activation='relu', padding='same',name = 'dis_0')(up_conv_64)
    discriminator_feature = Flatten(name = 'dis_3')(conv_d)
    #domain classifier
    #grl_layer = GradientReversal(-1.0)
    #dann classifier
    grl_layer = GradReverse(-1.0)
    discriminator_feature = grl_layer(discriminator_feature)
    domain_1 = Dense(256,activation='relu',name = 'dis_1')(discriminator_feature)
    domain_2 = Dropout(0.5,name = 'dis_2')(domain_1)

    discriminator_output = Dense(2,activation="softmax",name="discriminator_output")(domain_2)

    discriminator_model = Model(inputs=[input_img], outputs=[discriminator_output])
    # discriminator_model.compile(optimizer=LazyAdam(lr = 1e-3), 
    #                         loss='categorical_crossentropy', 
    #                         metrics=['accuracy'])

    model = Model(inputs = [input_img],outputs = [class_output,discriminator_output])
    # model.compile(optimizer=LazyAdam(lr = 1e-3), 
    #           loss={'class_output':dice_loss, 
    #                 'discriminator_output':'categorical_crossentropy'}, 
    #           loss_weights = {'class_output':1,'discriminator_output':0}, 

    source_classification_model = Model(inputs=[input_img], outputs=[class_output])
    # source_classification_model.compile(optimizer=LazyAdam(lr = 1e-3), 
    #                                 loss=dice_loss, 
    #                                 metrics=metrics)

    return discriminator_model, source_classification_model, model
    
def DANN_v3_Attention_Residual_Unet(input_shape, num_classes=1, dropout_rate = 0.0, batch_norm = True, filter_num=64):
    # https://github.com/zhou912/Learning-from-unlabeled-real-seismic-data-fault-detection-based-on-transfer-learning
    # Implement code as 2D with Attention Residual U-Net
    '''
    Residual UNet, with attention 

    '''
    # network structure  
    FILTER_NUM = 32 # number of basic filters for the first layer
    FILTER_SIZE = 3 # size of the convolutional filter
    UP_SAMP_SIZE = 2 # size of upsampling filters
    # input data
    # dimension of the image depth
    input_img = Input(input_shape, dtype=tf.float32)
    axis = 3

    # Downsampling layers
    # DownRes 1, double residual convolution + pooling
    conv_128 = res_conv_block(input_img, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    pool_64 = MaxPooling2D(pool_size=(2,2))(conv_128)
    # DownRes 2
    conv_64 = res_conv_block(pool_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    pool_32 = MaxPooling2D(pool_size=(2,2))(conv_64)
    # DownRes 3
    conv_32 = res_conv_block(pool_32, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    pool_16 = MaxPooling2D(pool_size=(2,2))(conv_32)
    # DownRes 4
    conv_16 = res_conv_block(pool_16, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)
    pool_8 = MaxPooling2D(pool_size=(2,2))(conv_16)
    # DownRes 5, convolution only
    conv_8 = res_conv_block(pool_8, FILTER_SIZE, 16*FILTER_NUM, dropout_rate, batch_norm)

    # Upsampling layers
    # UpRes 6, attention gated concatenation + upsampling + double residual convolution
    gating_16 = gating_signal(conv_8, 8*FILTER_NUM, batch_norm)
    att_16 = attention_block(conv_16, gating_16, 8*FILTER_NUM)
    up_16 = UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(conv_8)
    up_16 = concatenate([up_16, att_16], axis=axis)
    up_conv_16 = res_conv_block(up_16, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 7
    gating_32 = gating_signal(up_conv_16, 4*FILTER_NUM, batch_norm)
    att_32 = attention_block(conv_32, gating_32, 4*FILTER_NUM)
    up_32 = UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_16)
    up_32 = concatenate([up_32, att_32], axis=axis)
    up_conv_32 = res_conv_block(up_32, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 8
    gating_64 = gating_signal(up_conv_32, 2*FILTER_NUM, batch_norm)
    att_64 = attention_block(conv_64, gating_64, 2*FILTER_NUM)
    up_64 = UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_32)
    up_64 = concatenate([up_64, att_64], axis=axis)
    up_conv_64 = res_conv_block(up_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm, name="class_1")
    # UpRes 9
    gating_128 = gating_signal(up_conv_64, FILTER_NUM, batch_norm)
    att_128 = attention_block(conv_128, gating_128, FILTER_NUM)
    up_128 = UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_64)
    up_128 = concatenate([up_128, att_128], axis=axis)
    up_conv_128 = res_conv_block(up_128, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm, name="class_2")

    # 1*1 convolutional layers

    class_output = Conv2D(num_classes, kernel_size=(1,1))(up_conv_128)
    class_output = BatchNormalization(axis=axis)(class_output)
    
    if num_classes == 1:
        class_output = Activation('sigmoid', name="class_output")(class_output)
    else: 
        class_output = Activation('softmax', name="class_output")(class_output)

    conv_d = Conv2D(1, (1, 1), activation='relu', padding='same',name = 'dis_0')(up_conv_16)
    discriminator_feature = Flatten(name = 'dis_3')(conv_d)
    #domain classifier
    #grl_layer = GradientReversal(-1.0)
    #dann classifier
    grl_layer = GradReverse(-1.0)
    discriminator_feature = grl_layer(discriminator_feature)
    domain_1 = Dense(256,activation='relu',name = 'dis_1')(discriminator_feature)
    domain_2 = Dropout(0.5,name = 'dis_2')(domain_1)

    discriminator_output = Dense(2,activation="softmax",name="discriminator_output")(domain_2)

    discriminator_model = Model(inputs=[input_img], outputs=[discriminator_output])
    # discriminator_model.compile(optimizer=LazyAdam(lr = 1e-3), 
    #                         loss='categorical_crossentropy', 
    #                         metrics=['accuracy'])

    model = Model(inputs = [input_img],outputs = [class_output,discriminator_output])
    # model.compile(optimizer=LazyAdam(lr = 1e-3), 
    #           loss={'class_output':dice_loss, 
    #                 'discriminator_output':'categorical_crossentropy'}, 
    #           loss_weights = {'class_output':1,'discriminator_output':0}, 

    source_classification_model = Model(inputs=[input_img], outputs=[class_output])
    # source_classification_model.compile(optimizer=LazyAdam(lr = 1e-3), 
    #                                 loss=dice_loss, 
    #                                 metrics=metrics)

    return discriminator_model, source_classification_model, model
    
def DANN_Mini_Unet(input_shape, num_classes=1):
    input_img = Input(input_shape)
    conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool3)
    """
    de1 = UpSampling2D(size=(2, 2))(conv4)
    de2 = Conv2D(32, (3, 3), activation='relu', padding='same')(de1)
    de2 = Conv2D(32, (3, 3), activation='relu', padding='same')(de2)
    de3 = UpSampling2D(size=(2, 2))(de2)
    de4 = Conv2D(16, (3, 3), activation='relu', padding='same')(de3)
    de4 = Conv2D(16, (3, 3), activation='relu', padding='same')(de4)
    de5 = UpSampling2D(size=(2, 2))(de4)
    decoder_output = Conv2D(1, (3, 3), activation='relu', padding='same',name = 'decoder_output')(de5)
    """

    up5 = concatenate([UpSampling2D(size=(2, 2))(conv4), conv3], axis=3)
    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(up5)
    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv2], axis=3)
    conv6 = Conv2D(32, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv1], axis=3)
    conv7 = Conv2D(16, (3, 3), activation='relu', padding='same',name = "class_1")(up7)
    conv8 = Conv2D(16, (3, 3), activation='relu', padding='same',name = "class_2")(conv7)

    if num_classes == 1:
        class_output = Conv2D(num_classes, (1, 1), activation='sigmoid',name="class_output")(conv8)
    else: 
        class_output = Conv2D(num_classes, (1, 1), activation='softmax',name="class_output")(conv8)

    conv_d = Conv2D(1, (1, 1), activation='relu', padding='same',name = 'dis_0')(conv6)
    discriminator_feature = Flatten(name = 'dis_3')(conv_d)
    #domain classifier
    #grl_layer = GradientReversal(-1.0)
    #dann classifier
    grl_layer = GradReverse(-1.0)
    discriminator_feature = grl_layer(discriminator_feature)
    domain_1 = Dense(256,activation='relu',name = 'dis_1')(discriminator_feature)
    domain_2 = Dropout(0.5,name = 'dis_2')(domain_1)

    discriminator_output = Dense(2,activation="softmax",name="discriminator_output")(domain_2)

    discriminator_model = Model(inputs=[input_img], outputs=[discriminator_output])
    # discriminator_model.compile(optimizer=LazyAdam(lr = 1e-3), 
    #                         loss='categorical_crossentropy', 
    #                         metrics=['accuracy'])

    model = Model(inputs = [input_img],outputs = [class_output,discriminator_output])
    # model.compile(optimizer=LazyAdam(lr = 1e-3), 
    #           loss={'class_output':dice_loss, 
    #                 'discriminator_output':'categorical_crossentropy'}, 
    #           loss_weights = {'class_output':1,'discriminator_output':0}, 

    source_classification_model = Model(inputs=[input_img], outputs=[class_output])
    # source_classification_model.compile(optimizer=LazyAdam(lr = 1e-3), 
    #                                 loss=dice_loss, 
    #                                 metrics=[iou,iou_thresholded])
    
    return discriminator_model, source_classification_model, model
