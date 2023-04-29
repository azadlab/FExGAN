import tensorflow as tf
import tensorflow.keras.backend as K


OUTPUT_CHANNELS = 3

def downsampling_layer(nfilts, ksize, input_shape, add_batchnorm=True,add_activ=True):
    kinit = tf.random_normal_initializer(0., 0.02)

    out = tf.keras.Sequential()
    out.add(tf.keras.layers.Conv2D(nfilts, ksize, strides=2, padding='same', batch_input_shape=input_shape, 
                             kernel_initializer=kinit, use_bias=False))

    if add_batchnorm:
        out.add(tf.keras.layers.BatchNormalization())

    if(add_activ == True):
        out.add(tf.keras.layers.LeakyReLU())
    
    return out

def upsampling_layer(nfilts, ksize, input_shape, add_dropout=False):
    kinit = tf.random_normal_initializer(0., 0.02)

    out = tf.keras.Sequential()
    out.add(tf.keras.layers.Conv2DTranspose(nfilts, ksize, strides=2, batch_input_shape=input_shape,
                                    padding='same',kernel_initializer=kinit,use_bias=False))

    out.add(tf.keras.layers.BatchNormalization())

    if add_dropout:
        out.add(tf.keras.layers.Dropout(0.5))

    out.add(tf.keras.layers.ReLU())

    return out

# sampling function
def sampling(args):
    z_mu, z_log_sigma = args
    epsilon = K.random_normal(shape=(K.shape(z_mu)[0], 128),
                              mean=0., stddev=1.)
    return z_mu + K.exp(z_log_sigma) * epsilon

def makeGenerator():
    input1 = tf.keras.layers.Input(shape=[128,128,3],name='img_input') # The input image
    input2 = tf.keras.layers.Input(shape=(128,),name='latent_input')  # The latent Vector
    input3 = tf.keras.layers.Input(shape=(7,),name='label_input')    # The class label

    down_stack = [
        downsampling_layer(128, 4, (None, 128, 128, 4), add_batchnorm=False), # (BATCH_SIZE, 64, 64, 128)
        downsampling_layer(256, 4, (None, 64, 64, 128)), # (BATCH_SIZE, 32, 32, 256)
        downsampling_layer(512, 4, (None, 32, 32, 256)), # (BATCH_SIZE, 16, 16, 512)
        downsampling_layer(512, 4, (None, 16, 16, 512)), # (BATCH_SIZE, 8, 8, 512)
        downsampling_layer(512, 4, (None, 8, 8, 512)), # (BATCH_SIZE, 4, 4, 512)
        downsampling_layer(512, 4, (None, 4, 4, 512)), # (BATCH_SIZE, 2, 2, 512)
        downsampling_layer(128, 4, (None, 2, 2, 512),add_activ=False), # (BATCH_SIZE, 1, 1, 128)
    ]

    up_stack = [
        upsampling_layer(512, 4, (None, 1, 1, 512), add_dropout=True), # (BATCH_SIZE, 2, 2, 512)
        upsampling_layer(512, 4, (None, 2, 2, 512), add_dropout=True), # (BATCH_SIZE, 4, 4, 512)
        upsampling_layer(512, 4, (None, 4, 4, 512), add_dropout=True), # (BATCH_SIZE, 8, 8, 512)
        upsampling_layer(512, 4, (None, 8, 8, 512)), # (bBATCH_SIZEs, 16, 16, 512)
        upsampling_layer(256, 4, (None, 16, 16, 512)), # (BATCH_SIZE, 32, 32, 256)
        upsampling_layer(128, 4, (None, 32, 32, 256)), # (BATCH_SIZE, 64, 64, 128)        
    ]

    kinit = tf.random_normal_initializer(0., 0.02)
    last_layer = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                           strides=2,
                                           padding='same',
                                           kernel_initializer=kinit,
                                           activation='tanh',name='img_output') # (BATCH_SIZE, 128, 128, 3)
    
    # Encoding the input image and prior label
    
    x = input1
    y = tf.keras.layers.Dense(128*128,kernel_initializer=kinit,activation='relu')(input3)
    y = tf.keras.layers.Reshape((128,128,1))(y)
    x = tf.keras.layers.concatenate([x, y]) # (BATCH_SIZE, 128*128*4)
    
    for down in down_stack:
        x = down(x)
    

    x = tf.keras.layers.Flatten()(x)
    z_mu = tf.keras.layers.Dense(128,name='mu')(x)
    z_log_sigma = tf.keras.layers.Dense(128, kernel_initializer='zeros',bias_initializer='zeros',name='sigma')(x)
    
    # sample vector from the latent normal distribution
    Latent = sampling([z_mu, z_log_sigma])
    
    
    # Decoding the inputs
    
    y = tf.keras.layers.Dense(64)(input3)
    y = tf.keras.layers.Dense(256,kernel_initializer=kinit,activation='relu')(y)
    y = tf.keras.layers.Reshape((1,1,256))(y)
        
    x = tf.keras.layers.Dense(256,kernel_initializer=kinit,activation='relu')(input2)
    x = tf.keras.layers.Reshape((1,1,256))(x)
       
    L = tf.keras.layers.concatenate([x, y]) # (BATCH_SIZE, 1*1*512)
    
    
    
    
    for up in up_stack:
        L = up(L)
       

    L = last_layer(L)

    encoder = tf.keras.Model(inputs=[input1,input3], outputs=[Latent,z_mu,z_log_sigma])
    decoder = tf.keras.Model(inputs=[input2,input3], outputs=L)
    generator = tf.keras.Model(inputs=[input1,input3],outputs=decoder([Latent,input3]))
    return generator,encoder,decoder