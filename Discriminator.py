import tensorflow as tf

def downsampling_layers(nfilts, ksize, add_batchnorm=True):
    kinit = tf.random_normal_initializer(0., 0.02)

    out = tf.keras.Sequential()
    out.add(tf.keras.layers.Conv2D(nfilts, ksize, strides=2, padding='same', 
                             kernel_initializer=kinit, use_bias=False))

    if add_batchnorm:
        out.add(tf.keras.layers.BatchNormalization())

    out.add(tf.keras.layers.LeakyReLU())

    return out

def makeDiscriminator():
    kinit = tf.random_normal_initializer(0., 0.02)

    input_img = tf.keras.layers.Input(shape=[128, 128, 3], name='input_image')
    #label = tf.keras.layers.Input(shape=(7,))
    
    L = downsampling_layers(64, 4, False)(input_img) # (BATCH_SIZE, 64, 64, 64)
    L = downsampling_layers(128, 4)(L) # (BATCH_SIZE, 32, 32, 128)
    L = downsampling_layers(512, 4)(L) # (BATCH_SIZE, 32, 32, 512)
        
    L = tf.keras.layers.Flatten()(L)
    
    out1 = tf.keras.layers.Dense(1,activation='sigmoid')(L) # (BATCH_SIZE, 1)
    out2 = tf.keras.layers.Dense(7,activation='softmax')(L) # (BATCH_SIZE, 1, 7)

    return tf.keras.Model(inputs=input_img, outputs=[out1,out2])