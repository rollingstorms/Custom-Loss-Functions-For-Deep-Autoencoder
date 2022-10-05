from tensorflow.keras import Model, layers
import tensorflow as tf
import numpy as np

class Conv2DLayerBlock(Model):
    
    def __init__(self, latent_dim, filters=(16,32,64,128), kernel_size=3, strides=1, pool_size=2):
        super(Conv2DLayerBlock, self).__init__(latent_dim, filters, kernel_size, strides, pool_size)
        
        self.conv2D_layers = []
        for f in filters:
            if pool_size > 1:
                conv2D_layer = [
                    layers.Conv2D(f, kernel_size=kernel_size, strides=strides, padding='same', activation='swish'),
                    layers.MaxPooling2D(pool_size=pool_size),
                    layers.BatchNormalization(axis=-1)
                ]
            else:
                conv2D_layer = [
                    layers.Conv2D(f, kernel_size=kernel_size, strides=strides, padding='same', activation='swish'),
                    layers.BatchNormalization(axis=-1)
                ]
            self.conv2D_layers.append(conv2D_layer)
        self.dense_layer = layers.Dense(latent_dim)
        self.flatten = layers.Flatten()
        

    def call(self, x):
        for conv_layer in self.conv2D_layers:
            for layer in conv_layer:
                x = layer(x)
            
        self.final_shape = x.shape

        x = self.dense_layer(self.flatten(x))
        return x

class TimeFreqConv1DLayerBlock(Model):
    
    def __init__(self, latent_dim, filters=(16,32,64,128), kernel_size=3, strides=1, pool_size=2):
        super(TimeFreqConv1DLayerBlock, self).__init__(latent_dim, filters, kernel_size, strides, pool_size)
        
        self.time_conv1D_layers = []
        self.freq_conv1D_layers = []
        for f in filters:
            if pool_size > 1:
                time_conv1D_layer = [
                    layers.Conv1D(f, kernel_size=kernel_size, strides=strides, padding='same', activation='swish'),
                    layers.MaxPooling2D(pool_size=pool_size),
                    layers.BatchNormalization(axis=-1)
                ]
                freq_conv1D_layer = [
                    layers.Conv1D(f, kernel_size=kernel_size, strides=strides, padding='same', activation='swish'),
                    layers.MaxPooling2D(pool_size=pool_size),
                    layers.BatchNormalization(axis=-1)
                ]
            else:
                time_conv1D_layer = [
                    layers.Conv1D(f, kernel_size=kernel_size, strides=strides, padding='same', activation='swish'),
                    layers.BatchNormalization(axis=-1)
                ]
                freq_conv1D_layer = [
                    layers.Conv1D(f, kernel_size=kernel_size, strides=strides, padding='same', activation='swish'),
                    layers.BatchNormalization(axis=-1)
                ]
            self.time_conv1D_layers.append(time_conv1D_layer)
            self.freq_conv1D_layers.append(freq_conv1D_layer)

        self.dense_layer = layers.Dense(latent_dim)
        self.flatten = layers.Flatten()
        

    def call(self, x):
        x_t = tf.transpose(x, perm=[0,2,1,3])
        for conv_layer in self.time_conv1D_layers:
            for layer in conv_layer:
                x = layer(x)
        for conv_layer in self.freq_conv1D_layers:
            for layer in conv_layer:
                x_t = layer(x_t)


            
        self.final_shape = x.shape
        x = layers.Concatenate(axis=1)([self.flatten(x), self.flatten(x_t)])
        x = self.dense_layer(x)
        return x
    
    
class TileEncoderBlock(tf.keras.Model):
    
    def __init__(self, batch_size, image_shape, num_tiles, latent_dim, filters=(16,32,64,128), kernel_size=3, strides=1, pool_size=2):
        super(TileEncoderBlock, self).__init__(self, batch_size, image_shape, num_tiles, latent_dim, filters, kernel_size, strides, pool_size)
        self.batch_size = batch_size
        self.num_tiles = num_tiles
        self.latent_dim = latent_dim
        self.image_shape = image_shape
        self.conv2d_block = Conv2DLayerBlock(latent_dim, filters, kernel_size, strides, pool_size)

        
    def call(self, inputs):
        input_shape = inputs.shape
        x = inputs
        
        x = tf.reshape(x, shape=(self.batch_size*self.num_tiles, self.image_shape[0], self.image_shape[1], 1))  
        x = self.conv2d_block(x)
        x = tf.reshape(x, shape=(self.batch_size,self.num_tiles, -1))

        return x
    
def positional_encoding(length, depth):
    depth = depth/2

    positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)

    angle_rates = 1 / (10000**depths)         # (1, depth)
    angle_rads = positions * angle_rates      # (pos, depth)

    pos_encoding = np.concatenate(
      [np.sin(angle_rads), np.cos(angle_rads)],
      axis=-1) 

    return tf.cast(pos_encoding, dtype=tf.float32)

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, length, depth):
        super().__init__()
        self.depth = depth
        self.pos_encoding = positional_encoding(length=length, depth=depth)


    def call(self, x):
        length = tf.shape(x)[1]
        x *= tf.math.sqrt(tf.cast(self.depth, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :, :]
        return x

class AttentionBlock(tf.keras.Model):
    
    def __init__(self, num_tiles, latent_dim, num_heads):
        super(AttentionBlock, self).__init__(self, num_tiles, latent_dim, num_heads)
        self.batch_norm = layers.BatchNormalization()
        self.sequence_mask = tf.sequence_mask(range(1,num_tiles+1))
        self.attention_layer = layers.MultiHeadAttention(num_heads=num_heads, key_dim=2)
        self.positional_encoding = PositionalEncoding(length=num_tiles, depth=latent_dim)
        self.flatten_layer = layers.Flatten()
        self.dense_layer = layers.Dense(units=num_tiles*latent_dim, activation='swish')
        self.reshape = layers.Reshape(target_shape=(num_tiles, latent_dim))
        
    def call(self, inputs):
        
        x = self.batch_norm(inputs)
        x = self.positional_encoding(x)
        x = self.attention_layer(x, x) #, attention_mask=self.sequence_mask)
        res = x + inputs
        x = self.flatten_layer(x)
        x = self.dense_layer(x)
        x = self.reshape(x)
        x = x + res
        return x

class VariationalLayer(tf.keras.layers.Layer):
    def __init__(self, latent_dim):
        super().__init__()

        self.flatten = layers.Flatten()
        self.mu_dense = layers.Dense(latent_dim)
        self.log_variance_dense = layers.Dense(latent_dim)
 
    def sample_point_from_normal_distribution(self, args):
        mu, log_variance = args
        epsilon = tf.random.normal(shape=(self.mu_dense.units,), mean=0., stddev=1.)
        sampled_point = mu + tf.exp(log_variance / 2) * epsilon

        return sampled_point

    def call(self, x):
        x = self.flatten(x)

        self.mu = self.mu_dense(x)
        self.log_variance = self.log_variance_dense(x)

        x = layers.Lambda(self.sample_point_from_normal_distribution,
               name="encoder_output")((self.mu, self.log_variance))

        return x

class Encoder(tf.keras.Model):
    
    def __init__(self, batch_size, image_shape, num_tiles, latent_dim, final_dim, num_attn_layers=2, num_heads=2, filters=(16,32,64,128), kernel_size=3, strides=1, pool_size=2):
        super(Encoder, self).__init__(self, batch_size, image_shape, num_tiles, latent_dim, final_dim, num_attn_layers, num_heads, filters, kernel_size, strides, pool_size)
        
        self.tile_encoder_block = TileEncoderBlock(batch_size=batch_size, image_shape=image_shape, num_tiles=num_tiles, latent_dim=latent_dim, filters=filters, kernel_size=kernel_size, strides=strides, pool_size=pool_size)
        self.attention_blocks = [AttentionBlock(num_tiles=num_tiles, latent_dim=latent_dim, num_heads=num_heads) for _ in range(num_attn_layers)]
        self.dense_bottleneck = layers.Dense(units=final_dim) #, activation='swish')
        self.variational_layer = VariationalLayer(final_dim)
        
    def call(self, inputs):
        x = self.tile_encoder_block(inputs)
        for block in self.attention_blocks:
            x = block(x)
        x = layers.Flatten()(x)
        x = self.dense_bottleneck(x)
        # x = self.variational_layer(x)
        return x
    
class Conv2DTransposeLayerBlock(tf.keras.Model):
    
    def __init__(self, input_image_shape, latent_dim, filters=(16,32,64,128), kernel_size=3, strides=1, pool_size=2):
        super(Conv2DTransposeLayerBlock, self).__init__(input_image_shape, latent_dim, filters, kernel_size, strides, pool_size)
        
        self.dense_layer = layers.Dense(tf.math.reduce_prod(input_image_shape[1:]), activation='swish')
        self.reshape = layers.Reshape(input_image_shape[1:])
        
        self.conv2D_T_layers = []
        for f in filters[::-1]:
            if pool_size > 1:
                conv2D_T_layer = [
                    layers.Conv2DTranspose(f, kernel_size=kernel_size, strides=strides, padding='same', activation='swish'),
                    layers.UpSampling2D(size=pool_size),
                    layers.BatchNormalization(axis=-1)
                ]
            else:
                conv2D_T_layer = [
                    layers.Conv2DTranspose(f, kernel_size=kernel_size, strides=strides, padding='same', activation='swish'),
                    layers.BatchNormalization(axis=-1)
                ]
            self.conv2D_T_layers.append(conv2D_T_layer)
        self.final_conv2D_layer = layers.Conv2DTranspose(1, kernel_size=kernel_size, strides=1, padding='same', activation='sigmoid')

    def call(self, inputs):
        x = inputs
        x = self.dense_layer(x)
        x = self.reshape(x)
        for i, conv_layer in enumerate(self.conv2D_T_layers):
            for layer in conv_layer:
                x = layer(x)


        x = self.final_conv2D_layer(x)
        return x
        
class TileDecoderBlock(tf.keras.Model):
    
    def __init__(self, batch_size, input_image_shape, final_image_shape, num_tiles, latent_dim, filters=(16,32,64,128), kernel_size=3, strides=1, pool_size=2):
        super(TileDecoderBlock, self).__init__(self, batch_size, input_image_shape, final_image_shape, num_tiles, latent_dim, filters, kernel_size, strides, pool_size)
        self.batch_size = batch_size
        self.num_tiles = num_tiles
        self.latent_dim = latent_dim
        self.final_image_shape = final_image_shape
        self.conv2d_T_block = Conv2DTransposeLayerBlock(input_image_shape, latent_dim, filters, kernel_size, strides, pool_size)
        

    def call(self, inputs):
        input_shape = inputs.shape
        x = inputs
        x = tf.reshape(x, shape=(self.batch_size*self.num_tiles, -1))
        x = self.conv2d_T_block(x)
        x = tf.reshape(x, shape=(self.batch_size, self.num_tiles, self.final_image_shape[0], self.final_image_shape[1], 1))
        

        return x
        
    
class Decoder(tf.keras.Model):
    
    def __init__(self, input_image_shape, final_image_shape, batch_size, num_tiles, latent_dim, filters=(16,32,64,128), kernel_size=3, strides=1, pool_size=2):
        super(Decoder, self).__init__(self, input_image_shape, batch_size, num_tiles, latent_dim, filters, kernel_size, strides, pool_size)
        self.dense_layer = layers.Dense(units=num_tiles*latent_dim, activation='tanh')
        self.reshape = layers.Reshape(target_shape=(num_tiles, latent_dim))
        self.tile_decoder = TileDecoderBlock(batch_size, input_image_shape, final_image_shape, num_tiles, latent_dim, filters, kernel_size, strides, pool_size)
        
    def call(self, x):
        x = self.dense_layer(x)
        x = self.reshape(x)
        x = self.tile_decoder(x)
        
        return x

class Autoencoder(tf.keras.Model):
    
    def __init__(self, batch_size, image_shape, num_tiles, latent_dim, final_dim, num_heads=2, num_attn_layers=2, filters=(64,128,256,512), kernel_size=3, strides=1, pool_size=2, final_image_shape=(1, 4, 2, 512)):
        super(Autoencoder, self).__init__(self, batch_size, image_shape, num_tiles, latent_dim, final_dim, num_heads, filters, kernel_size, strides, pool_size, final_image_shape)
        convoluted_image_shape = np.array((1, *np.array(image_shape) / 2**len(filters), filters[-1]), dtype=np.int32)
        self.encoder = Encoder(batch_size=batch_size, image_shape=image_shape, num_tiles=num_tiles, latent_dim=latent_dim, final_dim=final_dim, num_heads=num_heads, num_attn_layers=num_attn_layers, filters=filters, kernel_size=kernel_size, strides=strides, pool_size=pool_size)
        self.decoder = Decoder(convoluted_image_shape, image_shape, batch_size, num_tiles, latent_dim, filters, kernel_size, strides, pool_size)
        
    def call(self, inputs):
        x = self.encoder(inputs)
        x = self.decoder(x)
        
        return x
    
class MelAutoencoder(Model):
    
    def __init__(self, latent_dim, filters=(64,128,256,512), kernel_size=3, strides=1, pool_size=2, final_image_shape=(1, 4, 4, 512)):
        super(MelAutoencoder, self).__init__(latent_dim, filters, kernel_size, strides, pool_size, final_image_shape)

        self.encoder = Conv2DLayerBlock(latent_dim, filters, kernel_size, strides, pool_size)
        self.decoder = Conv2DTransposeLayerBlock(final_image_shape, latent_dim, filters, kernel_size, strides, pool_size)
        
    def call(self, inputs):
        x = inputs
        x = self.encoder(x)
        x = self.decoder(x)
        
        return x

