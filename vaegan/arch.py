import numpy as np

import tensorflow as tf

from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Lambda, Reshape, Layer, BatchNormalization, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

INPUT_DIM = (64,64,3)

CONV_FILTERS = [64,128,256]
CONV_KERNEL_SIZES = [5,5,5]
CONV_STRIDES = [2,2,2]
CONV_ACTIVATIONS = ['relu', 'relu', 'relu']

DENSE_SIZE = 1024

CONV_T_FILTERS = [64,64,32,3]
CONV_T_KERNEL_SIZES = [5,5,6,6]
CONV_T_STRIDES = [2,2,2,2]
CONV_T_ACTIVATIONS = ['relu','relu','relu','tanh']

Z_DIM = 32

GEN_DIM = (8, 8, 256)
GEN_SIZE = np.prod(GEN_DIM)

D_CONV_FILTERS = [32,128,256,256]
D_CONV_KERNEL_SIZES = [5,5,5,5]
D_CONV_STRDIES =  [2,2,2,2]
D_CONV_ACTIVATIONS = ['relu', 'relu', 'relu', 'relu']
DISC_SIZE = 256

BATCH_SIZE = 100
BUFFER_SIZE = 10000
LEARNING_RATE = 0.0001
KL_TOLERANCE = 0.5



class Sampling(Layer):
    def call(self, inputs):
        mu, log_var = inputs
        epsilon = K.random_normal(shape=K.shape(mu), mean=0., stddev=1.)
        return mu + K.exp(log_var / 2) * epsilon

# Define helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output, random_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    random_loss = cross_entropy(tf.zeros_like(random_output), random_output)
    total_loss = (0.5 * real_loss) + (0.25 * fake_loss) + (0.25 * random_loss)
    return total_loss



class VAEGANModel(Model):
    def __init__(self, encoder, generator, discriminator, **kwargs):
        super(VAEGANModel, self).__init__(**kwargs)
        self.encoder = encoder
        self.generator = generator
        self.discriminator = discriminator

        self.encoder_opt = Adam(lr=LEARNING_RATE)
        self.generator_opt = Adam(lr=LEARNING_RATE)
        self.discriminator_opt = Adam(lr=LEARNING_RATE)

        self.normal_loss_coef = 0.1
        self.kl_loss_coef = 0.01
    
    @tf.function
    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]

        random = tf.random.normal((BATCH_SIZE, Z_DIM))
        with tf.GradientTape(persistent=True) as tape:
            # Get model outpus
            z_mean, z_log_var, z = self.encoder(data)
            generator_output = self.generator(z)
            disc_inner_real, disc_output_real = self.discriminator(data)
            disc_inner_fake, disc_output_fake = self.discriminator(generator_output)
            disc_inner_random, disc_output_random = self.discriminator(self.generator(random))

            # Compute inner output loss
            inner_output_diff = disc_inner_fake - disc_inner_real
            inner_output_loss = tf.reduce_mean(tf.square(inner_output_diff))

            # Compute normal loss
            mean, var = tf.nn.moments(z, axes=0)
            normal_loss = tf.reduce_mean(tf.square(mean)) + tf.reduce_mean(tf.square(var - 1))

            # Compute KL loss
            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_sum(kl_loss, axis=-1) * -0.5
            kl_loss = tf.reduce_mean(kl_loss)

            # Compute discriminator loss
            disc_loss = discriminator_loss(disc_output_real, disc_output_fake, disc_output_random)

            # Define encoder loss
            enc_loss = inner_output_loss + (kl_loss * self.kl_loss_coef) + (normal_loss * self.normal_loss_coef)

            # Define generator loss
            gen_loss = inner_output_loss - disc_loss

        # Compute gradients
        enc_grads = tape.gradient(enc_loss, self.encoder.trainable_variables)
        gen_grads = tape.gradient(gen_loss, self.generator.trainable_variables)
        disc_grads = tape.gradient(disc_loss, self.discriminator.trainable_variables)

        del tape

        # Apply gradients
        self.encoder_opt.apply_gradients(zip(enc_grads, self.encoder.trainable_variables))
        self.generator_opt.apply_gradients(zip(gen_grads, self.generator.trainable_variables))
        self.discriminator_opt.apply_gradients(zip(disc_grads, self.discriminator.trainable_variables))

        return {
            "inner_output_loss": inner_output_loss,
            "normal_loss": normal_loss,
            "kl_loss": kl_loss,
            "encoder_loss": enc_loss,
            "generator_loss": gen_loss,
            "discriminator_loss": disc_loss,
        }
    
    def call(self,inputs):
        latent = self.encoder(inputs)
        return self.generator(latent)



class VAEGAN():
    def __init__(self):
        self.models = self._build()
        self.encoder = self.models[0]
        self.generator = self.models[1]
        self.discriminator = self.models[2]
        self.full_model = self.models[3]

        self.input_dim = INPUT_DIM
        self.z_dim = Z_DIM
        self.learning_rate = LEARNING_RATE
        self.kl_tolerance = KL_TOLERANCE

    def _build(self):
        vae_x = Input(shape=INPUT_DIM, name='observation_input')

        # Encoder block 1: conv -> batch norm -> relu
        vae_c1 = Conv2D(filters=CONV_FILTERS[0], kernel_size=CONV_KERNEL_SIZES[0], strides=CONV_STRIDES[0], padding='same', name='conv_layer_1')(vae_x)
        vae_bn1 = BatchNormalization(name='bn_layer_1')(vae_c1)
        vae_act1 = Activation(CONV_ACTIVATIONS[0], name='act_layer_1')(vae_bn1)

        # Encoder block 2
        vae_c2 = Conv2D(filters=CONV_FILTERS[1], kernel_size=CONV_KERNEL_SIZES[1], strides=CONV_STRIDES[1], padding='same', name='conv_layer_2')(vae_act1)
        vae_bn2 = BatchNormalization(name='bn_layer_2')(vae_c2)
        vae_act2 = Activation(CONV_ACTIVATIONS[1], name='act_layer_2')(vae_bn2)

        # Encoder block 3
        vae_c3 = Conv2D(filters = CONV_FILTERS[2], kernel_size = CONV_KERNEL_SIZES[2], strides = CONV_STRIDES[2], padding='same', name='conv_layer_3')(vae_act2)
        vae_bn3 = BatchNormalization(name='bn_layer_3')(vae_c3)
        vae_act3 = Activation(CONV_ACTIVATIONS[2], name='act_layer_3')(vae_bn3)

        # Flatten tensor
        vae_z_in = Flatten()(vae_act3)

        # Dense activation block
        vae_z_den = Dense(DENSE_SIZE, name='dense_layer_1')(vae_z_in)
        vae_z_bn = BatchNormalization(name='bn_layer_4')(vae_z_den)
        vae_z_act = Activation('relu', name='act_layer_4')(vae_z_bn)

        # Return mu and log var
        vae_z_mean = Dense(Z_DIM, name='mu')(vae_z_act)
        vae_z_log_var = Dense(Z_DIM, name='log_var')(vae_z_act)
        vae_z = Sampling(name='z')([vae_z_mean, vae_z_log_var])
        

        #### GENERATOR: 
        gen_z_in = Input(shape=(Z_DIM,), name='z_input')

        # Generator dense block
        gen_dense = Dense(DENSE_SIZE, name='gen_dense_layer')(gen_z_in)
        gen_dense_bn = BatchNormalization(name='gen_dense_bn_layer')(gen_dense)
        gen_dense_act = Activation('relu', name='gen_dense_act_layer')(gen_dense_bn)

        # Unflatten tensor
        deconv_in = Reshape((1, 1, DENSE_SIZE), name='unflatten')(gen_dense_act)

        # Generator block 1: deconv -> batch norm -> relu
        gen_d1 = Conv2DTranspose(filters=CONV_T_FILTERS[0], kernel_size=CONV_T_KERNEL_SIZES[0], strides=CONV_T_STRIDES[0], name='gen_deconv_layer_1')(deconv_in)
        gen_bn1 = BatchNormalization(name='gen_bn_layer_1')(gen_d1)
        gen_act1 = Activation(CONV_T_ACTIVATIONS[0], name='gen_act_layer_1')(gen_bn1)

        # Generator block 2
        gen_d2 = Conv2DTranspose(filters=CONV_T_FILTERS[1], kernel_size=CONV_T_KERNEL_SIZES[1], strides=CONV_T_STRIDES[1], name='gen_deconv_layer_2')(gen_act1)
        gen_bn2 = BatchNormalization(name='gen_bn_layer_2')(gen_d2)
        gen_act2 = Activation(CONV_T_ACTIVATIONS[1], name='gen_act_layer_2')(gen_bn2)

        # Generator block 3
        gen_d3 = Conv2DTranspose(filters=CONV_T_FILTERS[2], kernel_size=CONV_T_KERNEL_SIZES[2], strides=CONV_T_STRIDES[2], name='gen_deconv_layer_3')(gen_act2)
        gen_bn3 = BatchNormalization(name='gen_bn_layer_3')(gen_d3)
        gen_act3 = Activation(CONV_T_ACTIVATIONS[2], name='gen_act_layer_3')(gen_bn3)

        # Output reconstructed image
        gen_out = Conv2DTranspose(filters=CONV_T_FILTERS[3], kernel_size=CONV_T_KERNEL_SIZES[3], strides=CONV_T_STRIDES[3], activation=CONV_T_ACTIVATIONS[3], name='reconstructed')(gen_act3)


        #### DISCRIMINATOR:
        disc_in = Input(shape=(INPUT_DIM))

        # Discriminator block 1 (no batch norm)
        disc_c1 = Conv2D(filters=D_CONV_FILTERS[0], kernel_size=D_CONV_KERNEL_SIZES[0], strides=D_CONV_STRDIES[0], padding='same', name='disc_conv_layer_1')(disc_in)
        disc_act1 = Activation(D_CONV_ACTIVATIONS[0], name='disc_act_layer_1')(disc_c1)

        # Discriminator block 2: conv -> batch norm -> relu 
        disc_c2 = Conv2D(filters=D_CONV_FILTERS[1], kernel_size=D_CONV_KERNEL_SIZES[1], strides=D_CONV_STRDIES[1], padding='same', name='disc_conv_layer_2')(disc_act1)
        disc_bn1 = BatchNormalization(name='disc_bn_layer_1')(disc_c2)
        disc_act2 = Activation(D_CONV_ACTIVATIONS[1], name='disc_act_layer_2')(disc_bn1)

        # Discriminator block 3
        disc_c3 = Conv2D(filters=D_CONV_FILTERS[2], kernel_size=D_CONV_KERNEL_SIZES[2], strides=D_CONV_STRDIES[2], padding='same', name='disc_conv_layer_3')(disc_act2)
        disc_bn2 = BatchNormalization(name='disc_bn_layer_2')(disc_c3)
        disc_act3 = Activation(D_CONV_ACTIVATIONS[2], name='disc_act_layer_3')(disc_bn2)

        # Discriminator block 4
        disc_c4 = Conv2D(filters=D_CONV_FILTERS[3], kernel_size=D_CONV_KERNEL_SIZES[3], strides=D_CONV_STRDIES[3], padding='same', name='disc_conv_layer_4')(disc_act3)
        inner_output = Flatten()(disc_c4)
        disc_bn3 = BatchNormalization(name='disc_bn_layer_3')(disc_c4)
        disc_act4 = Activation(D_CONV_ACTIVATIONS[3], name='disc_act_layer_4')(disc_bn3)

        # Flatten tensor to be fed into discriminator decision network
        disc_dense_in = Flatten()(disc_act4)

        # Discriminator decision network
        disc_dense = Dense(DISC_SIZE, name='disc_dense_layer')(disc_dense_in)
        disc_dense_bn = BatchNormalization(name='disc_dense_bn_layer')(disc_dense)
        disc_dense_act = Activation('relu', name='disc_dense_act_layer')(disc_dense_bn)
        disc_out = Dense(1, activation='sigmoid', name='decision')(disc_dense_act)


        #### MODELS:

        encoder = Model(vae_x, [vae_z_mean, vae_z_log_var, vae_z], name = 'encoder')
        generator = Model(gen_z_in, gen_out, name = 'generator')
        discriminator = Model(disc_in, [inner_output, disc_out], name='discriminator')

        full_model = VAEGANModel(encoder, generator, discriminator)
        
        return (encoder, generator, discriminator, full_model)

    def set_weights(self, filepath):
        self.full_model.load_weights(filepath)

    def train(self, data):
        # Batch data
        batched_data = tf.data.Dataset.from_tensor_slices(data).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

        i = 0
        for image_batch in batched_data:
            metrics = self.full_model.train_step(image_batch)
            if i % 100 == 0:
                print(metrics)
            i += 1

        
    def save_weights(self, filepath):
        self.full_model.save_weights(filepath)
