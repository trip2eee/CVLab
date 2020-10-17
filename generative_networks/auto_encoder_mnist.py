"""
@fn auto_encoder_mnist.py
@references
1. Jakub Langr and Vladimir Bok, GANs in Action, Manning Publications, 2019.
2. https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73
"""

import tensorflow as tf
from tensorflow.keras.datasets import mnist
import numpy as np
import cv2

img_width = 28
img_height = 28

batch_size = 100
dim_image = (28 * 28)
dim_latent = 2
dim_intermediate = 256
epochs = 50
epsilon_std = 1.0

"""
Encoder
"""
class Encoder(tf.keras.layers.Layer):
    def __init__(self):
        super(Encoder, self).__init__()

        self.dense_encoding = tf.keras.layers.Dense(dim_intermediate, activation='relu', name='encoder_h')
        self.dense_mean = tf.keras.layers.Dense(dim_latent, name='mean')
        self.dense_var = tf.keras.layers.Dense(dim_latent, name='log_var')
        self.sampling = tf.keras.layers.Lambda(function=self.__sampling, output_shape=(dim_latent))

    def __sampling(self, args):
        z_mean, z_log_var = args        
        epsilon = tf.random.normal(shape=(tf.shape(z_mean)[0], dim_latent), mean=0.0, stddev=epsilon_std)

        return z_mean + tf.exp(z_log_var / 2.0) * epsilon

    def call(self, x):
        x = self.dense_encoding(x)

        # Encode input as a distribution over the latent space.
        z_mean = self.dense_mean(x)
        z_log_var = self.dense_var(x)
        z = self.sampling([z_mean, z_log_var])

        return [z_mean, z_log_var, z]


"""
Decoder
"""
class Decoder(tf.keras.layers.Layer):
    def __init__(self):
        super(Decoder, self).__init__()

        self.dense_decoder = tf.keras.layers.Dense(dim_intermediate, activation='relu', name='decoder_h')
        self.dense_output = tf.keras.layers.Dense(dim_image, activation='sigmoid', name='output')

    def call(self, x):
        x = self.dense_decoder(x)
        x = self.dense_output(x)

        return x

class AutoEncoder(tf.keras.Model):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def call(self, x):

        z_mean, z_log_var, z = self.encoder(x)
        x = self.decoder(z)

        # add regularization term.
        self.add_loss(self.__kl_loss(z_mean, z_log_var))

        return x

    def __kl_loss(self, z_mean, z_log_var):
        # Kullback-Leibler divergence DKL( N(z_mean, z_var) || N(0, I)).
        # https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Multivariate_normal_distributions
        kl = 1.0 + z_log_var - tf.exp(z_log_var) - tf.square(z_mean)
        kl_loss = -0.5 * tf.reduce_sum(kl, axis=-1)
        
        return tf.reduce_mean(kl_loss) / tf.cast(dim_image, tf.float32)


model = AutoEncoder()
model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss=tf.keras.losses.BinaryCrossentropy())

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

if False:
    model.fit(x_train, x_train, shuffle=True, epochs=epochs, batch_size=batch_size)


    model.summary()

    model.save_weights('models/vae')

model.load_weights('models/vae')

x_pred = model.predict(x_test)

x_test = x_test.reshape((-1, img_height, img_width))
x_pred = x_pred.reshape((-1, img_height, img_width))


cv2.namedWindow("input", cv2.WINDOW_NORMAL)
cv2.namedWindow("ouptut", cv2.WINDOW_NORMAL)

for i in range(len(x_pred)):
    cv2.imshow("input", x_test[i])
    cv2.imshow("ouptut", x_pred[i])

    cv2.waitKey(0)


