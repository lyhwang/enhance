from __future__ import print_function, division
import scipy

from keras.datasets import mnist
from keras_contrib.layers.normalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import Subtract, Add
from calculate import Divide, Multiply
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import datetime
import matplotlib.pyplot as plt
import sys
import math
from data_loader import DataLoader
import numpy as np
import cv2
import math
import os
from keras import backend as K
import tensorflow as tf
import argparse

class GAN():
    def __init__(self):
        # Input shape
        self.img_rows = 512
        self.img_cols = 512
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        # Configure data loader
        self.dataset_name = 'dark'
        self.data_loader = DataLoader(dataset_name=self.dataset_name,
                                      img_res=(self.img_rows, self.img_cols))

        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 2**4)
        self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 32
        self.df = 64

        # Loss weights
        self.lambda_valid = 1
        self.lambda_cycle = 10.0                    # Cycle-consistency loss
        self.lambda_tvA = 0.0005
        self.lambda_tvB = 0.0005
        self.lambda_hA = 5
        self.lambda_hB = 5

        self.epsilon = 0.
        self.alpha = 0.8
        self.beta = 0.2

        optimizer = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999)

        # Build and compile the discriminators
        self.d_I = self.build_discriminator()
        self.d_I.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generators
        self.g_A = self.build_generator()
        self.g_B = self.build_generator()

        # Input images from both domains
        img_I = Input(shape=self.img_shape)

        # Translate images to the other domain
        gA_H = self.g_A(img_I)
        gen_J = Divide()([img_I, gA_H])

        # Translate images back to original domain
        gB_H = self.g_B(gen_J)

        reconstr_I = Multiply()([gen_J, gB_H])

        # For the combined model we will only train the generators
        self.d_I.trainable = False

        # Discriminators determines validity of translated images
        valid_I = self.d_I(reconstr_I)

        # Combined model trains generators to fool discriminators
        self.combined = Model(inputs=[img_I],
                              outputs=[ valid_I,
                                        reconstr_I,
                                        gA_H,
                                        gB_H,
                                        gA_H,
                                        gB_H
                                        ])
        self.combined.compile(loss=['mse',
                                    'mae',
                                    self.total_variation_loss,
                                    self.total_variation_loss,
                                    'mae',
                                    'mae',
                                    ],
                            loss_weights=[ self.lambda_valid,
                                           self.lambda_cycle,
                                           self.lambda_tvA,
                                           self.lambda_tvB,
                                           self.lambda_hA,
                                           self.lambda_hB
                                           ],
                            optimizer=optimizer)

    def build_generator(self):
        """U-Net Generator"""

        def conv2d(layer_input, filters, f_size=4):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            d = InstanceNormalization()(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = InstanceNormalization()(u)
            u = Concatenate()([u, skip_input])
            return u

        # Image input
        d0 = Input(shape=self.img_shape)

        # Downsampling
        d1 = conv2d(d0, self.gf)
        d2 = conv2d(d1, self.gf*2)
        d3 = conv2d(d2, self.gf*4)
        d4 = conv2d(d3, self.gf*8)

        # Upsampling
        u1 = deconv2d(d4, d3, self.gf*4)
        u2 = deconv2d(u1, d2, self.gf*2)
        u3 = deconv2d(u2, d1, self.gf)

        u4 = UpSampling2D(size=2)(u3)
        output_img = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u4)

        return Model(d0, output_img)

    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=4, normalization=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if normalization:
                d = InstanceNormalization()(d)
            return d

        img = Input(shape=self.img_shape)

        d1 = d_layer(img, self.df, normalization=False)
        d2 = d_layer(d1, self.df*2)
        d3 = d_layer(d2, self.df*4)
        d4 = d_layer(d3, self.df*8)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

        return Model(img, validity)

    def predict(self, path):
        output_path = "./outputs/"

        self.g_A.load_weights('generatorAA')
        self.g_B.load_weights('generatorBB')
        self.d_I.load_weights('discriminatorD')

        for filename in os.listdir(path):
            imgs_I = self.data_loader.load_lowlightimg(path + filename)[2]
            imgwidth = self.data_loader.load_lowlightimg(path + filename)[0]
            imgheight = self.data_loader.load_lowlightimg(path + filename)[1]

            gA_H = self.g_A.predict(imgs_I)
            gA_H = self.rgb2gray(gA_H)

            H = (gA_H * 0.5 + 0.5) * self.alpha + self.beta

            A = self.getAtmLight(imgs_I, gA_H)

            gen_J = np.true_divide(np.subtract(imgs_I * 0.5 + 0.5, A), H) + A + self.epsilon
            gen_J = gen_J * 2 - 1

            gB_H = self.g_B.predict(gen_J)
            gB_H = self.rgb2gray(gB_H)
            H = (gB_H * 0.5 + 0.5) * self.alpha + self.beta

            A = self.getAtmLight(gen_J, gB_H)

            reconstr_I = np.add(np.multiply(gen_J * 0.5 + 0.5 - A - self.epsilon, H), A)
            reconstr_I = reconstr_I * 2 - 1

            outputs, ax = plt.subplots(figsize=(imgwidth / 100.0, imgheight / 100.0))
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.axis('off')
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0, 0)
            dehazed = np.zeros((self.img_shape))
            dehazed[:, :, :] = gen_J[0, :, :, :] * 0.5 + 0.5
            dehazed = cv2.resize(dehazed, (imgwidth, imgheight))
            ax.imshow(dehazed, aspect='equal')
            outputs.savefig(output_path + filename)
            plt.close()

    def total_variation_loss(self, y, x):
        img_rows, img_cols = self.img_rows, self.img_cols
        a = K.square(
            x[:, :img_rows - 1, :img_cols - 1, :] -
            x[:, 1:, :img_cols - 1, :]
        )
        b = K.square(
            x[:, :img_rows - 1, :img_cols - 1, :] -
            x[:, :img_rows - 1, 1:, :]
        )

        c = K.square(
            y[:, :img_rows - 1, :img_cols - 1, :] -
            y[:, 1:, :img_cols - 1, :]
        )
        d = K.square(
            y[:, :img_rows - 1, :img_cols - 1, :] -
            y[:, :img_rows - 1, 1:, :]
        )

        return (K.sum(K.pow(a + b, 1.25)) + K.sum(K.pow(c + d, 1.25))) / 2

    def rgb2gray(self, rgb):
        r, g, b = rgb[:, :, :, 0], rgb[:, :, :, 1], rgb[:, :, :, 2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

        img = np.expand_dims(cv2.resize(gray, (self.img_rows, self.img_cols)), axis=3)

        gray = np.zeros((1, self.img_rows, self.img_cols, 3))
        for i in range(0, 3):
            gray[0, :, :, i] = img[0, :, :, 0]

        return gray

    def getAtmLight(self, img=None, Gray=None):
        height, width, __ = self.img_rows, self.img_cols, 3
        totalPixels = width * height

        img = img * 0.5 + 0.5
        Gray = Gray * 0.5 + 0.5

        im = np.zeros((self.img_shape))
        im[:, :, :] = img[0, :, :, :]
        Dark = np.zeros((self.img_rows, self.img_cols, 1))
        Dark[:, :, 0] = Gray[0, :, :, 0]

        ImVec = np.reshape(im, (totalPixels, 3))
        indices = np.argsort(np.reshape(Dark, (totalPixels, 1)), axis=0).flatten()
        topPixels = math.floor(totalPixels / 1000.0)
        indices = indices[:topPixels]
        tempAtm = np.zeros((1, 3))

        for ind in range(0, int(topPixels)):
            tempAtm = tempAtm + ImVec[indices[ind], :]

        A = tempAtm / topPixels
        print(A.flatten())

        return A.flatten()


if __name__ == '__main__':
    gan = GAN()
    gan.predict("./inputs/")


