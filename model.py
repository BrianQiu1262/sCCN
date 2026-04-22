from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, AvgPool2D, Flatten, Dense, GlobalAveragePooling2D
from tensorflow.keras.layers import BatchNormalization
import tensorflow.keras as keras
import tensorflow as tf

HIDDEN = 32


def IRB(inputs, out_channel, kernel):
    x = Conv2D(out_channel, kernel_size=[1, 1], strides=[1, 1], padding='same', kernel_initializer='he_uniform', bias_initializer='lecun_uniform', use_bias=False)(inputs)
    x = DepthwiseConv2D(kernel_size=kernel, strides=[1, 1], padding='same', kernel_initializer='he_uniform', bias_initializer='lecun_uniform', use_bias=False)(x)
    x = BatchNormalization(fused=False)(x)
    x = keras.layers.ReLU(6.)(x)
    x = Conv2D(out_channel, kernel_size=[1, 1], strides=[1, 1], padding='same')(x)
    x = BatchNormalization(fused=False)(x) + inputs

    return x


def MobileNetv2():
    inputs = keras.Input(shape=(4, 10, 64), name='input')
    x = Conv2D(HIDDEN, kernel_size=[3, 3], padding='valid', kernel_initializer='he_uniform', bias_initializer='lecun_uniform')(inputs)
    x = IRB(x, HIDDEN, [3, 3])
    x = IRB(x, HIDDEN, [3, 3])
    x = AvgPool2D(pool_size=[2, 2], strides=[2, 2])(x)
    x = Conv2D(256, kernel_size=1, padding='same', strides=1, activation='relu', kernel_initializer='he_uniform', bias_initializer='lecun_uniform')(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(12, kernel_initializer='he_uniform', bias_initializer='lecun_uniform')(x)
    return keras.Model(inputs, x, name='sCNN')


class sCCN(Model):
    def __init__(self):
        super(sCCN, self).__init__()
        self.sccn = MobileNetv2()

    def call(self, x):
        x = self.sccn(x)
        return x




