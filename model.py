import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, Concatenate, Input
from tensorflow.keras.models import Model


#This is the same as old pytorch model
#Comments remain same as before
#Adjust the blocks


class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = Conv2D(out_channels, kernel_size=(3, 3), padding='same', activation='relu')
        self.conv2 = Conv2D(out_channels, kernel_size=(3, 3), padding='same', activation='relu')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        return x

class MixPool(tf.keras.layers.Layer):
    def __init__(self, out_channels):
        super(MixPool, self).__init__()
        # Define layers in MixPool

    def call(self, inputs, masks):
        # Define the forward pass of MixPool
        return output

class EncoderBlock(tf.keras.layers.Layer):
    def __init__(self, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv1 = Conv2D(out_channels, kernel_size=(3, 3), padding='same', activation='relu')
        self.conv2 = Conv2D(out_channels, kernel_size=(3, 3), padding='same', activation='relu')
        self.pool = MaxPooling2D(pool_size=(2, 2))

    def call(self, inputs, masks):
        x = self.conv1(inputs)
        x = self.conv2(x)
        p = x  # Replace with the appropriate operations for MixPool
        o = self.pool(p)
        return o, x

class DecoderBlock(tf.keras.layers.Layer):
    def __init__(self, out_channels):
        super(DecoderBlock, self).__init__()
        self.upsample = Conv2DTranspose(out_channels, kernel_size=(4, 4), strides=(2, 2), padding='same', activation='relu')
        self.conv1 = Conv2D(out_channels, kernel_size=(3, 3), padding='same', activation='relu')
        self.conv2 = Conv2D(out_channels, kernel_size=(3, 3), padding='same', activation='relu')

    def call(self, inputs, skip, masks):
        x = self.upsample(inputs)
        x = Concatenate()([x, skip])
        x = self.conv1(x)
        x = self.conv2(x)
        p = x  # Replace with the appropriate operations for MixPool
        return p

class FANet(Model):
    def __init__(self):
        super(FANet, self).__init__()
        self.e1 = EncoderBlock(32)
        self.e2 = EncoderBlock(64)
        self.e3 = EncoderBlock(128)
        self.e4 = EncoderBlock(256)
        self.d1 = DecoderBlock(128)
        self.d2 = DecoderBlock(64)
        self.d3 = DecoderBlock(32)
        self.d4 = DecoderBlock(16)
        self.output_layer = Conv2D(1, kernel_size=1)

    def call(self, x):
        inputs, masks = x[0], x[1]
        p1, s1 = self.e1(inputs, masks)
        p2, s2 = self.e2(p1, masks)
        p3, s3 = self.e3(p2, masks)
        p4, s4 = self.e4(p3, masks)
        d1 = self.d1(p4, s4, masks)
        d2 = self.d2(d1, s3, masks)
        d3 = self.d3(d2, s2, masks)
        d4 = self.d4(d3, s1, masks)
        d5 = Concatenate()([d4, masks])
        output = self.output_layer(d5)
        return output

if __name__ == "__main__":
    x = tf.random.normal((2, 256, 256, 3))
    m = tf.random.normal((2, 256, 256, 1))
    model = FANet()
    y = model([x, m])
    print(y.shape)
