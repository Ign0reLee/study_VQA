import tensorflow as tf
import tensorflow.keras as k

from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG19 as vgg19
from tensorflow.keras.applications.vgg19 import preprocess_input

def pre_vgg(layers):
    vgg = vgg19(weights="imagenet")
    vgg.trainable = False
    outputs = vgg.get_layer(layers).output
    model = Model(vgg.input, outputs)
    return model


class VQA(Model):

    def __init__(self,tx_size, em_size=1024, features=4096, rate=0.5):
        super(VQA, self).__init__(name="VQA")

        self.text_process = TextProcessing(tx_size,em_size, features, rate)
        self.image_process = ImageProcessing()
        
    def call(self, inputs):
        
        question, image = inputs
        text = self.text_process(question)
        image = self.image_process(image)

class TextProcessing(Model):

    def __init__(self, tx_size, em_size=1024,features = 2048, rate=0.5):
        super(TextProcessing, self).__init__(name="TextProcessing")

        self.init = k.initializers.GlorotUniform()

        self.text_embedding = layers.Embedding(input_dim=tx_size, output_dim=em_size,embeddings_initializer=self.init,mask_zero=True)
        self.dropout = layers.Dropout(rate=rate)
        self.tanh = layers.Activation(tf.nn.tanh)
        self.lstm = layers.LSTM(units = features)
        self.lstm2 = layers.LSTM(units = features)
        self.dense = layers.Dense(1024, activation=tf.nn.tanh)

    def call(self, inputs):
        
        x = self.text_embedding(inputs)
        x = self.dropout(x)
        x = self.tanh(x)
        x = self.lstm(x)
        x = self.dropout(x)
        x = self.tanh(x)
        x = self.lstm2(x)
        x = self.dense(x)

        return x

class ImageProcessing(Model):

    def __init__(self):
        super(ImageProcessing,self).__init__(name="ImageProcessing")

        self.dense = layers.Dense(units=1024, activation= "relu")

    def call(self, inputs):

        x = tf.nn.l2_normalize(inputs)
        x = self.dense(x)
        
        return x


