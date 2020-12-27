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

    def __init__(self):
        super(VQA, self).__init__(name="VQA")
    
    def call(self):


