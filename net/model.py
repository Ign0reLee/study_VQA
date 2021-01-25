import tensorflow as tf
import tensorflow.keras as k

from tensorflow.keras import layers
from tensorflow.keras.models import Model


class VQA(Model):

    def __init__(self,tx_size, em_size=1024, features=4096, rate=0.5):
        super(VQA, self).__init__(name="VQA")

        self.text_process = TextProcessing(tx_size,em_size, features, rate)
        
    def call(self, inputs):
        question, image = inputs
        text = self.text_process(question)

class TextProcessing(Model):

    def __init__(self, tx_size, em_size=1024,features = 4096, rate=0.5):
        super(TextProcessing, self).__init__(name="TextProcessing")

        self.init = k.initializers.GlorotUniform()

        self.text_embedding = layers.Embedding(input_dim=tx_size, output_dim=em_size,embeddings_initializer=self.init,mask_zero=True)
        self.dropout = layers.Dropout(rate=rate)
        self.tanh = layers.Activation(tf.nn.tanh)
        self.lstm = layers.LSTM(units = features )

    def call(self, inputs):
        
        x = self.text_embedding(inputs)
        x = self.dropout(x)
        x = self.tanh(x)
        x = self.lstm(x)

        return x


