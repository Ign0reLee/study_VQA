import os,sys, cv2, glob, argparse

import numpy as np
import random as rd
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.keras as k


from net import VQA
from net import pre_vgg
from net import DataPreprocessing
from tensorflow.keras import optimizers
from tensorflow.keras.applications import VGG19 as vgg19
from tensorflow.keras.applications.vgg19 import preprocess_input


# Set GPU Operation Growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Set Layer Name Of vgg 19
layer_name = 'fc2'

# Set argparser 
parser = argparse.ArgumentParser(usage="train.py --question Question Json File --answer Answer Json Files --vocab Output Folder [Option]")
parser.add_argument("-q","--question", type=str, help="Qustion Json File Path")
parser.add_argument("-a","--answer", type=str, help="Answer Json File Path")
parser.add_argument("-v","--vocab", type=str, help="Output Vocab Json File Path")
parser.add_argument("-i","--image", type=str, help="Image File Folder Path")
parser.add_argument("--batch_size", help="Set Batch Size", default=8)
parser.add_argument("--lr", help="Set learning rate", default=1e-4)
parser.add_argument("--lr_decay", help="Set Learning rate decay", default=5e-5)
parser.add_argument("--epoch", help="Set Epoch", default=100)
parser.add_argument("--answer_len", help="Set Max Answer Length", default=3000)


#Parsing
args = parser.parse_args()

question_path = args.question
answer_path   = args.answer
vocab_path    = args.vocab
image_path    = args.image
batch_size    = int(args.batch_size)
lr            = float(args.lr)
lr_decay      = float(args.lr_decay)
epochs        = int(args.epoch)
answer_len    = int(args.answer_len)

# Define Function

# Set train_step()
"""
@tf.function
def train_step():
    with tf.GradientTape() as tape:
        output = model()
"""

@tf.function
Data = DataPreprocessing(question_path, answer_path, vocab_path, image_path, answer_len, layer_name)
Model = VQA(Data.question_max_len)

lr_schedule = optimizers.schedules.InverseTimeDecay(lr, decay_steps=1, decay_rate=lr_decay, staircase=False)
optimizer = optimizers.Adam(learning_rate=lr_schedule)


def main():

    for epoch in epochs:
        Data.shuffle()

        train_question, train_answer, train_img = Data.make_batch(batch_size=batch_size)
        
        

if __name__ == '__main__':
    main()
