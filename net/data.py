import json, os, cv2
import itertools
import numpy as np
import random as rd

from nltk import word_tokenize
from collections import Counter


def Json_Load(json_path):
    with open(json_path, 'r') as json_file:
        json_data = json.load(json_file)
    return json_data

def prepare_question(question_data):
    """
    Prepare MS COCO VQA V1 Question Data
    """
    
    questions = [q['question'] for q in question_data['questions']]
    for question in questions:
        question = word_tokenize(question.lower())
        yield question

def prepare_answer(answer_data):
    """
    Prepare MS COCO VQA V1 Annotation Data
    """
    
    answers = [a['multiple_choice_answer'] for a in answer_data['annotations']]
    for answer in answers:
        answer = word_tokenize(answer.lower())
        yield answer

def prepare_image(question_data):
    """
    Prepare MS COCO VQA V1 Image Number
    """
    
    image_ids = [q['image_id'] for q in question_data['questions']]
    for image_id in image_ids:
        yield image_id

def Check_Data(questions, answers):
    """
    Check right Pair Data Input
    """
    qa_pairs = list(zip(questions['questions'],answers['annotations']))
    assert all(q['question_id'] == a['question_id'] for q, a in qa_pairs), 'Questions not aligned with answer'
    assert all(q['image_id'] == a['image_id'] for q, a in qa_pairs), 'Image id of question and answer don\'t match'
    assert questions['data_type'] == answers['data_type'], 'Mismatched data types'
    assert questions['data_subtype'] == answers['data_subtype'], 'Mismatched data subtypes'

def make_max_length(data):
    """
    Find Max Length of data
    """
    return max(map(len, data))



class DataPreprocessing():
    
    def __init__(self, q_path, a_path, v_path, i_path ,answer_len=3000):

        question_data = Json_Load(q_path)
        answer_data   = Json_Load(a_path)
        vocab_data    = Json_Load(v_path)
        self.image_path = i_path

        Check_Data(question_data, answer_data)

        self.question_dict = vocab_data['question']
        self.answer_dict   = vocab_data['answer']

        questions = list(prepare_question(question_data))
        image_ids = prepare_image(question_data)
        answers = prepare_answer(answer_data)    

        self.qa_pairs = list(zip(questions, answers, image_ids))
        self.qa_size = len(self.qa_pairs)
        self.question_max_len = make_max_length(questions)
        self.answer_max_len = answer_len

    def shuffle(self):
        rd.shuffle(self.qa_pairs)

    def _encode_question(self, question):

        onehot = np.zeros(self.question_max_len, dtype=np.int32)
        for i, token in enumerate(question):
            onehot[i] = self.question_dict.get(token)
        return onehot

    def _encode_answer(self, answer):

        onehot = np.zeros(self.answer_max_len, dtype=np.int32)
        for i, token in enumerate(answer):
            onehot[i] = self.answer_dict.get(token)
        return onehot
    
    def _image_path_maker(self, image_id=int, default_file_name = "COCO_train2014_", zero_size=12, file_format = ".jpg"):
        
        id_len = len(image_id)
        zero_size -= id_len
        image_name = "0" * zero_size + str(image_id)

        return os.path.join(self.image_path, default_file_name + image_name + file_format)

    def _image_loader(self, image_path):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img
        

    def make_batch(self, batch_size=8):

        for i in range(0, self.qa_size, batch_size):

            if i+batch_size > self.qa_size:

                question = []
                answer   = []
                image    = []

                for q, a, ids in self.qa_pairs[i:]:
                    path = self._image_path_maker(ids)
                    question.append(self._encode_question(q))
                    answer.append(self._encode_answer(a))
                    image.append(self._image_loader(path))

                yield question, answer, image

            else:

                question = []
                answer   = []
                image    = []

                for q, a, ids in self.qa_pairs[i:i+batch_size]:
                    path = self._image_path_maker(ids)
                    question.append(self._encode_question(q))
                    answer.append(self._encode_answer(a))
                    image.append(self._image_loader(path))

                yield question, answer, image


