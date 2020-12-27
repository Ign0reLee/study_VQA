import json
import itertools

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