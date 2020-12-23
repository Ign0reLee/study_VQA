import json
import itertools

from nltk import word_tokenize
from collections import Counter

def prepare_question(question_data):
    
    questions = [q['question'] for q in question_data['questions']]
    for question in questions:
        question = word_tokenize(question.lower())
        yield question

def prepare_answer(answer_data):
    
    answers = [a['multiple_choice_answer'] for a in answer_data['annotations']]
    for answer in answers:
        answer = word_tokenize(answer.lower())
        yield answer