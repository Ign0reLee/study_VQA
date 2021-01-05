import os
import json
import argparse
import itertools

from net import data
from collections import Counter



parser = argparse.ArgumentParser(usage="Make_Vocab.py --question Question Json File --answer Answer Json Files --vocab Output Folder [Option]")
parser.add_argument("-q","--question", type=str, help="Qustion Json File Path")
parser.add_argument("-a","--answer", type=str, help="Answer Json File Path")
parser.add_argument("-v","--vocab", type=str, help="Output Vocab Json File Path", default=os.path.join("."))
parser.add_argument("--vocab_name", type=str, help="Output Vocab File Name", default="Vocab.json")
parser.add_argument("--max_answer", type=int, help="Max Answer Size", default=3000)
parser.add_argument("--start", type=int, help="Start Token Size", default=1)
parser.add_argument("--set", action="store_true", help="If want to use setting json file", default=False)
parser.add_argument("-s","--set_path", type=str, help="If using setting json file, Input set.json file path", default=os.path.join(".","set.json"))
parser.add_argument("--set_task", type=str, help="Set Task", choices=["OpenEnded", "Multiple"],default="OpenEnded")
parser.add_argument("--set_mode", type=str, help="Set Mode", default="train")

args = parser.parse_args()

question_path    = args.question
answer_path      = args.answer
vocab_path       = args.vocab
vocab_name       = args.vocab_name
maxanswer        = args.max_answer
start            = args.start
setting          = bool(args.set)
set_path         = args.set_path
set_task         = args.set_task
set_mode         = args.set_mode

if setting:

    if (set_path is None) or not os.path.exists(set_path):
        raise ValueError("Please Input Set Json File Path")

    with open(set_path, 'r') as json_file:
        SetInfo = json.load(json_file)

    question_path = SetInfo[set_task][set_mode]["question"]

    if not (set_mode =="test"):
        answer_path   = SetInfo[set_task][set_mode]["answer"]

if (question_path is None) or not os.path.exists(question_path):
    raise ValueError("Please Input Question Json File Path")
if (set_mode == "test") or (answer_path is None) or not os.path.exists(answer_path) :
    raise ValueError("Please Input Answer Json File Path")
if not os.path.exists(vocab_path):
    raise ValueError("Please Input Output Vocab Json File Path")

vocab_path = os.path.join(vocab_path, vocab_name)

def extract_vocab(data, top_k=None, start=0):
    
    data = itertools.chain.from_iterable(data)
    counter = Counter(data)
    
    if top_k:
        most_common = counter.most_common(top_k)
        most_common = (t for t,c in most_common)
    else:
        most_common = counter.keys()
    
    tokens = sorted(most_common, key=lambda x: (counter[x], x), reverse=True)
    vocab = {t: i for i, t in enumerate(tokens, start= start)}
    return vocab

def main():

    with open(question_path, 'r') as json_file:
        question_data = json.load(json_file)

    with open(answer_path, 'r') as json_file:
        answer_data = json.load(json_file)
    
    questions = data.prepare_question(question_data)
    answers   = data.prepare_answer(answer_data)

    question_vocab = extract_vocab(questions, start=start)
    answer_vocab   = extract_vocab(answers, top_k = maxanswer)

    vocabs ={
        'question' : question_vocab,
        'answer'   : answer_vocab
    }

    with open(vocab_path,'w') as json_file:
        json.dump(vocabs, json_file)

if __name__ == '__main__':
    main()