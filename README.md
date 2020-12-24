# study_VQA

VQA for tensorflow2

[Original Paper](https://arxiv.org/pdf/1505.00468.pdf)(br/)
[Site](https://visualqa.org/)

## Make Vocab

Make_Vocab.py makes Vocab.json files and supports two modes

**First mode**, just input all of file path
```cmd
python Make_Vocab.py --question QuestionFilePath --answer AnswerFilePath [optional]
```
options
- --vocab : Output files path (default: now folder)
- --vocab_name : Output files name (default : Vocab.json)
- --max_answer : maximum answer token count (default : 3000)
- --start : question vocab's starting point (default : 1)

<br/>

**Second mode**, Using set.json file, input file path in that
```cmd
python Make_Vocab.py --set [optional]
```
options
- --set_task : choice task (default : OpenEnded)
- --set_mode : choice file path (default : train)
- --max_answer : maximum answer token count (default : 3000)
- --start : question vocab's starting point (default : 1)