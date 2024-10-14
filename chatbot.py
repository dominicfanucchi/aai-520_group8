import json
from transformers import pipeline
from datasets import load_dataset

question_answerer = pipeline("question-answering", model="my_awesome_qa_model", device=0)
squad_dataset = load_dataset('squad')


def chat():
    pass

if __name__ == "__main__":
    chat()