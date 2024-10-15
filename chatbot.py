import json
from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer
from datasets import load_dataset

question_answerer = pipeline("question-answering", model="my_awesome_qa_model", device=0)
squad_dataset = load_dataset('squad')

class Chatbot:
    def __init__(self, model_path):
        # using AutoModel and AutoTokenizer because it detects the correct model and tokenizer from the saved files
        # this should allow us to swap in different models in the future with only needing to change the `model_path` parameter
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.question_answerer = pipeline('question-answering', model=self.model, tokenizer=self.tokenizer, device=0)
        self.context = ''

    def chat(self, user_input):
        result = self.question_answerer(question=user_input, context=self.context)
        return result['answer']

if __name__ == "__main__":
    # possible model folder name change
    model_path = 'my_awesome_qa_model/'
    chatbot = Chatbot(model_path)

    print("Chatbot: Hello! Feel free to ask me questions.")

    # after building a working python script, this code will need to be reworked to incorporate TKinter
    while True:
        user_input = input("User: ")
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("Chatbot: Goodbye! I hope you enjoyed our conversation.")
            break

        response = chatbot.chat(user_input)
        print(f'Chatbot: {response}')