import json
import gradio as gr
from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer
import nltk
nltk.download('punkt')

file_path = 'the haunted how on willow street.txt'
question_path = 'willow street questions.txt'

class Chatbot:
    def __init__(self, model_path):
        # using AutoModel and AutoTokenizer because it detects the correct model
        # and tokenizer from the saved files
        # this should allow us to swap in different models in the future with 
        # only needing to change the `model_path` parameter
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.question_answerer = pipeline('question-answering', model=self.model, tokenizer=self.tokenizer, device=0)
        self.context = ''
        self.conversation_history = []

    def load_context(self, file):
        try:
            self.context = file.decode('utf-8')
            tokens = self.tokenizer.encode(self.context)
            # checking token length of loaded files and giving the user a 
            # warning if it exceeds the max
            # the context length can be addressed in future versions of this 
            # chatbot by using other methods to work around DistilBERT's short context length
            if len(tokens) > 512:
                return f"Warning: The loaded file is {len(tokens)} tokens long, 
                which exceeds DistilBERT's maximum context length of 512 tokens.
                The text will be truncated."
            
            return 'File read successfully.'
        except FileNotFoundError:
            print(f'Error: The file {file_path} was not found.')
        except Exception as e:
            print(f'An erroroccured while attempting to read the file: {str(e)}')

    def chat(self, user_input):
        result = self.question_answerer(question=user_input, context=self.context)
        return result['answer']
    
    def save_conversation(self):
        with open('conversation_history.json', 'w') as f:
            json.dump(self.conversation_history, f)
        return 'Conversation history saved!'

if __name__ == "__main__":
    # possible model folder name change
    model_path = 'my_awesome_qa_model/'
    
    chatbot = Chatbot(model_path)
    chatbot.load_context(file_path=file_path)

    print("Chatbot: Hello! Feel free to ask me questions.")

    # after building a working python script, this code will need to be reworked
    #  to incorporate Gradio
    while True:
        user_input = input("User: ")
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("Chatbot: Goodbye! I hope you enjoyed our conversation.")
            break

        response = chatbot.chat(user_input)
        print(f'Chatbot: {response}')