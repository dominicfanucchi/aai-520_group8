import json
import gradio as gr
from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer
import nltk
nltk.download('punkt')

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
            if file is None:
                return 'Please upload a file.'
            
            print(f'File object: {file}') # debug print
            
            # this should now work with Gradio's file handling
            if hasattr(file, 'name'):
                print(f'Reading file from path: {file.name}') # debug print
                with open(file.name, 'r', encoding='utf-8') as f:
                    content = f.read()
            else:
                print(f'Reading file using read() method') # debug print
                content = file.read().decode('utf-8')
            
            self.context = content
            print(f'Context set. Length: {len(self.context)}') # debug print

            tokens = self.tokenizer.encode(self.context)
            # checking token length of loaded files and giving the user a 
            # warning if it exceeds the max
            # the context length can be addressed in future versions of this 
            # chatbot by using other methods to work around DistilBERT's short context length
            if len(tokens) > 512:
                return (f"Warning: The loaded file is {len(tokens)} tokens long, "
                        "which exceeds DistilBERT's maximum context length of 512 tokens. "
                        "The text will be truncated.")
            
            return f'File loaded successfully. Content length: {len(self.context)} characters.'
        
        except FileNotFoundError:
            print('FileNotFoundError occured') # debug print
            return f'Error: The file was not found.'
        
        except Exception as e:
            print(f'Exception occured: {str(e)}') # debug print
            return f'An erroroccured while attempting to read the file: {str(e)}'

    def chat(self, user_input):
        print(f'Chat method called. Context length: {len(self.context)}') # debug print

        if not self.context:
            print('Context is empty') # debug print
            return 'Please load a file before asking questions.'
        
        tokens = self.tokenizer.encode(self.context)
        if len(tokens) > 512:
            truncated_tokens = tokens[:512]
            truncated_context = self.tokenizer.decode(truncated_tokens)
            print(f'Context truncated. New length: {len(truncated_context)}') # debug print
        
        else:
            truncated_context = self.context

        print(f'Asking question: {user_input}') # debug print
        result = self.question_answerer(question=user_input, context=self.context)
        
        print(f"Answer received: {result['answer']}") # debug print
        return result['answer']
    
    def save_conversation(self):
        with open('conversation_history.json', 'w') as f:
            json.dump(self.conversation_history, f)
        return 'Conversation history saved!'
    

def create_gradio_interface(chatbot):
    with gr.Blocks(theme=gr.themes.Ocean()) as demo:
        gr.Markdown("Chatbot")

        with gr.Row():
            with gr.Column(scale=2):
                chatbot_interface = gr.Chatbot(type='messages')
            with gr.Column(scale=1):
                file_input = gr.File(label='Upload File')
                file_output = gr.Textbox(label='File Status')
                # msg = gr.Textbox(placeholder='Type your question here...', container=False, scale=7)
                # clear = gr.ClearButton([msg, chatbot_interface])
                save_btn = gr.Button('Save')
            with gr.Column(scale=7):
                msg = gr.Textbox(placeholder='Type your question here...', container=False)
                clear = gr.ClearButton([msg, chatbot_interface], scale=1)


        file_input.upload(chatbot.load_context, inputs=[file_input], outputs=[file_output])
        msg.submit(lambda m, h: respond(m, h, chatbot), inputs=[msg, chatbot_interface], outputs=[msg, chatbot_interface])
        save_btn.click(chatbot.save_conversation, outputs=[file_output])

    return demo

def respond(message, chat_history, chatbot):
    print(f'Respond function called. Chatbot instance: {chatbot}') # debug print

    bot_message = chatbot.chat(message)
    chat_history.append((message, bot_message))
    chatbot.conversation_history.append({'user': message, 'bot': bot_message})

    if len(chat_history) > 10:
        chat_history = chat_history[-10]
    return "", chat_history

if __name__ == "__main__":
    # possible model folder name change
    model_path = 'my_awesome_qa_model/'
    
    chatbot = Chatbot(model_path)

    demo = create_gradio_interface(chatbot)
    demo.launch()