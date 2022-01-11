from logging import debug
import joblib
import tensorflow as tf
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/', methods = ['POST', 'GET'])
def prediction():
  model1 = joblib.load(open('Text_Generator.model','rb'))
  tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
  # sentence1 = 'Who is the GOAT in the UFC'
  text = request.form['text']
  input_ids1 = tokenizer.encode(text, return_tensors='pt')
  output1 = model1.generate(input_ids1, min_length = 80, max_length = 100, num_beams = 5, no_repeat_ngram_size = 2, early_stopping = True)
  # print(tokenizer.decode(output1[0], skip_special_tokens=True))
  # return render_template('index.html')
  return tokenizer.decode(output1[0], skip_special_tokens=True)

if __name__ == "__main__":
  app.run(debug=True)