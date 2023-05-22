from flask import Flask, request, jsonify, render_template, send_file
import requests

from textblob import TextBlob

import csv
import tensorflow as tf
import tensorflow_text as text
import numpy as np
from datetime import datetime




import tensorflow_hub as hub

application = Flask(__name__)
app = application
url = 'http://localhost:5000'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chatbot', methods=['POST'])
def chatbot():
    data = request.get_json()
    user_input = data.get('input', '')
    
    # Call the sentiment analysis route to get the chatbot response
    payload = {'text': user_input}
    response = requests.post(url + '/sentiment', json=payload)
    
    if response.status_code == 200:
        data = response.json()
        sentiment = data['sentiment']
        chatbot_response = get_chatbot_response(sentiment)
    else:
        chatbot_response = "Oops! Something went wrong."
    
    return jsonify({'response': chatbot_response})

def get_chatbot_response(sentiment):
    # Customize the chatbot responses based on sentiment
    return "The senitement is: " + str(sentiment)





LABELS = ['admiration', 'approval', 'annoyance', 'gratitude', 'disapproval', 'amusement', 'curiosity', 'love', 'optimism', 'disappointment', 'joy', 'realization', 'anger']

reloaded_model = tf.saved_model.load("Best_SmallBert")



def save_to_csv(text, prediction):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data = [[timestamp, text, prediction]]

    with open('requests.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)




def predict_line(inp):
    output = reloaded_model(tf.constant([str(inp)]))
    predictions = np.argmax(output, axis=-1)
    predicted_label = LABELS[predictions[0]]
    return predicted_label
 





@app.route('/sentiment', methods=['POST'])
def analyze_sentiment():
    data = request.get_json()
    text = data.get('text', '')
    text = str(text)
    prediction = predict_line(text)

    save_to_csv(text, prediction)

    response = {
        'sentiment': prediction
    }
    return jsonify(response)


@app.route('/all_requests', methods=['GET'])
def get_all_requests():
    with open('requests.csv', 'r') as file:
        reader = csv.reader(file)
        data = list(reader)
    
    response = {
        'requests': data
    }
    return jsonify(response)


@app.route('/download')
def download():
    return send_file('requests.csv', as_attachment=True)

if __name__ == '__main__':
    app.run(threaded=True)
