import requests
import time
import requests

# Define the URL of the Flask application
url = 'http://localhost:5000'

# Example of calling the sentiment analysis route
def analyze_sentiment(text):
    # Prepare the request payload
    payload = {
        'text': text
    }

    # Send a POST request to the sentiment analysis route
    response = requests.post(url + '/sentiment', json=payload)
    
    # Check the response status code
    if response.status_code == 200:
        data = response.json()
        sentiment = data['sentiment']
        print(f'Sentiment: {sentiment}')
    else:
        print('Error occurred during sentiment analysis.')

# Example of calling the get all requests route
def get_all_requests():
    # Send a GET request to the get all requests route
    response = requests.get(url + '/all_requests')
    
    # Check the response status code
    if response.status_code == 200:
        data = response.json()
        all_requests = data['requests']
        for request in all_requests:
            print(request)
    else:
        print('Error occurred while retrieving all requests.')

# Example usage
text_to_analyze = 'This is a sample text for sentiment analysis.'
analyze_sentiment(text_to_analyze)

get_all_requests()


def measure_response_time():
    base_url = 'http://localhost:5000/sentiment'
    text_lengths = [10, 50, 100, 200, 500, 1000]  # Varying lengths of text

    response_times = []
    for length in text_lengths:
        text = 'a' * length  # Create a text with the specified length

        start_time = time.time()

        payload = {
            'text': text
        }

        response = requests.post(base_url, json=payload)

        end_time = time.time()

        response_time = end_time - start_time
        response_times.append(response_time)

        print(f"Text Length: {length}, Response Time: {response_time:.4f} seconds")

    return response_times

response_times = measure_response_time()


def measure_response_time():
    base_url = 'http://localhost:5000/sentiment'
    text_length = 100
    num_requests = 100

    response_times = []
    for _ in range(num_requests):
        text = 'a' * text_length

        start_time = time.time()

        payload = {
            'text': text
        }

        response = requests.post(base_url, json=payload)

        end_time = time.time()

        response_time = end_time - start_time
        response_times.append(response_time)

        print(f"Response Time: {response_time:.4f} seconds")

    return response_times

response_times = measure_response_time()