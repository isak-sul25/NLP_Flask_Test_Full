import requests
import json

# API endpoint URL
url = 'http://localhost:5000/sentiment'

# Text to be sent in the request
text = "I'm really excited about this!"
payload = {'text': text}

# Send POST request
response = requests.post(url, json=payload)
print("Text is: " + text)
# Check response status code
if response.status_code == 200:
    # Parse the JSON response
    data = json.loads(response.text)
    print(str(data))

else:
    print('Error occurred during the request.')
