import requests

url = "http://127.0.0.1:8000/api/predict/"
path_file_test_audio = "../../../data/raw/2.wav"

with open(path_file_test_audio, 'rb') as f:
    headers = {'Content-Disposition': 'attachment; filename="12.wav"'}
    response = requests.post(url, data=f, headers=headers)
    print(response.json())