# Emotions from sound

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
││
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         emotions_from_sound and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── emotions_from_sound   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes emotions_from_sound a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

## Emotion Prediction API
This project implements a Django REST API for predicting emotions from audio files. The API leverages the pre-trained Wav2Vec2 model for feature extraction and a Random Forest classifier for emotion prediction. It supports .wav audio files and classifies emotions from the audio recordings into one of four categories: Happy, Sad, Angry, Calm.

## Project Workflow
**Exploratory Data Analysis (EDA)**: Understand the audio data, extract key features, and visualize audio waveforms and MFCCs.
Feature Engineering and Embedding Extraction: Use the Wav2Vec2 model to extract embeddings from audio files.

**Clustering and Manual Labeling**: Apply K-means clustering to group audio files and manually assign emotion labels based on cluster patterns.
Model Training: Train a Random Forest model using the manually labeled data.
API Integration: Deploy the trained model as a REST API using Django.

##Features
**Audio Analysis**: Extract features like MFCCs, zero-crossing rates, and spectral contrast for exploratory analysis.

**Pre-trained Embedding Extraction**: Use Wav2Vec2 for feature extraction.
**Emotion Classification**: Classify emotions with a trained Random Forest model.
REST API: Simple API endpoint for real-time emotion prediction.
Technologies Used
Django: Backend framework for the API.
Django REST Framework (DRF): Simplified API implementation.
Wav2Vec2: Pre-trained audio model from the Hugging Face Transformers library.
Scikit-learn: For clustering and training the Random Forest model.
Librosa: Audio processing and feature extraction library.
Matplotlib: For visualizing audio features.

## Setup Instructions
1. **Prerequisites**
Python 3.8+ installed on your machine.
Ensure pip is updated:
pip install --upgrade pip
2. **Clone the Repository**
git clone <repository-url>
cd emotion-prediction-api
3. **Install Dependencies**
Install the required Python packages:
pip install -r requirements.txt
4. **Configure the Database**
Run the following commands to apply Django migrations:
python manage.py makemigrations
python manage.py migrate
5. **Start the Development Server**
Run the Django development server:
python manage.py runserver
The API will be available at http://127.0.0.1:8000/api/predict/.

## Data Exploration and Modeling
**Exploratory Data Analysis**
The following steps were performed to understand the provided audio data:

Waveform Visualization: Displayed waveforms for initial audio samples to observe silence at the start and end.
MFCC Visualization: Plotted Mel-frequency cepstral coefficients (MFCCs) to understand feature patterns.
Feature Extraction: Extracted useful audio features such as:
MFCCs
Zero-crossing rate (ZCR)
Spectral contrast
Chromagram

**Key insights:**
- The audio files are clean with no background noise.
- Each file contains the same phrase, "Kids are talking by the door," making it easier to isolate emotions.
* All files are stereo and share a consistent sample rate.
* Feature Engineering and Embedding Extraction
* Used the Wav2Vec2 model to generate embeddings from audio files.

**Clustering and Manual Labeling**
- Applied K-means clustering to group audio files based on Wav2Vec2 embeddings.
- Plotted the clusters using PCA for dimensionality reduction.
- Manually labeled the clusters with emotion labels (Happy, Sad, Angry, Calm).

**Model Training**
- Split the data into training and testing sets using stratified sampling.
- Trained a Random Forest classifier with hyperparameter tuning via grid search.
- Achieved a classification accuracy of ~90% on test data.

**API Endpoint**
Description: Accepts an audio file and returns the predicted emotion.
Parameters:
file (form-data): .wav audio file.
Response: JSON object with the predicted emotion.

curl -X POST -F "file=@path_to_audio.wav" http://127.0.0.1:8000/api/predict/

Using Python
import requests

url = "http://127.0.0.1:8000/api/predict/"
path_to_audio = "path/to/audio.wav"
files = {'file': ('audio.wav', open(path_to_audio, 'rb'), 'audio/wav')}

response = requests.post(url, files=files)
print(response.json())

Testing
import requests

url = "http://127.0.0.1:8000/api/predict/"
path_file_test_audio = "../../../data/raw/2.wav"

with open(path_file_test_audio, 'rb') as f:
    headers = {'Content-Disposition': 'attachment; filename="12.wav"'}
    response = requests.post(url, data=f, headers=headers)
    print(response.json())

Use curl, Postman, or the Python script (described above) to test the API manually.

## Future Improvements
- Improve emotion prediction accuracy with fine-tuned Wav2Vec2 models.
- Django to automatically take the latest model.
- Add unit testing, de and other tests
- Potentially add features from EDA to improve model performance.
- Create API front-end
- Automatically move processed audio files to a separate folder
- Automate data labeling for real-world datasets.

## License
This project is licensed under the MIT License. See LICENSE for more details.