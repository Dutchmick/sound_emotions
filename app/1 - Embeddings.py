# Objective
''''
Extract embeddings from the audio files using a pre-trained model
which can be used to classify the emotions in the files into
4 different emotions
'''

# Load required libraries
import os
import numpy as np
import librosa
import torch
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor

# Folder structure
folder_raw = "../data/raw/"
folder_processed = "../data/processed/"
folder_interim = "../data/interim/"

# Function to creat embeddings
def extract_audio_embeddings(
    folder_raw, model_name="facebook/wav2vec2-large-960h-lv60-self"
):
    # Load pretrained model and feature extractor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Wav2Vec2Model.from_pretrained(
        model_name, torch_dtype=torch.float32 if device == "cuda" else torch.float32
    )
    model = model.to(device).eval()
    embeddings_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)

    # Store results
    embeddings = []
    for filename in os.listdir(folder_raw):
        if filename.endswith(".wav"):
            # Load audio data
            audio, sr = librosa.load(
                os.path.join(folder_raw, filename), sr=16000
            )  # Use 16kHz sampling rate

            # Process audio for the pretrained model
            inputs = embeddings_extractor(audio, sampling_rate=sr, return_tensors="pt")
            inputs = {key: tensor.to(device) for key, tensor in inputs.items()}

            # Extract embeddings
            with torch.no_grad():
                outputs = model(**inputs)
                embedding = (
                    outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
                )
            # Append the embedding to the embedding list
            embeddings.append(embedding)

    return np.array(embeddings)

# Run function to extract embeddings from the audio files using Wav2Vec2
embeddings = extract_audio_embeddings(folder_raw)

# Export data
np.save(folder_interim+'embeddings_array.npy', embeddings)