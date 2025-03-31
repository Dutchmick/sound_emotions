'''
Module provides functionality to extract numerical embeddings 
from audio files using a pre-trained Wav2Vec2 model. 
These embeddings can then be used for downstream tasks like 
classifying the emotion expressed in the audio (e.g. Happy, Sad, Angry, Calm).
'''

import logging
import torch
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor

def load_model_and_extractor(model_name, device):
    """
    Load pretrained Wav2Vec2 model and feature extractor.
    Args:
        model_name: Hugging Face model ID (e.g., 'facebook/wav2vec2-base').
        device: 'cuda' or 'cpu'. Auto-detects if None.
    Returns:
        (model, extractor)
    """
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = Wav2Vec2Model.from_pretrained(
            model_name, torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
        model = model.to(device).eval()
        embeddings_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        return model, embeddings_extractor
    except Exception as e:
        logging.error("Model loading failed: %s", e)
        raise

def extract_embeddings(model, extractor, audio, sampling_rate, device):
    """
    Extracts mean-pooled embeddings from audio.
    Args:
        audio: Input waveform (1D array) or list of waveforms.
        sampling_rate: Must match Wav2Vec2's expected rate (16kHz).
    """
    try: 
        inputs = extractor(
            audio, 
            sampling_rate=sampling_rate, 
            return_tensors="pt",
            padding=True)
        inputs = {key: tensor.to(device) for key, tensor in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        return embedding
    except Exception as e:
        logging.error("Embedding extraction failed: %s", e)
        raise
