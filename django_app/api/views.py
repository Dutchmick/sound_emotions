import librosa
import torch
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import FileUploadParser
import pickle

# Load model and initialize components
model_path = "../models/latest_model.pkl"
wav2vec_model_name = "facebook/wav2vec2-large-960h-lv60-self"

# Load the trained RandomForest model
with open(model_path, "rb") as f:
    trained_model = pickle.load(f)

# Load Wav2Vec2 model and feature extractor
device = "cuda" if torch.cuda.is_available() else "cpu"
wav2vec_model = Wav2Vec2Model.from_pretrained(
    wav2vec_model_name, torch_dtype=torch.float32 if device == "cuda" else torch.float32
)
wav2vec_model = wav2vec_model.to(device).eval()
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(wav2vec_model_name)


# Extract audio embeddings from a given file using Wav2Vec2
def extract_audio_embeddings(file):
    audio, sr = librosa.load(file, sr=16000)  # 16kHz sampling rate
    inputs = feature_extractor(audio, sampling_rate=sr, return_tensors="pt")
    inputs = {key: tensor.to(device) for key, tensor in inputs.items()}

    with torch.no_grad():
        outputs = wav2vec_model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return embedding


# API endpoint for predicting emotion from an uploaded audio file
class EmotionPredictionView(APIView):
    parser_classes = [FileUploadParser]

    def post(self, request, *args, **kwargs):
        # Check if file is in the request
        if "file" not in request.data:
            return Response({"error": "No file uploaded."}, status=400)

        file = request.data["file"]

        # Process the uploaded file to extract embeddings
        try:
            embeddings = extract_audio_embeddings(file)
            embeddings = embeddings.reshape(
                1, -1
            )  # Ensure correct shape for prediction

            # Predict emotion
            prediction = trained_model.predict(embeddings)
            emotion = prediction[0]
            return Response({"emotion": emotion}, status=200)

        except Exception as e:
            return Response({"error": str(e)}, status=500)