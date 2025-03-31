import logging
import os
import numpy as np
import yaml
import torch
from src.dataloader import load_and_resample_audio
from src.embedding import extract_embeddings, load_model_and_extractor
from src.labelling import label_creation
from src.modelling import modelling_workflow
from src.evaluate import evaluate_model

# Load configuration from YAML file
with open("sound_emotions/config.yaml", "r") as f:
    config = yaml.safe_load(f)

folder_raw = config["data"]["folder_raw"]

def setup_logging(log_file):
    """Configures logging settings.

    Args:
        log_file (str): Path to the log file where logs will be written.

    This function ensures that log messages are stored in a file and
    follow a consistent format.
    It removes any existing logging handlers to avoid duplicate log entries.
    """
    # Remove any existing handlers to prevent duplicate log entries
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

def run_pipeline(config):
    setup_logging(config["logging"]["log_file"])
    folder_raw = config["data"]["folder_raw"]
    folder_interim = config["data"]["folder_interim"]
    folder_models = config["data"]["folder_models"]
    sample_rate = config["data"]["sample_rate"]
    model_name = "facebook/wav2vec2-large-960h-lv60-self"
    """Main function to extract embeddings with error handling."""

    if not os.path.exists(folder_raw):
        logging.error(f"Input folder '{folder_raw}' does not exist.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        logging.info("Loading model and feature extractor...")
        
        # load model
        model, extractor = load_model_and_extractor(model_name, device)
        
        # Create embeddings from pretrained model
        embeddings = []
        for filename in os.listdir(folder_raw):
            if filename.endswith(".wav"):
                filepath = os.path.join(folder_raw, filename)
                try:
                    logging.info(f"Processing file: {filename}")
                    audio, _ = load_and_resample_audio(filepath, sample_rate)
                    embedding = extract_embeddings(
                        model, extractor, audio, sample_rate, device
                    )
                    embeddings.append(embedding)
                except Exception as e:
                    logging.error(f"Error processing file {filename}: {e}")
                    continue
        embeddings_array = np.array(embeddings)
        logging.info(f"Successfully processed {len(embeddings)} files.")

        # Label creation
        logging.info("Creating labelled dataset")
        labels = label_creation(embeddings_array, folder_interim)

        # Model creation
        logging.info("Creating model")
        model, X_test, y_test = modelling_workflow(embeddings, labels, folder_models)

        # Evaluate model
        logging.info("Evaluating model performance")
        evaluate_model(model, X_test, y_test)

        logging.info("Pipeline execution completed successfully")
    except Exception as e:
        logging.error("Error in pipeline execution: %s", e)
        raise

# Run function to extract embeddings from the audio files using Wav2Vec2
try:
    run_pipeline(config)
except Exception as e:
    print(f"An error occurred: {e}")
