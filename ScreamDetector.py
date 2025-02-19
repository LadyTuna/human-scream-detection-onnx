# scream_detection.py

import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import onnxruntime as ort
#from IPython.display import display, Audio
import os

class ScreamDetector:
    def __init__(self, model_name='scream_detection_model.onnx'):
        """
        Initialize the ScreamDetector class.
        :param model_name: Path to the ONNX model file.
        """
        self.model_name = model_name
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit([0, 1])  # 0 for non-scream, 1 for scream
        self.onnx_model = self._load_onnx_model()

    def _load_onnx_model(self):
        """
        Load the ONNX model.
        :return: ONNX runtime inference session.
        """
        return ort.InferenceSession(self.model_name)

    def extract_features(self, file_path, mfcc=True, chroma=True, mel=True):
        """
        Extract audio features using librosa.
        :param file_path: Path to the audio file.
        :param mfcc: Whether to extract MFCC features.
        :param chroma: Whether to extract chroma features.
        :param mel: Whether to extract mel spectrogram features.
        :return: Extracted features as a numpy array.
        """
        y, sr = librosa.load(file_path, mono=True)
        features = []
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
            features.extend(mfccs)
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
            features.extend(chroma)
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr), axis=1)
            features.extend(mel)
        return np.array(features)

    def predict_audio(self, file_path):
        """
        Predict whether an audio file contains a scream or not.
        :param file_path: Path to the audio file.
        :return: Predicted label ("Scream" or "Non-Scream").
        """
        # Extract features from the audio file
        feature = self.extract_features(file_path)
        feature = feature.reshape(1, -1)  # Reshape for model input

        # Run the ONNX model
        outputs = self.onnx_model.run(None, {"args_0": feature})

        # Display the audio for playback
        #display(Audio(file_path))

        # Handle empty outputs
        if len(outputs) == 0:
            return "Unknown"

        # Decode the predicted label
        predicted_label = self.label_encoder.inverse_transform([np.argmax(outputs)])
        return predicted_label[0]
    

    def predict_audio_dir(self, directory):
        """
        Predict labels for all audio files in a directory.
        :param directory: Path to the directory containing audio files.
        :return: A list of dictionaries with file names and predicted labels.
        """
        outputs = []

        label_mapping = {0: "Scream Not Detected", 1: "Scream Detected"}
        # Loop over all files in the directory
        for file_name in os.listdir(directory):
            file_path = os.path.join(directory, file_name)
            # Check if it's an audio file (for example, with .wav extension)
            if file_path.endswith(".wav"):
                # Predict the label for the audio file
                predicted_label = self.predict_audio(file_path)

                string_label = label_mapping.get(predicted_label, "Unknown") 

                # Append as a dictionary instead of a tuple
                outputs.append({"Audio_path": directory+'/'+file_name, "prediction": string_label})  
        return outputs


def main():
    # Initialize the scream detector
    detector = ScreamDetector()

    # Test 1: Non-scream audio
    audio_file_to_test = "Assets/testing/n1.wav"
    predicted_label = detector.predict_audio(audio_file_to_test)
    print("Test 1 label:", predicted_label)

    # Test 2: Scream audio
    audio_file_to_test = "Assets/testing/p1.wav"
    predicted_label = detector.predict_audio(audio_file_to_test)
    print("Test 2 label:", predicted_label)

    # Test 3: Scream audios in a dir
    audio_file_to_test = "Assets/testing"
    predicted_label = detector.predict_audio_dir(audio_file_to_test)
    print("Test 3 labels:", predicted_label)


if __name__ == "__main__":
    main()