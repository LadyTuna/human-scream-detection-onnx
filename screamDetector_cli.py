'''
scremDetector_cli.py

This code is to run the Scream Detection model on the terminal 
using command line inputs. The correct input format is

python screamDetector_cli.py --input_dir <input files directory> --output_dir <output destination directory>
'''
import onnxruntime as ort
import numpy as np
from ScreamDetector import ScreamDetector
import argparse
from pathlib import Path
import csv
from pprint import pprint

def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert numpy array to a regular list
    elif isinstance(obj, np.generic):
        return obj.item()  # Convert numpy scalar to a native Python type (e.g., int, float)
    return obj  # If it's a regular type, just return it as is

# Arguments
parser = argparse.ArgumentParser(
    description="Run the Scream Detection model on audio files in the INPUT_DIR and save the predictions to a CSV file."
)
parser.add_argument("--input_dir", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
args = parser.parse_args()

input_dir = Path(args.input_dir)
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

# Initialize the scream detection model
model = ScreamDetector('scream_detection_model.onnx')

# Get predictions
outputs = model.predict_audio_dir(input_dir)
pprint(outputs)

# Convert all outputs to serializable objects
serializable_outputs = [convert_to_serializable(output) for output in outputs]

# Write to CSV file
csv_file_path = output_dir / "output.csv"
with open(csv_file_path, mode="w", newline="") as file:
    # Define the column headers (keys of the dictionary)
    fieldnames = ["image_path", "prediction"]
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    
    writer.writeheader()  # Write the header
    writer.writerows(serializable_outputs)  # Write the rows

print(f"Predictions saved to {csv_file_path}")
