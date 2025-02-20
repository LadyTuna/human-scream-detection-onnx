# Human Scream Detection ONNX Model

## Steps to export the ONNX model
1. Clone the [Human Scream Detection repo](https://github.com/Arav1ndE5/Human-Scream-Detection-1-phase) and use the README.md and the python notebook to get the model running. 
2. In the main.py file, after line 133 add the following code (make sure to add the corresponding import) to export the model in the ONNX format
```
import onnx

#Define input signature (e.g., shape (batch_size, input_size))
#This helps to map the shape of the input

input_signature = [tf.TensorSpec(shape=[None, X_train.shape[1]], dtype=tf.float32)]

# Convert the model to ONNX using the input signature
onnx_model,_ = tf2onnx.convert.from_keras(model_new, input_signature=input_signature)

onnx.save_model(onnx_model, 'scream_detection_model.onnx')
print("ONNX model saved successfully!")
```
3. Now the model is saved in the ONNX format and can be used independently to make predictions, since our aim is to run it with rescue box we will need to do the next steps.

_Alternatively instead of following steps 1, 2 and 3, you can use Scream_det_convert_to_ONNX.ipynb found in this repository to get the trained model in the .h5 and the .onnx format._

4. Download and configure [Rescue Box Desktop](https://github.com/UMass-Rescue/RescueBox-Desktop/releases), follow the README.md file. To ensure rescue box is working you can try running a saple code found in the [documentation](https://umass-rescue.github.io/Flask-ML/materials/guides/getting-started/).
5. Use the exported ONNX model and the pre-processing (and post-processing) functions to run the Human Scream Detection model using ONNX.

## Steps to use the Human Scream Detection model
### Create a vitual Environment and install the dependencies
Create a virtual environment using the following commands. The virtual environment name used in this example is _screamDet_, but you can choose any name you'd like~

(As a side note I find that for this project python version 3.11.11 works the best and encompasses al the dependencies in the requirements.txt file~~)

```bash
python -m venv screamDet
```

Next we use the following command to install the required dependencies

```bash
pip install -r requirements.txt
```

When you clone this repository you will find the trained model in the ONNX format, named _scream_detection_model.onnx_

If you'd like to test this project out using the CLI you can use this command

```bash
python screamDetector_cli.py --input_dir path/to/audio_dir --output_dir path/to/output_dir
```
where audio_dir is the directory that has all the audio's you'd like to be classified and output_dir is the directory where you'd like the output csv to be saved.

To run this application using a GUI we need to download and run [Rescue Box Desktop](https://github.com/UMass-Rescue/RescueBox-Desktop/releases)! You can follow the instructions mentioned in the repo's README.md file

To run the server run the following command
```bash
python server.py
```
Now open the rescue box application 