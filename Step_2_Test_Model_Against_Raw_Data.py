import tensorflow as tf
import shutil
import os
import numpy as np
from keras.models import load_model

# Load the trained Keras model from disk
new_model = load_model(os.path.join('Models', 'readwatermetermodel.keras'))
# Class labels for digit output predictions
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
# Directory containing raw images to classify
directory = r'rawimages'
# Base folder where classified images will be moved
base_output_path = r'Images'
# Folder for images that do not map to a known digit label
Unknown_Directory = r'Unknown'

# List all raw image filenames and sort them for deterministic processing
items = os.listdir(directory)
sorted_items = sorted(items)

meterreading = ""
for filename in sorted_items:
    if filename.lower().endswith('.jpg'):
        file_path = os.path.join(directory, filename)

        # Read and decode the JPEG image from disk
        image_raw = tf.io.read_file(file_path)
        img = tf.io.decode_jpeg(image_raw)

        # Resize the image to the input size expected by the model
        resize = tf.image.resize(img, (256, 256))

        # Run prediction on the normalized image batch
        new_pred = new_model.predict(np.expand_dims(resize / 255.0, 0))
        meterreading = str(class_names[new_pred[0].argmax()])

        # Choose destination folder based on predicted digit
        if meterreading in class_names:
            dest_folder = os.path.join(base_output_path, meterreading)
        else:
            dest_folder = Unknown_Directory

        # Create destination folder if it does not exist
        if not os.path.exists(dest_folder):
            os.makedirs(dest_folder)
            print(f"Created folder: {dest_folder}")

        # Move the processed image into the predicted label folder
        dest_file_path = os.path.join(dest_folder, filename)
        shutil.move(file_path, dest_file_path)

        print(f"Extracted text: {meterreading}")
        print(f"Moved to: {dest_file_path}\n")

# Convert the loaded Keras model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(new_model)
tflite_model = converter.convert()

# Save the converted TensorFlow Lite model to disk
with open(os.path.join('Models', 'readwatermetermodel.tflite'), 'wb') as f:
    f.write(tflite_model)

