import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt
from tensorflow import keras
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.models import Sequential

# Path to the labeled image dataset directory
# The directory should contain class subfolders with meter images.
data_dir = r'Sanitized Data'
# Directory where TensorBoard logs will be written during training
log_dir = r'Logs'
# Supported image extensions for reference (unused in current code)
image_extensions = ['.jpg', '.jpeg', '.png']

def get_class_names_from_folder(directory):
    """Return sorted class names from the dataset directory structure."""
    import pathlib
    import numpy as np

    data_dir = pathlib.Path(directory)
    # Each subfolder name is treated as a class label
    class_names = np.array(sorted([item.name for item in data_dir.glob("*")]))
    return class_names

    # Note: the print statement below is unreachable because it comes after return.
    print(class_names)

def main():
    # Load the class labels from the dataset folders
    class_names = get_class_names_from_folder(directory=data_dir)
    class_names

    # Verify that each image file can be opened and decoded
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            file_path = os.path.join(root, file)
            #print(f"Processing file: {file_path}")
            try:
                raw_image = tf.io.read_file(file_path)
                tf.io.decode_jpeg(raw_image)
            except:
                print(f"Error file: {file_path}")

    # Create a TensorFlow dataset from the image directory
    data = tf.keras.utils.image_dataset_from_directory(data_dir)

    # Scale images to float values in [0, 1]
    data = data.map(lambda x, y: (x / 255.0, y))

    # Estimate how many batches will be used for each split
    print(len(data))
    train_size = int(len(data) * 0.7)
    val_size = int(len(data) * 0.2) + 1
    test_size = int(len(data) * 0.1)
    print(train_size, val_size, test_size)
    print(train_size + val_size + test_size)

    # Split the dataset into train/validation/test sets
    train = data.take(train_size)
    val = data.skip(train_size).take(val_size)
    test = data.skip(train_size + val_size).take(test_size)
    print(len(train))
    print(len(val))
    print(len(test))

    # Build a small convolutional neural network architecture
    model = Sequential()
    model.add(Conv2D(16, (3, 3), 1, activation='relu', input_shape=(256, 256, 3)))
    model.add(MaxPooling2D())
    model.add(Conv2D(32, (3, 3), 1, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(16, (3, 3), 1, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    # Compile model for multi-class classification with integer labels
    model.compile('adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
    # Alternative for binary classification:
    # model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
    model.summary()

    # Train model and write logs for TensorBoard
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir)
    history = model.fit(train, epochs=20, validation_data=val, callbacks=[tensorboard_callback])

    # Plot training and validation loss history
    fig = plt.figure()
    plt.plot(history.history['loss'], color='teal', label='loss')
    plt.plot(history.history['val_loss'], color='orange', label='val_loss')
    fig.suptitle('Loss', fontsize=20)
    plt.legend(loc="upper left")
    plt.show()

    # Plot training and validation accuracy history
    fig = plt.figure()
    plt.plot(history.history['accuracy'], color='teal', label='accuracy')
    plt.plot(history.history['val_accuracy'], color='orange', label='val_accuracy')
    fig.suptitle('Accuracy', fontsize=20)
    plt.legend(loc="upper left")
    plt.show()

    # Evaluate the final model on the test set
    model.evaluate(test)

    # Save the trained model to disk
    model.save(r'Models/readwatermetermodel.keras')

if __name__ == "__main__":
    main()