import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Argument parser for training/testing
parser = argparse.ArgumentParser()
parser.add_argument('--train', action='store_true', help='Train the model')
parser.add_argument('--test', type=str, help='Path to image for testing')
args = parser.parse_args()

# Dataset paths
train_dir = 'dataset/'

# CNN Model definition
def build_model():
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax')
    ])
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

if args.train:
    # Data preprocessing
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_gen = datagen.flow_from_directory(
        train_dir, target_size=(128,128), batch_size=32,
        class_mode='categorical', subset='training'
    )
    val_gen = datagen.flow_from_directory(
        train_dir, target_size=(128,128), batch_size=32,
        class_mode='categorical', subset='validation'
    )

    model = build_model()

    history = model.fit(train_gen, validation_data=val_gen, epochs=10)

    # Save model
    os.makedirs('model', exist_ok=True)
    model.save('model/plant_cnn_model.h5')
    print("✅ Model trained and saved as plant_cnn_model.h5")

    # Plot accuracy graph
    plt.plot(history.history['accuracy'], label='train acc')
    plt.plot(history.history['val_accuracy'], label='val acc')
    plt.legend()
    plt.show()

elif args.test:
    model = load_model('model/plant_cnn_model.h5')
    image = load_img(args.test, target_size=(128,128))
    img_arr = img_to_array(image)/255.0
    img_arr = np.expand_dims(img_arr, axis=0)

    prediction = model.predict(img_arr)
    class_index = np.argmax(prediction)

    if class_index == 0:
        print("🌿 The leaf is Healthy!")
    else:
        print("⚠️ The leaf is Diseased!")

else:
    print("Please specify either --train or --test argument.")
