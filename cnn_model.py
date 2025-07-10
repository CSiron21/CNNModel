import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import Sequence
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input, Conv2D, Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

class DataGenerator(Sequence):
    def __init__(self, images_dir, labels_csv, batch_size=32, img_size=(128, 128)):
        self.images_dir = images_dir
        self.labels = pd.read_csv(labels_csv)
        self.batch_size = batch_size
        self.img_size = img_size
        self.indexes = np.arange(len(self.labels))

    def __len__(self):
        return int(np.ceil(len(self.labels) / self.batch_size))

    def __getitem__(self, idx):
        batch_indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_images = []
        batch_targets = []
        for i in batch_indexes:
            row = self.labels.iloc[i]
            img_path = os.path.join(self.images_dir, row['image'])
            img = load_img(img_path, color_mode='grayscale', target_size=self.img_size)
            img = img_to_array(img) / 255.0
            batch_images.append(img)
            batch_targets.append(row['label'])
        return np.array(batch_images), np.array(batch_targets)

    def on_epoch_end(self):
        np.random.shuffle(self.indexes)

def build_mobilenet_v2_grayscale(input_shape=(128, 128, 1), num_classes=10, alpha=0.35):
    base_model = MobileNetV2(
        input_shape=(128, 128, 3),  # MobileNetV2 expects 3 channels
        include_top=False,
        weights=None,
        alpha=alpha
    )
    # Adapt for grayscale: repeat channel
    input_layer = Input(shape=input_shape)
    x = tf.keras.layers.Concatenate()([input_layer, input_layer, input_layer])
    x = base_model(x)
    x = GlobalAveragePooling2D()(x)
    output = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=input_layer, outputs=output)
    return model

def train_model():
    images_dir = 'train_data/images'
    labels_csv = 'train_data/labels.csv'
    batch_size = 32
    img_size = (128, 128)
    num_classes = pd.read_csv(labels_csv)['label'].nunique()

    train_gen = DataGenerator(images_dir, labels_csv, batch_size, img_size)
    model = build_mobilenet_v2_grayscale(input_shape=(128, 128, 1), num_classes=num_classes, alpha=0.35)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_gen, epochs=10)
    model.save('mobilenetv2_grayscale.h5')
    return model

def convert_to_tflite(model_path, labels_csv, tflite_path='mobilenetv2_grayscale.tflite'):
    model = tf.keras.models.load_model(model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    # Attach labels as metadata (simple append for demo)
    labels = pd.read_csv(labels_csv)['label'].unique()
    with open(tflite_path, 'ab') as f:
        f.write(b'\nLABELS:\n')
        for label in labels:
            f.write(f'{label}\n'.encode())
    print(f'TFLite model exported to {tflite_path} with labels.')

if __name__ == '__main__':
    model = train_model()
    convert_to_tflite('mobilenetv2_grayscale.h5', 'train_data/labels.csv') 