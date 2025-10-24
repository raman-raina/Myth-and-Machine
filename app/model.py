import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image

def train_model(d1):
	datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

	train_generator = datagen.flow_from_directory(
    		d1,
    		target_size=(150, 150),
    		batch_size=8,
    		class_mode='categorical'
	)

	model = models.Sequential([
    		layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    		layers.MaxPooling2D(2, 2),
    		layers.Conv2D(64, (3, 3), activation='relu'),
    		layers.MaxPooling2D(2, 2),
    		layers.Conv2D(128, (3, 3), activation='relu'),
    		layers.MaxPooling2D(2, 2),
    		layers.Flatten(),
    		layers.Dense(128, activation='relu'),
    		layers.Dense(len(train_generator.class_indices), activation='softmax')
	])

	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

	model.fit(train_generator, epochs=3)

	classes = train_generator.class_indices

	return model, classes

def predict_images(d2, model, class_indices):
    image_files = [f for f in os.listdir(d2) if f.endswith(('.jpg', '.jpeg', '.png'))]

    results = {}
    for img_file in image_files:
        img_path = os.path.join(d2, img_file)
        img = image.load_img(img_path, target_size=(150, 150))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        predictions = model.predict(img_array)

        predicted_class_index = np.argmax(predictions, axis=1)[0]

        class_labels = list(class_indices.keys())
        predicted_label = class_labels[predicted_class_index]

        results[img_file] = predicted_label

    return results