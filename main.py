import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img, img_to_array

parent_folder = '/content/dataset/Elder futhark'

num_images_needed = 200
augmentations_per_image = num_images_needed // 5  

folders = [os.path.join(parent_folder, name) for name in os.listdir(parent_folder)
           if os.path.isdir(os.path.join(parent_folder, name))]

datagen = ImageDataGenerator(
    rotation_range=0,  
    width_shift_range=0.05, 
    height_shift_range=0.05,  
    zoom_range=(1.0, 1.1),  
    horizontal_flip=False,  
    vertical_flip=False,
    fill_mode='nearest'  
)

for folder in folders:
    print(f"Processing folder: {folder}")

    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)
        
        if os.path.isfile(img_path) and img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            img = load_img(img_path) 
            img_array = img_to_array(img) 
            img_array = img_array.reshape((1,) + img_array.shape)  

            for i, batch in enumerate(datagen.flow(img_array, batch_size=1, save_to_dir=folder, save_prefix='aug', save_format='jpeg')):
                if i >= augmentations_per_image:
                    break

print(f"\nGenerated {num_images_needed * len(folders)} augmented images.\n")

for folder_path in folders:
    files = sorted(os.listdir(folder_path))  

    for i, filename in enumerate(files):
        old_file = os.path.join(folder_path, filename)
        new_file = os.path.join(folder_path, f'image_{i + 1}.jpg')

        if os.path.isfile(old_file):
            os.rename(old_file, new_file)

print("Renaming completed for all folders.")

import tensorflow as tf

dataset_path = '/content/dataset/Elder futhark'  # This should contain folders like a, b, c, etc.

datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(150, 150),  
    batch_size=8,
    class_mode='categorical'
)

from tensorflow.keras import layers, models

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

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


model.fit(train_generator, epochs=3)
