import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def augment(d1):

	num_images_needed = 200
	augmentations_per_image = num_images_needed // 5

	folders = [os.path.join(d1, name) for name in os.listdir(d1)
           	if os.path.isdir(os.path.join(d1, name))]

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
    		files = sorted(os.listdir(folder_path))  # Sort to keep order consistent

    		for i, filename in enumerate(files):
        		old_file = os.path.join(folder_path, filename)
        		new_file = os.path.join(folder_path, f'image_{i + 1}.jpg')

        		if os.path.isfile(old_file):
            			os.rename(old_file, new_file)

	print("Renaming completed for all folders.")