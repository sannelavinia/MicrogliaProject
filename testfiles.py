import os

# Path to the folders
images_folder = 'images'
labels_folder = 'labels'

# Get all image filenames and remove the ".ome.tif" extension
images = [f[:-8] for f in os.listdir(images_folder) if f.endswith('.ome.tif')]

# Get all label filenames and remove the "-lables.png" extension
labels = [f[:-11] for f in os.listdir(labels_folder) if f.endswith('-labels.png')]

# Find images without a corresponding label
missing_labels = [image for image in images if image not in labels]

if missing_labels:
    print("Images without corresponding labels:")
    for image in missing_labels:
        print(image + '.ome.tif')
else:
    print("All images have corresponding labels.")
