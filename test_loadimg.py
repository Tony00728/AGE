import os
from PIL import Image


def load_images(directory, filename, age):

    images = {}
    # Construct the expected filename pattern
    pattern = f"{filename}_{age}.jpg"  # Adjust the file extension if needed

    # Full path to search for the image
    file_path = os.path.join(directory, pattern)

    # Check if the file exists
    if os.path.isfile(file_path):
        try:
            img = Image.open(file_path).convert('RGB')
            images[pattern] = img
        except Exception as e:
            print(f"Error loading image {file_path}: {e}")
    else:
        print(f"File not found: {file_path}")

    return images


# Usage example
directory = 'D:\FADING-master\output'
filename = '00204'  # Base filename without age suffix
age = '20'  # or '80'

loaded_images = load_images(directory, filename, age)



# Print loaded image filenames and display images
for fname, img in loaded_images.items():
    print(f"Loaded image: {fname}")
    img.show()  # Display the image