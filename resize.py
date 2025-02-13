# from PIL import Image
#
# img = Image.open("C:/paper_img/img/FFHQ-Aging/AF/img/30_39_54456.jpg")
# (w, h) = img.size
# print('w=%d, h=%d', w, h)
# img.show()
#
# new_img = img.resize((256, 256))
# new_img.show()
# new_img.save("C:/paper_img/img/FFHQ-Aging/AF/img/54456.jpg")

from PIL import Image
import os

# "E:/age_dataset/Dataset/dataset2/AF/test/21-30/""C:/paper_img/img/MORPH/img"
# Define the directory containing the images and the output directory
input_dir = "C:/paper_img/img/AgeDB/img/"
output_dir = "C:/paper_img/img/AgeDB/resized/"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Process each file in the input directory
for filename in os.listdir(input_dir):
    # Construct the full path to the input file
    input_path = os.path.join(input_dir, filename)

    # Skip non-image files
    if not (filename.lower().endswith(".jpg") or filename.lower().endswith(".png")):
        print(f"Skipping non-image file: {filename}")
        continue

    try:
        # Open the image
        img = Image.open(input_path)

        # Print the original dimensions
        w, h = img.size
        print(f"Processing {filename}: original size (w={w}, h={h})")

        # Resize the image to 256x256
        new_img = img.resize((256, 256))

        # Save the resized image to the output directory
        output_path = os.path.join(output_dir, filename)
        new_img.save(output_path)
        print(f"Saved resized image to {output_path}")

    except Exception as e:
        print(f"Error processing {filename}: {e}")

