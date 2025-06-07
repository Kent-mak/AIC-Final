from PIL import Image
import os

def convert_to_grayscale(image_path, output_path=None):
    # Load the image
    image = Image.open(image_path)
    print(f"Original mode: {image.mode}")

    # Check if the image is RGB
    if image.mode == "RGB":
        print("Converting RGB to grayscale...")
        image = image.convert("L")
    elif image.mode == "L":
        print("Image is already grayscale.")
    else:
        print(f"Warning: unexpected image mode '{image.mode}', attempting to convert...")
        image = image.convert("L")

    # Set output path if not provided
    if output_path is None:
        base, ext = os.path.splitext(image_path)
        output_path = base + "_grayscale" + ext

    # Save the grayscale image
    image.save(output_path)
    print(f"Grayscale image saved to {output_path}")

# Example usage
convert_to_grayscale("validation/val_VAID_1.png")
