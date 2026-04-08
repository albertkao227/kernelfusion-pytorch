import cv2
import os

def convert_tiff_to_png(tiff_path, png_path):
    """
    Converts a TIFF image to PNG while strictly preserving pixel values and bit depth.
    """
    if not os.path.exists(tiff_path):
        raise FileNotFoundError(f"Could not find {tiff_path}")

    # cv2.IMREAD_UNCHANGED is the magic flag here. 
    # It tells OpenCV NOT to convert 16-bit images to 8-bit or mess with the channels.
    img = cv2.imread(tiff_path, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise ValueError(f"Failed to load image. Ensure '{tiff_path}' is a valid TIFF.")

    # Print the data type and shape to verify what we loaded
    print(f"Loaded {tiff_path}")
    print(f"  Shape: {img.shape}")
    print(f"  Data type: {img.dtype}")

    # CRITICAL CAVEAT: 
    # PNG format only supports up to 16-bit integers natively.
    # If your TIFF is a 32-bit float (common in some scientific datasets), 
    # it cannot be saved as a PNG without altering the values.
    if img.dtype in ['float32', 'float64']:
        print("\nWARNING: Your TIFF contains floating-point pixel values.")
        print("The PNG format only supports integer values (up to 16-bit).")
        print("To save this as a PNG, you must normalize and convert it to uint16 or uint8 first,")
        print("which WILL change your raw pixel values.\n")
        return False

    # Save to PNG. OpenCV automatically handles the 8-bit/16-bit integer encoding based on img.dtype
    # The default PNG compression is lossless, so pixel values remain identical.
    success = cv2.imwrite(png_path, img)
    
    if success:
        print(f"Successfully saved exact pixel values to {png_path}")
    else:
        print("Failed to save the PNG.")
        
    return success

if __name__ == "__main__":
    input_file = "data/images/lena_color.tiff"  # Can be .tif or .tiff
    output_file = "output_image.png"
    
    # Run conversion
    convert_tiff_to_png(input_file, output_file)