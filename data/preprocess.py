"""
preprocess.py
Convert lena_color.tiff to a 16-bit grayscale image.
"""

import os
from PIL import Image
import numpy as np

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, "images", "lena_color.tiff")
    output_path = os.path.join(script_dir, "images", "lena_gray16.tiff")

    img = Image.open(input_path)
    print(f"Input : {input_path}")
    print(f"  mode={img.mode}, size={img.size}")

    # Convert to grayscale (8-bit "L")
    gray = img.convert("L")

    # Scale from [0, 255] → [0, 65535] and cast to uint16
    arr = np.array(gray, dtype=np.float64)
    arr = (arr / 255.0 * 65535.0).round().astype(np.uint16)

    # Save as 16-bit TIFF via PIL (mode "I;16")
    out_img = Image.fromarray(arr)
    out_img.save(output_path)

    print(f"Output: {output_path}")
    print(f"  dtype={arr.dtype}, min={arr.min()}, max={arr.max()}, shape={arr.shape}")

if __name__ == "__main__":
    main()
