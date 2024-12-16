import cv2
import numpy as np
from matplotlib import pyplot as plt

from compress import compress_image, evaluate_metrics
from morfologi import improve_segmentation
from segmentasi import segment_image

def main():
    # Input image path
    input_path = "./bird.jpg" 
    output_path = "output_compressed.jpg"

    # Step 1 - 2: Segment image & Improve segmentation
    original_image, segmented = segment_image(input_path)

    # Step 3: Compress final image
    compress_image(segmented, output_path)

    # Evaluate PSNR and SSIM
    psnr, ssim_val = evaluate_metrics(segmented, output_path)
    print(f"PSNR: {psnr:.2f} dB")
    print(f"SSIM: {ssim_val:.4f}")

    # Display results
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(original_image)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Segmented Image")
    plt.imshow(segmented)
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Compressed Image")
    image_out = cv2.imread(output_path)
    image_out = cv2.cvtColor(image_out, cv2.COLOR_BGR2RGB)
    plt.imshow(image_out)
    plt.axis("off")

    plt.show()

    print(f"Compressed image saved to {output_path}")

if __name__ == "__main__":
    main()
