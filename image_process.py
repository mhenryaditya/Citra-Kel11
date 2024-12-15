import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity as ssim

def segment_image(image_path):
    # Load image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply threshold to create binary image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Remove noise with morphological operations
    kernel = np.ones((18, 18), np.uint8)
    opening = improve_segmentation(binary, kernel)

    # Sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # Unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    _, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0
    markers = markers + 1

    # Mark the unknown region with 0
    markers[unknown == 255] = 0

    # Apply the Watershed algorithm
    markers = cv2.watershed(image, markers)

    # Mask for the segmented object
    object_mask = np.zeros_like(markers, dtype=np.uint8)
    object_mask[markers == 1] = 255  # Keep only the object area (markers == 1)
    
    # Apply the mask to the original image
    segmented_object = cv2.bitwise_and(image_rgb, image_rgb, mask=object_mask)

    return image_rgb, segmented_object 

def improve_segmentation(binary, kernel):
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    return cleaned

def evaluate_metrics(original_image, compressed_image):
    original = cv2.imread(original_image)
    compressed = cv2.imread(compressed_image)
    psnr_value = cv2.PSNR(original, compressed)
    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    compressed_gray = cv2.cvtColor(compressed, cv2.COLOR_BGR2GRAY)
    ssim_value = ssim(original_gray, compressed_gray)
    return psnr_value, ssim_value

def compress_image(image, output_path, quality=50):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print("Dimensi gambar awal:", img.shape)
    img_YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    cr = img_YCrCb[:, :, 1]
    cb = img_YCrCb[:, :, 2]
    target_shape = (cr.shape[1] // 2, cr.shape[0] // 2)

    cr_resized = cv2.resize(cr, target_shape, interpolation=cv2.INTER_LINEAR)
    cb_resized = cv2.resize(cb, target_shape, interpolation=cv2.INTER_LINEAR)

    img_YCrCb[:, :, 1] = cv2.resize(cr_resized, img_YCrCb[:, :, 0].shape[::-1])
    img_YCrCb[:, :, 2] = cv2.resize(cb_resized, img_YCrCb[:, :, 0].shape[::-1])

    img_compressed = cv2.cvtColor(img_YCrCb, cv2.COLOR_YCrCb2BGR)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, img_compressed = cv2.imencode('.jpg',img_compressed, encode_param)

    with open(output_path, 'wb') as f:
        f.write(img_compressed)

def main():
    # Input image path
    input_path = "./bird.jpg"  # Replace with your image path
    output_path = "output_compressed.jpg"

    # Step 1 - 2: Segment image & Improve segmentation
    original_image, segmented = segment_image(input_path)

    # Step 3: Compress final image
    compress_image(segmented, output_path)

    # Evaluate PSNR and SSIM
    psnr, ssim_val = evaluate_metrics(input_path, output_path)
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
