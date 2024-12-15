import cv2
from skimage.metrics import structural_similarity as ssim

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

def evaluate_metrics(original_image, compressed_image):
    original = cv2.imread(original_image)
    compressed = cv2.imread(compressed_image)
    psnr_value = cv2.PSNR(original, compressed)
    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    compressed_gray = cv2.cvtColor(compressed, cv2.COLOR_BGR2GRAY)
    ssim_value = ssim(original_gray, compressed_gray)
    return psnr_value, ssim_value