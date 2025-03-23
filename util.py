import cv2
import numpy as np

def calculate_psnr(image1, image2):
    # 计算均方误差
    mse = np.mean((image1 - image2) ** 2)
    if mse == 0:
        return float('inf')  # 两张图片完全相同
    max_pixel = 255.0  # 像素值范围
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


if __name__ == "__main__":
    # 读取图片
    image1 = cv2.imread('tmp/input/10.png')
    image2 = cv2.imread('tmp/output/10.png')
    image3 = cv2.imread('tmp/output_tile/10.png')

    # 确保两张图片大小一致
    psnr_value1 = calculate_psnr(image1, image2)
    psnr_value2 = calculate_psnr(image1, image3)
    psnr_value3 = calculate_psnr(image2, image3)
    print(f'PSNR1: {psnr_value1} dB')
    print(f'PSNR2: {psnr_value2} dB')
    print(f'PSNR3: {psnr_value3} dB')
    