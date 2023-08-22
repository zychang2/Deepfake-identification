from PIL import Image
import numpy as np
import cv2
import os


def generate_dct(filename, in_folder_path, out_folder_path):
    
    image = cv2.imread(f"{in_folder_path}{filename}")

    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    dct_grayscale_image = cv2.dct(np.float32(img_gray) / 255.0)

    img_dct = np.uint8(dct_grayscale_image*255.0)

    out_path = f"{out_folder_path}{filename}"

    cv2.imwrite(out_path, img_dct)

    # Generate colormapped images (for paper)
    # imc = cv2.applyColorMap(img_dct, cv2.COLORMAP_VIRIDIS)

    # cv2.imwrite("fake_dct_colormap.png", imc)


os.mkdir(f"{os.environ['DATA_DIR']}dct_fake_imgs")
os.mkdir(f"{os.environ['DATA_DIR']}dct_real_imgs")


for filename in os.listdir(f"{os.environ['DATA_DIR']}fake_imgs"):
    in_folder_path = f"{os.environ['DATA_DIR']}fake_imgs/"
    out_folder_path = f"{os.environ['DATA_DIR']}dct_fake_imgs/"
    generate_dct(filename, in_folder_path, out_folder_path)

for filename in os.listdir(f"{os.environ['DATA_DIR']}real_imgs"):
    in_folder_path = f"{os.environ['DATA_DIR']}real_imgs/"
    out_folder_path = f"{os.environ['DATA_DIR']}dct_real_imgs/"
    generate_dct(filename, in_folder_path, out_folder_path)

