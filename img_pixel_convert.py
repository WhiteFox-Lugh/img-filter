import numpy as np
import math
import argparse
import cv2
import time
from PIL import Image

flag_pil_to_cv = False


def read_image(file_path: str) -> np.ndarray:
    return cv2.imread(file_path)


def pil_to_opencv(img: Image) -> np.ndarray:
    rgb_img = img.convert('RGB')
    flag_pil_to_cv = True
    return np.asarray(rgb_img)


def opencv_to_pil(img: np.ndarray) -> Image:
    bgr_to_rgb = img[:, :, ::-1].copy() if flag_pil_to_cv else img
    return Image.fromarray(bgr_to_rgb)


def gray_scale(origin_img: np.ndarray) -> np.ndarray:
    pil_img = opencv_to_pil(origin_img)
    gray_img = pil_img.convert('LA')
    res = pil_to_opencv(gray_img)
    return res


def sepia_scale(origin_img: np.ndarray) -> np.ndarray:
    sepia_filter_matrix = np.array(
        [[0.131, 0.534, 0.272], [0.168, 0.686, 0.349], [0.189, 0.769, 0.393]])
    img_1 = np.apply_along_axis(
        lambda p: sepia_filter_matrix @ p, 2, origin_img)
    img_2 = np.apply_along_axis(lambda p: np.where(p > 255, 255, p), 2, img_1)
    res = img_2.astype(np.uint8)
    return res


def mosaic(origin_img: np.ndarray, size: int) -> np.ndarray:
    h, w, c = origin_img.shape
    rw = int(w / size)
    rh = int(h / size)
    im_resize = cv2.resize(origin_img, (rw, rh),
                           interpolation=cv2.INTER_NEAREST)
    res = cv2.resize(im_resize, (w, h), interpolation=cv2.INTER_NEAREST)
    return res


def gamma_correction(origin_img: np.ndarray, gamma: int) -> np.ndarray:
    n = 256
    lut = np.empty((1, n), dtype=np.uint8)
    for i in range(n):
        lut[0][i] = np.clip(
            np.power(i / (n - 1.0), 1.0 / gamma) * (n - 1), 0, n-1)
    res = cv2.LUT(origin_img, lut)
    return res


def quantize(origin_img: np.ndarray, color_num: int, pixel_size: int) -> np.ndarray:
    pil_img = opencv_to_pil(origin_img)
    quantized_img = pil_img.quantize(
        colors=color_num, method=0, kmeans=1, dither=1)
    res = pil_to_opencv(quantized_img)
    return res


def pixelate_img(origin_img: np.ndarray, pixel_size: int) -> np.ndarray:
    img = opencv_to_pil(origin_img)
    w, h = img.size
    w_mesh = math.ceil(w / pixel_size)
    h_mesh = math.ceil(h / pixel_size)
    pos_list = [(x, y) for x in range(w_mesh) for y in range(h_mesh)]
    diff_pos_list = [(dx, dy) for dx in range(pixel_size)
                     for dy in range(pixel_size)]
    mesh_list = list(map(lambda t: [(min(t[0] * pixel_size + dx, w - 1), min(
        t[1] * pixel_size + dy, h - 1)) for (dx, dy) in diff_pos_list], pos_list))
    unique_pos_list = list(map(lambda lst: set(lst), mesh_list))
    pixel_set = list(map(lambda pos_set: list(
        map(lambda pos: img.getpixel((pos[0], pos[1])), pos_set)), unique_pos_list))
    filtered_pixel_set = list(map(lambda lst: tuple(
        np.mean(lst, axis=0, dtype='int')), pixel_set))
    list(map(lambda pos_lst, color: list(map(lambda pos: img.putpixel(
        pos, color), pos_lst)), mesh_list, filtered_pixel_set))
    res = pil_to_opencv(img)
    return res


def dither(img: np.ndarray) -> np.ndarray:
    mod = 4
    mat = np.array([[0, 8, 2, 10], [12, 4, 14, 6],
                    [3, 11, 1, 9], [15, 7, 13, 5]])
    mat *= 16
    mat += 8
    w = img.shape[1]
    h = img.shape[0]
    w_mesh = math.ceil(w / mod)
    h_mesh = math.ceil(h / mod)
    mat = np.tile(mat, (h_mesh, w_mesh))
    mat_w_size = mat.shape[1]
    mat_h_size = mat.shape[0]
    if mat_w_size > img.shape[1]:
        diff = mat_w_size - img.shape[1]
        for i in range(diff):
            mat = np.delete(mat, -1, 1)
    if mat_h_size > img.shape[0]:
        diff = mat_h_size - img.shape[0]
        for i in range(diff):
            mat = np.delete(mat, -1, 0)
    blue, green, red = cv2.split(img)
    d_blue = np.where(blue < mat, 0, blue)
    d_green = np.where(green < mat, 0, green)
    d_red = np.where(red < mat, 0, red)
    res = cv2.merge((d_blue, d_green, d_red))
    return res


def main(args):
    t_start = time.time()
    res_img = read_image(args.file_path)
    if args.pixel >= 2:
        res_img = pixelate_img(res_img, args.pixel) if args.hq else mosaic(
            res_img, args.pixel)
        res_img = quantize(res_img, args.color, args.pixel)
    if args.dither:
        res_img = dither(res_img)
    if args.gray:
        res_img = gray_scale(res_img)
    elif args.sepia:
        res_img = sepia_scale(res_img)
    if args.gamma > 0:
        res_img = gamma_correction(res_img, args.gamma)
    cv2.imwrite('output.png', res_img)
    t_end = time.time()
    elapsed_time = t_end - t_start
    print("Convert completed (elapsed time : " + str(round(elapsed_time, 3)) + "s)")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", help="image file path")
    parser.add_argument("--pixel", "-p", help="size of pixel size",
                        type=int, default=-1, choices=[2, 3, 4, 5, 6, 7, 8])
    parser.add_argument("--hq", "-q", help="pixelate", action="store_true")
    parser.add_argument(
        "--color", "-c", help="number of color after filtered", type=np.uint8, default=4)
    parser.add_argument("--gray", "-g", help="gray scale", action="store_true")
    parser.add_argument("--sepia", "-s", help="sepia color",
                        action="store_true")
    parser.add_argument("--dither", "-d", help="dithering",
                        action="store_true")
    parser.add_argument(
        "--gamma", "-gc", help="gamma correction", type=float, default=-1.0)
    args = parser.parse_args()
    main(args)
