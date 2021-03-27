import numpy as np
import math
import argparse
import cv2
from PIL import Image


def read_image(file_path):
  img_origin = Image.open(file_path)
  ret = img_origin.convert('RGB')
  return ret


def pil_to_opencv(img):
  return np.asarray(img)


def opencv_to_pil(img):
  return Image.fromarray(img)


def pixelate_img(origin_img, pixel_size):
  img = origin_img
  w, h = img.size
  w_mesh = math.ceil(w / pixel_size)
  h_mesh = math.ceil(h / pixel_size)
  pos_list = [(x, y) for x in range(w_mesh) for y in range(h_mesh)]
  diff_pos_list = [(dx, dy) for dx in range(pixel_size) for dy in range(pixel_size)]
  mesh_list = list(map(lambda t: [(min(t[0] * pixel_size + dx, w - 1), min(t[1] * pixel_size + dy, h - 1)) for (dx, dy) in diff_pos_list], pos_list))
  unique_pos_list = list(map(lambda lst: set(lst), mesh_list))
  pixel_set = list(map(lambda pos_set: list(map(lambda pos: img.getpixel((pos[0], pos[1])), pos_set)), unique_pos_list))
  filtered_pixel_set = list(map(lambda lst: tuple(np.mean(lst, axis=0, dtype='int')), pixel_set))
  list(map(lambda pos_lst, color: list(map(lambda pos: img.putpixel(pos, color), pos_lst)), mesh_list, filtered_pixel_set))
  return img


def gray_scale(origin_img):
  return origin_img.convert('LA')


def sepia_scale(origin_img):
  sepia_filter_matrix = np.array([[0.393, 0.769, 0.189], [0.349, 0.686, 0.168], [0.272, 0.534, 0.131]])
  data = pil_to_opencv(origin_img)
  img_1 = np.apply_along_axis(lambda p: sepia_filter_matrix @ p, 2, data)
  img_2 = np.apply_along_axis(lambda p: np.where(p > 255, 255, p), 2, img_1)
  filtered = img_2.astype(np.uint8)
  res = opencv_to_pil(filtered)
  return res


def quantize(origin_img, color_num):
  res = origin_img.quantize(colors=color_num, method=0, kmeans=1, dither=1)
  return res


def main(args):
  img = read_image(args.file_path)
  px_filtered_img = pixelate_img(img, args.pixel)
  if args.gray:
    px_filtered_img = gray_scale(px_filtered_img)
  elif args.sepia:
    px_filtered_img = sepia_scale(px_filtered_img)
  res_img = quantize(px_filtered_img, args.color)
  res_img.show()
  res_img.save('./output.png')

  return


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("file_path", help="image file path")
  parser.add_argument("--pixel", "-p", help="size of pixel size", type=int, default=2)
  parser.add_argument("--color", "-c", help="number of color after filtered", type=int, default=4)
  parser.add_argument("--gray", "-g", help="gray scale", action="store_true")
  parser.add_argument("--sepia", "-s", help="sepia color", action="store_true")
  args = parser.parse_args()
  main(args)
