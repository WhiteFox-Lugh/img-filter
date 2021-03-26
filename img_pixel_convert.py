import time
import numpy as np
import math
from PIL import Image

def read_image():
  img_origin = Image.open('./test.png')
  ret = img_origin.convert('RGB')
  return ret


def pixelate_img(origin_img, pixel_size):
  img = origin_img
  w, h = img.size
  print(w, h)
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


def main():
  img = read_image()
  start_time = time.time()
  px_filtered_img = pixelate_img(img, 8)
  end_time = time.time()
  elapsed_time = end_time - start_time
  print(elapsed_time)
  px_filtered_img.show()

  return


if __name__ == "__main__":
  main()
