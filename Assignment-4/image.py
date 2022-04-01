import sys
import numpy as np
from PIL import Image

filename = sys.argv[1]

with open(filename, "r") as f:
    n, m = [int(n) for n in f.readline().split()]
    print(n, m)
    image = [int(num) for num in f.readline().split()]
    image = np.array(image, dtype=np.uint8)
    image = image.reshape((n, m, 3))
    pil_image = Image.fromarray(image, 'RGB')
    pil_image.save("data_2.png")
    pil_image.show()



