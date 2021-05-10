import numpy as np
import matplotlib.pyplot as plt

f = open("train-images.idx3-ubyte", "rb")

image_size = 28
num_images = 190

f.read(16)
buf = f.read(image_size * image_size * num_images)
data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
data = data.reshape(num_images, image_size, image_size, 1)

image = np.asarray(data[2]).squeeze()
plt.imshow(image)
plt.show()