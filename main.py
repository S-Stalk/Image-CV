import cv2
import numpy as np
import matplotlib.pyplot as plt
import cvlib as cv
from cvlib.object_detection import draw_bbox

image = cv2.imread("cup.jpg")
box, label, count = cv.detect_common_objects(image)
outuput = draw_bbox(image, box, label, count)

print(label)

print(f"Колличество объектов на картинке: {label.count('cup')}")

plt.imshow(outuput)
plt.show()
