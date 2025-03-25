import cv2
import numpy as np
import matplotlib.pyplot as plt

label = cv2.imread('./labels/NHB01_S03-labels.png', cv2.IMREAD_GRAYSCALE)
print(label)

# Modify the pixel values
label[label == 119] = 1
label[label == 255] = 0

label = np.expand_dims(label, axis=-1)
label

plt.imshow(label)
plt.show()


