import cv2
import matplotlib.pyplot as plt
import numpy as np

image = cv2.imread("./labels_classification_test/NHB12_S05-labels.png", cv2.IMREAD_COLOR_RGB)
image = image[:,:,2]



print(np.unique(image))


plt.figure()
plt.imshow(image, cmap='viridis')
plt.colorbar()
plt.show()
