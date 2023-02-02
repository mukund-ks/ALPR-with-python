from skimage.io import imread
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt

car_img = imread("car.jpg", as_gray=True)
print(car_img.shape) # image is imported as 2-D Matrix

gray_car_img = car_img * 255 # 0 to 255 intensity values are easier for us to relate.

fig, (ax1, ax2) = plt.subplots(1,2)

ax1.imshow(gray_car_img, cmap="gray")

threshold_val = threshold_otsu(gray_car_img)

binary_car_img = gray_car_img > threshold_val
ax2.imshow(binary_car_img, cmap="gray")

plt.show()