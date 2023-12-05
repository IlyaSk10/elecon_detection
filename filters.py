import cv2


# ------------------Read and transform image------------------------
class Filters:

    def __init__(self, path, size=None):
        self.path = path
        self.size = size

    def read_image(self):
        img = cv2.imread(self.path, cv2.IMREAD_COLOR)
        if self.size is not None:
            xmin, xmax, ymin, ymax = self.size
            img = self.read_image()[xmin:xmax, ymin:ymax]
        return img


#
# # Convert to grayscale.
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# fig, ax = plt.subplots()
# ax.imshow(gray, cmap='gray')
# ax.set_title('gray image')
#
# # Blur the image for better edge detection
# img_blur = cv2.GaussianBlur(gray, (3, 3), 0)
# fig, ax = plt.subplots()
# ax.imshow(img_blur, cmap='gray')
# ax.set_title('gaussian blur image')
#
# # Canny detector
# canny = cv2.Canny(img_blur, 0, 100)
# fig, ax = plt.subplots()
# ax.imshow(canny, cmap='gray')
# ax.set_title('Canny detector')
#
# # Sobel/Laplacian detector
# depth = cv2.CV_16S  # -1
# kernel_size = 5
# xorder = 1
# yorder = 0
# # laplacian = cv2.Laplacian(img_blur,ddepth=depth,ksize=kernel_size)
# sobel = cv2.Sobel(img_blur, depth, xorder, yorder, kernel_size)
# fig, ax = plt.subplots()
# ax.imshow(sobel, cmap='gray')
# ax.set_title('Sobel/Laplacian detector')
#
# # Filter2d
# depth = cv2.CV_16S  # -1
# kernel_size = 3
# prewitt = cv2.filter2D(img_blur, depth, kernel_size)
# fig, ax = plt.subplots()
# ax.imshow(prewitt, cmap='gray')
# ax.set_title('Filter2d')
#
# # Scharr detector
# depth = cv2.CV_16S  # -1
# xorder = 0
# yorder = 1
# scharr = cv2.Scharr(img, depth, xorder, yorder)
# fig, ax = plt.subplots()
# ax.imshow(scharr, cmap='gray')
# ax.set_title('Scharr detector')
