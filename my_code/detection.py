import numpy as np
import cv2
import copy
import matplotlib.pyplot as plt

'''
Takes numpy array in BGR order
'''
def detect_object(img_bgr, limit=0.5, display=False):
  # Convert to RGB & HSV
  img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
  img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

  # take 20x20 square in the middle of the picture to capture the color of the object
  x = 20
  size = (x*2)*(x*2)
  center_hsv = img_hsv[(img_hsv.shape[0]//2-x):(img_hsv.shape[0]//2+x), (img_hsv.shape[1]//2-x):(img_hsv.shape[1]//2+x), :]
  center_hsv = center_hsv.sum((0,1))
  center_hsv = center_hsv / size
  center_hsv = center_hsv.astype('uint8')

  # find other pixels in similar color and create a binary mask
  # similar color
  # hue +/- 50
  # saturation +/- 70
  # value +/- 70
  light = center_hsv.copy()
  h,s,v = 50, 70, 70
  light[0] -= h if light[0]>=h else 0
  light[1] -= s if light[1]>=s else light[1]
  light[2] -= v if light[2]>=v else light[2]

  dark = center_hsv.copy()
  dark[0] += h if dark[0]<=(180-h) else (180-dark[0])
  dark[1] += s if dark[1]<=(250-s) else (250-dark[1])
  dark[2] += v if dark[2]<=(250-v) else (250-dark[2])

  kernel = np.ones((3,3),np.uint8)
  mask = cv2.inRange(img_hsv, light, dark)
  # Apply closing operation on the mask to close up the missing regions
  for i in range(5):
      mask = cv2.dilate(mask, kernel, iterations = 1)
      mask = cv2.erode(mask, kernel, iterations = 1)

  # Deduct the countours given mask
  contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  if len(contours) == 0:
      return None

  # Find the biggest bounding box having middle pixel in it
  bounding_img = np.copy(img_rgb)
  area = [cv2.contourArea(c) for c in contours]
  idx = np.argmax(area)
  contour = contours[idx]
  middle_coords = (img_rgb.shape[1]//2, img_rgb.shape[0]//2)

  # Bounding box should be at least of size 50x50
  if area[idx] < 50*50 or area[idx] > limit * (img_rgb.shape[0] * img_rgb.shape[1]):
    return None

  # Add padding around the object
  coords = cv2.boundingRect(contour)
  x, y, w, h = coords
  x -= 5
  y -= 5
  w += 10
  h += 10

  # I noticed the bounding box would often hit the side edges of the image - which was wrong - so I exclude those
  if x < 10 or x+w >img_rgb.shape[1]-10:
      return None

  masked_img = np.copy(img_rgb)

  # If close to black, make brighter
  if center_hsv[2] < 50:
    masked_img = np.clip((masked_img)+100, 0, 255)

  # Cut out the object from the whole image
  masked_img = cv2.bitwise_and(masked_img,masked_img,mask=mask)
  masked_img = masked_img[y:y+h,x:x+w, :]

  # Append padding around object
  BLACK = [0,0,0]
  p = 20
  masked_img= cv2.copyMakeBorder(masked_img.copy(),p,p,p,p,cv2.BORDER_CONSTANT,value=BLACK)
  gray = cv2.cvtColor(masked_img, cv2.COLOR_RGB2GRAY)
  gray_down = cv2.resize(gray, (28,28))

  if display:
    plt.figure(figsize=(10, 10), dpi=100)
    plt.subplot(1, 3, 1)
    plt.imshow(img_rgb)
    plt.subplot(1, 3, 2)
    plt.imshow(masked_img)
    plt.subplot(1, 3, 3)
    plt.imshow(gray_down, cmap='gray')

  return gray_down, masked_img, coords
