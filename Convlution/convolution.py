import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def convolution2D(image2D, kernel3x3):
    height, width = image2D.shape
    convolved2D = np.zeros((height-2, width-2))
    # ToDo: Write your code here...
    for i in range(1, height - 1):
       for j in range (1, width - 1):
          patch = image2D[i-1:i+2, j-1:j+2]
          value = np.sum(patch * kernel3x3)
          convolved2D[i-1, j-1] = value
    return convolved2D

image2D = np.loadtxt('Convlution/my-cat.csv', delimiter=',')
sns.heatmap(image2D, cmap='gray')
plt.title('Original image - Size = ' + str(image2D.shape))
plt.show()

edge_detect_filter_3x3 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

for i in range(2):
  convolved_image = convolution2D(image2D, edge_detect_filter_3x3)
  sns.heatmap(convolved_image, cmap='gray')
  plt.title('Convolution iteration ' + str(i) + ' - Size = ' + str(convolved_image.shape))
  plt.show()
  image2D = convolved_image 
