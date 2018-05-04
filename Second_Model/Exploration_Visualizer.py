### Data exploration visualization goes here.
### Feel free to use as many code cells as needed.
import matplotlib.pyplot as plt
import random
import cv2
import numpy
from Data_Explorer import *

# show image of N random data points
count = 10
fig, axs = plt.subplots(count, 3, figsize=(count, count*3))
fig.subplots_adjust(hspace = .2, wspace=.001)
axs = axs.ravel()
for i in range(0, count*3, 3):
    index = random.randint(0, len(X_train)-1)
    #print(index)
    image = X_train[index]

    axs[i].axis('off')
    axs[i].imshow(image)
    axs[i].set_title(sign_names[int(y_train[index])])
    #print(sign_names[int(y_train[index])])

    bw = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    axs[i+1].axis('off')
    axs[i+1].imshow(bw, cmap='gray')
    axs[i+1].set_title("B/W")

    bw = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    axs[i+2].axis('off')
    axs[i+2].imshow(bw, cmap='gray')
    axs[i+2].set_title("B/W")

    '''equ = cv2.equalizeHist(bw)
    axs[i+2].axis('off')
    axs[i+2].imshow(equ, cmap='gray')
    axs[i+2].set_title("Histogram Equalized")'''

# plotting the count of each sign

y_pos = range(n_classes)
#print(y_pos)
label_list = y_train.tolist()
#print(label_list)
#print(n_classes)
sign_type = [label_list.count(y) for y in range(n_classes)]

plt.bar(y_pos, sign_type, width=0.8, align='center')
plt.ylabel('Sample Count')
plt.xlabel('Sample Class')
plt.show()