### Preprocess the data here.
import cv2
import numpy as np
from numpy import newaxis
from First_Model_Exploration_Visualizer import *
import scipy.ndimage
from sklearn.model_selection import train_test_split

# convert to B/W
#X_train_bw = np.array([cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) for image in X_train])
#X_test_bw = np.array([cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) for image in X_test])

# apply histogram equalization
X_train_hst_eq = np.array([cv2.equalizeHist(image) for image in X_train])
X_test_hst_eq = np.array([cv2.equalizeHist(image) for image in X_test])

# reshape for conv layer
X_train_reshaped = X_train_hst_eq[..., newaxis]
X_test_reshaped = X_test_hst_eq[..., newaxis]
print('Before shaping:', X_train_hst_eq.shape)
print('After shaping:', X_train_reshaped.shape)

# normalize range
X_train_normalized = X_train_reshaped - np.mean(X_train_reshaped)
X_test_normalized = X_test_reshaped - np.mean(X_test_reshaped)
print('Mean, min and max before normalizing:', np.mean(X_train_reshaped), np.min(X_train_reshaped), np.max(X_train_reshaped))
print('Mean, min and max after normalizing:', np.mean(X_train_normalized), np.min(X_train_normalized), np.max(X_train_normalized))

def create_variant(image):
    if (random.choice([True, False])):
        image = scipy.ndimage.interpolation.shift(image, [random.randrange(-2, 2), random.randrange(-2, 2), 0])
    else:
        image = scipy.ndimage.interpolation.rotate(image, random.randrange(-10, 10), reshape=False)
    return image

# show image of N random data points
count = 10
fig, axs = plt.subplots(count, 3, figsize=(count, count*3))
fig.subplots_adjust(hspace = .2, wspace=.001)
axs = axs.ravel()
for i in range(0, count*3, 3):
    index = random.randint(0, len(X_train)-1)
    image = X_train[index]
    
    axs[i].axis('off')
    axs[i].imshow(image)
    axs[i].set_title(sign_names[int(y_train[index])])

    aug1 = create_variant(image)
    axs[i+1].axis('off')
    axs[i+1].imshow(aug1)
    axs[i+1].set_title("Augmented")
    
    aug2 = create_variant(image)
    axs[i+2].axis('off')
    axs[i+2].imshow(aug2)
    axs[i+2].set_title("Augmented")
### Generate data additional data (OPTIONAL!)
### and split the data into training/validation/testing sets here.
### Feel free to use as many code cells as needed.
    
# data augmentation
REQ_NUM_SAMPLES = 1000

generated_features = []
generated_labels = []
for class_index in range(len(sign_type)):
    class_sample_count = sign_type[class_index]

    augment_multiple = round(REQ_NUM_SAMPLES / class_sample_count)
    if augment_multiple <= 1:
        continue
    
    print("Class {:d} has only {:d} samples, hence augmenting {:d} times.".format(class_index, class_sample_count, augment_multiple))
    for test_feature, test_label in zip(X_train_normalized, y_train):
        if class_index == test_label:
            for augment_iter in range(augment_multiple):
                generated_features.append(create_variant(test_feature))
                generated_labels.append(test_label)

# append generated data to original data
print( np.array(generated_features))
X_train_augmented = np.append(np.array(X_train_normalized), np.array(generated_features), axis=0)
y_train_augmented = np.append(np.array(y_train), np.array(generated_labels), axis=0)

# create validation set from training data
X_training, X_validation, y_training, y_validation = train_test_split(X_train_augmented, y_train_augmented, test_size=0.2)