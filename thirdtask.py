import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from cuml import SVC

def load_and_preprocess_images(directory, image_size):
    features = []
    labels = []
    for image_name in tqdm(os.listdir(directory), desc=f"Processing Images in {directory}"):
        if image_name.startswith('cat'):
            label = 0
        else:
            label = 1
        image = cv2.imread(os.path.join(directory, image_name))
        if image is not None:
            image_resized = cv2.resize(image, image_size)
            image_normalized = image_resized / 255.0
            features.append(image_normalized)
            labels.append(label)
    return np.asarray(features), np.asarray(labels)

train_dir = "train"
image_size = (50, 50)
X_train, y_train = load_and_preprocess_images(train_dir, image_size)

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, shuffle=True, random_state=42)

X_train_flattened = X_train.reshape(len(X_train), -1)
X_test_flattened = X_test.reshape(len(X_test), -1)

model = SVC(C=1, kernel='poly', gamma='auto', probability=True)
model.fit(X_train_flattened, y_train)

predictions = model.predict(X_test_flattened)

accuracy = model.score(X_test_flattened, y_test)
print("Accuracy:", accuracy)

plt.figure(figsize=(10, 8))
num_images = 10
for i in range(num_images):
    plt.subplot(2, 5, i + 1)
    if X_test[i].shape[-1] == 3:
        plt.imshow(X_test[i])
    else:
        plt.imshow(X_test[i], cmap='gray')
    plt.title(f"Actual: {y_test[i]}, Predicted: {predictions[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()
