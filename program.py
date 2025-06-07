'''Task 3:-
Implement a support vector machine (SVM) to classify images of cats and dogs from the Kaggle dataset.'''
import os
import cv2
import random
import numpy as np
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Update this path to point to your dataset folder
DATA_DIR = r"C:\Users\Dell\OneDrive\Desktop\task\dogs-vs-cats\train"
SAMPLE_SIZE = 2000
IMG_SIZE = 64  # Resize images to 64x64

def label_img(filename):
    return 0 if "cat" in filename.lower() else 1

def load_data():
    data = []
    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.jpg')]
    random.shuffle(files)  # Mix cat and dog images
    files = files[:SAMPLE_SIZE]

    for f in tqdm(files, desc="Loading data"):
        try:
            label = label_img(f)
            img_path = os.path.join(DATA_DIR, f)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            data.append([img.flatten(), label])
        except Exception as e:
            print(f"Skipping {f}: {e}")
            continue
    return data

# Load data
print("Loading data...")
data = load_data()
X = [i[0] for i in data]
y = [i[1] for i in data]

# Debug: Show class distribution
print("✅ Unique classes in data:", set(y))
print("✅ Count - Cats (0):", y.count(0))
print("✅ Count - Dogs (1):", y.count(1))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train SVM
print("Training SVM...")
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)

# Evaluate
print("Evaluating...")
preds = svm.predict(X_test)
print(classification_report(y_test, preds))

# Visualize some predictions
for i in range(5):
    idx = random.randint(0, len(X_test) - 1)
    img = X_test[idx].reshape(IMG_SIZE, IMG_SIZE)
    true_label = y_test[idx]
    pred_label = svm.predict([X_test[idx]])[0]
    
    plt.imshow(img, cmap='gray')
    plt.title(f"True: {'Cat' if true_label == 0 else 'Dog'} | Pred: {'Cat' if pred_label == 0 else 'Dog'}")
    plt.axis('off')
    plt.show()