# Age and Gender Prediction from Images using CNN + Transfer Learning

# 📦 1. Import Libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import cv2
from tqdm import tqdm

# 📁 2. Load and Prepare Dataset
# UTKFace filenames format: [age]_[gender]_[race]_[date&time].jpg

data_dir = "UTKFace/"
image_size = 128
images = []
ages = []
genders = []

for filename in tqdm(os.listdir(data_dir)):
    try:
        parts = filename.split("_")
        age = int(parts[0])
        gender = int(parts[1])  # 0: male, 1: female

        img_path = os.path.join(data_dir, filename)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (image_size, image_size))
        img = img / 255.0

        images.append(img)
        ages.append(age)
        genders.append(gender)
    except:
        continue

X = np.array(images)
y_age = np.array(ages)
y_gender = to_categorical(genders, 2)  # Binary classification

# Yaşları normalize etmek (opsiyonel): y_age = y_age / 100

# 🧪 3. Train-Test Split
X_train, X_test, y_age_train, y_age_test, y_gender_train, y_gender_test = train_test_split(
    X, y_age, y_gender, test_size=0.2, random_state=42
)

# 🔄 4. Image Augmentation
train_datagen = ImageDataGenerator(
    rotation_range=15,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
train_generator = train_datagen.flow(X_train, {'age_output': y_age_train, 'gender_output': y_gender_train}, batch_size=32)

# 🧠 5. Model Architecture (Multi-output with MobileNetV2)
base_model = MobileNetV2(input_shape=(image_size, image_size, 3), include_top=False, weights='imagenet')
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)

# Age output (regression)
age_output = Dense(1, activation='linear', name='age_output')(x)

# Gender output (classification)
gender_output = Dense(2, activation='softmax', name='gender_output')(x)

model = Model(inputs=base_model.input, outputs=[age_output, gender_output])

model.compile(
    loss={'age_output': 'mse', 'gender_output': 'categorical_crossentropy'},
    optimizer=Adam(learning_rate=0.0001),
    metrics={'age_output': 'mae', 'gender_output': 'accuracy'}
)

model.summary()

# 🏋️ 6. Training
callbacks = [EarlyStopping(patience=5, restore_best_weights=True)]

history = model.fit(
    train_generator,
    validation_data=(X_test, {'age_output': y_age_test, 'gender_output': y_gender_test}),
    epochs=30,
    callbacks=callbacks
)

# 📊 7. Evaluation and Visualization
losses = history.history

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(losses['age_output_mae'], label='Train MAE')
plt.plot(losses['val_age_output_mae'], label='Val MAE')
plt.title('Age Prediction MAE')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(losses['gender_output_accuracy'], label='Train Accuracy')
plt.plot(losses['val_gender_output_accuracy'], label='Val Accuracy')
plt.title('Gender Prediction Accuracy')
plt.legend()

plt.show()

# 🔍 8. Predict on Sample
sample_img = X_test[0].reshape(1, image_size, image_size, 3)
age_pred, gender_pred = model.predict(sample_img)

print(f"Predicted Age: {int(age_pred[0][0])}")
print(f"Predicted Gender: {'Male' if np.argmax(gender_pred[0]) == 0 else 'Female'}")
