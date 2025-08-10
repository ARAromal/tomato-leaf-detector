# train_model.py

import tensorflow as tf
import matplotlib.pyplot as plt
import os

# --- 1. DEFINE CONSTANTS AND PATHS ---
IMG_SIZE = (224, 224) # MobileNetV2 requires 224x224 images
BATCH_SIZE = 32
DATA_DIR = os.path.join('tomato', 'tomato') # Path from your project folder to the 'train' and 'val' folders

train_dir = os.path.join(DATA_DIR, 'train')
validation_dir = os.path.join(DATA_DIR, 'val')

print("--- Data Loading ---")

# --- 2. LOAD DATASET AND PREPARE IT ---
# Use image_dataset_from_directory to load the images
train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    validation_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

# Get class names
class_names = train_dataset.class_names
print("Classes found:", class_names)
NUM_CLASSES = len(class_names)

# Configure dataset for performance
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)

# --- 3. BUILD THE MODEL USING TRANSFER LEARNING ---
print("\n--- Model Building ---")

# Define a simple data augmentation layer
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.2),
])

# Load the pre-trained MobileNetV2 model
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False, # Don't include the final classification layer
    weights='imagenet'
)

# Freeze the base model's layers so we don't retrain them
base_model.trainable = False

# Create our new model on top
inputs = tf.keras.Input(shape=(224, 224, 3))
x = data_augmentation(inputs) # Apply augmentation
x = tf.keras.applications.mobilenet_v2.preprocess_input(x) # Preprocess for MobileNetV2
x = base_model(x, training=False) # Run the base model
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x) # Add dropout for regularization
outputs = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x) # Our new classifier

model = tf.keras.Model(inputs, outputs)

# --- 4. COMPILE THE MODEL ---
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# --- 5. TRAIN THE MODEL ---
print("\n--- Starting Training ---")
EPOCHS = 10
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS
)

# --- 6. SAVE THE FINAL MODEL ---
print("\n--- Training complete. Saving model... ---")
model.save('tomato_leaf_disease_detector.keras')
print("Model saved as tomato_leaf_disease_detector.keras")

# --- 7. (Optional) Plot Training History ---
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,2.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.savefig('training_history.png')
print("Training history plot saved as training_history.png")