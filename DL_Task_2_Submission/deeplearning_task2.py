import numpy as np
import matplotlib.pyplot as plt
import os

from keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.optimizers import Adam

# Create output directory
output_dir = "DL_Task2_Outputs"
os.makedirs(output_dir, exist_ok=True)

# Load dataset
(trainX, trainy), (testX, testy) = fashion_mnist.load_data()
print('Train data shape:', trainX.shape)
print('Test data shape:', testX.shape)

# Visualize and save sample images
plt.figure(figsize=(8, 8))
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(trainX[i], cmap='gray')
    plt.title(f"Label: {trainy[i]}")
    plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "sample_images.png"))
plt.show()

# Normalize and reshape input data
trainX = trainX.astype('float32') / 255.0
testX = testX.astype('float32') / 255.0
trainX = np.expand_dims(trainX, -1)
testX = np.expand_dims(testX, -1)

# Build the CNN model
def create_model():
    model = Sequential()
    model.add(Conv2D(64, (5, 5), padding="same", activation="relu", input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (5, 5), padding="same", activation="relu"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(256, (5, 5), padding="same", activation="relu"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    return model

model = create_model()
model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model
history = model.fit(trainX, trainy, epochs=10, validation_split=0.2, batch_size=64)

# Save model weights
model.save_weights(os.path.join(output_dir, "fashion_model_weights.h5"))

# Plot and save Accuracy graph
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "accuracy_plot.png"))
plt.show()

# Plot and save Loss graph
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "loss_plot.png"))
plt.show()

# Make prediction on a sample image
labels = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
sample_index = 0
sample = testX[sample_index:sample_index+1]
prediction = model.predict(sample)
predicted_label = labels[np.argmax(prediction)]

# Show and save prediction result
plt.imshow(testX[sample_index].reshape(28, 28), cmap='gray')
plt.title(f"Predicted: {predicted_label}")
plt.axis('off')
plt.savefig(os.path.join(output_dir, "sample_prediction.png"))
plt.show()

print(f"\n✅ Prediction complete! Result: {predicted_label}")