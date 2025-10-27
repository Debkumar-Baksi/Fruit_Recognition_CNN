import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the saved model
model = tf.keras.models.load_model('fruit_recognition_model.h5')

# Class labels (must match the folder names used during training)
class_labels = ['Apples', 'Mangoes', 'Oranges']

# Path to the image you want to test
img_path = 'FruitsDB/Mangoes/Test/Mango_14_A.JPG'   # replace with your image file path

# Load and preprocess the image
img = image.load_img(img_path, target_size=(128, 128))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # convert to batch format
img_array = img_array / 255.0  # normalize

# Make prediction
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions)

# Print the result
print(f"Predicted Fruit: {class_labels[predicted_class]}")
