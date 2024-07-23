from tensorflow.keras.preprocessing import image
import tensorflow as tf

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch dimension
    img_array = img_array / 255.0  # Rescale to [0,1]
    return img_array
