import tensorflow as tf

from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

IMG_SIZE = 224

datagen = ImageDataGenerator(rescale=1./255)

test_data = datagen.flow_from_directory(
    "data/processed",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=32
)

model = load_model("models/best_model.h5")

loss, acc = model.evaluate(test_data)
print("Accuracy:", acc)