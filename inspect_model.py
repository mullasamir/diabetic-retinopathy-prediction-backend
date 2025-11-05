from tensorflow.keras.models import load_model

model = load_model("model/custom_cnn.h5")
model.summary()
