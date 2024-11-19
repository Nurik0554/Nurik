import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report



(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

x_train = tf.image.resize(x_train, (32, 32))
x_test = tf.image.resize(x_test, (32, 32))

base_model = MobileNetV2(input_shape=(32, 32, 3), include_top=False, weights="imagenet")
base_model.trainable = False

baseline_model = models.Sequential([
    layers.Conv2D(3, (3, 3), padding="same"),  # Преобразуем в 3 канала для MobileNetV2
    base_model,
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(10, activation="softmax")
])

baseline_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

print("Training Baseline Model...")
baseline_model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

enhanced_model = models.Sequential([
    layers.Conv2D(3, (3, 3), padding="same"),
    base_model,
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(10, activation="softmax")
])

enhanced_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

print("Training Enhanced Model...")
enhanced_model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

baseline_loss, baseline_acc = baseline_model.evaluate(x_test, y_test, verbose=0)
enhanced_loss, enhanced_acc = enhanced_model.evaluate(x_test, y_test, verbose=0)



baseline_predictions = baseline_model.predict(x_test)
enhanced_predictions = enhanced_model.predict(x_test)
baseline_labels = baseline_predictions.argmax(axis=1)
enhanced_labels = enhanced_predictions.argmax(axis=1)
true_labels = y_test.argmax(axis=1)
print(f"Baseline Model Accuracy: {baseline_acc * 100:.2f}%")
print(f"Enhanced Model Accuracy: {enhanced_acc * 100:.2f}%")
print("Baseline Model Metrics:")
print(classification_report(true_labels, baseline_labels, digits=4))
print("Enhanced Model Metrics:")
print(classification_report(true_labels, enhanced_labels, digits=4))
