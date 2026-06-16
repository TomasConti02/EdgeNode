import numpy as np
import tensorflow as tf
"""
A real model is too heavy for my testing setting
The real CV model have to be something like a RestNet architecture 
activation of 224 x 224
and output emedding of 512 with the respective classification
"""
# MOCK MODEL DEFINITION
inputs = tf.keras.layers.Input(shape=(28, 28, 1), name="input")
x = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)
x = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)   # Output shape: (64,) of the embedding !!!!

embedding = tf.keras.layers.Lambda(lambda t: t, name="embedding")(x) #backbone embedding vector
probabilities = tf.keras.layers.Dense(10, activation="softmax", name="probabilities")(x) #softmax for prob distribution classification execution 
# output will be the max from the softmax prob distribution
predicted_class = tf.keras.layers.Lambda(
    lambda t: tf.argmax(t, axis=-1, output_type=tf.int32), 
    name="predicted_class"
)(probabilities)

# modle have 3 output !!!!!!!!!!!!!!!
model = tf.keras.Model(
    inputs=inputs,
    outputs={
        "probabilities": probabilities,   # vector of 10 floats
        "embedding": embedding,           # vector of 64 floats (reduced size)
        "predicted_class": predicted_class,  # integer (0-9)
    },
)

# 2. Compilation only the loss on probabilities contributes to training !!!!!!
model.compile(
    optimizer="adam", #standard
    loss={
        "probabilities": "sparse_categorical_crossentropy", 
        "embedding": "mse",               # dummy
        "predicted_class": "mse",         # dummy 
    },
    loss_weights={
        "probabilities": 1.0,
        "embedding": 0.0,                 # ignored
        "predicted_class": 0.0,           # ignored
    },
    metrics={"probabilities": "accuracy"},
)

# Fake training
print("Fake training...")
x_train = np.random.random((100, 28, 28, 1)).astype("float32")
y_train = np.random.randint(10, size=(100,))

dummy_embedding = np.zeros((100, 64), dtype=np.float32)
dummy_predicted_class = np.zeros((100,), dtype=np.int32)

model.fit(
    x_train,
    {
        "probabilities": y_train,
        "embedding": dummy_embedding,
        "predicted_class": dummy_predicted_class,
    },
    epochs=1,
)
# SUPER IMPORTANT !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# 4. Export for KServe / TensorFlow Serving
export_path = "./model_repo/1"
model.export(export_path)

print(f"model saved on: {export_path}")
