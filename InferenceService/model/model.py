import numpy as np
import tensorflow as tf

# 1. Model definition
inputs = tf.keras.layers.Input(shape=(28, 28, 1), name="input")
x = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)
x = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)   # Output shape: (64,)

# Output embedding (vector from the backbone)
embedding = tf.keras.layers.Lambda(lambda t: t, name="embedding")(x)

# Output probabilities (softmax)
probabilities = tf.keras.layers.Dense(10, activation="softmax", name="probabilities")(x)

# Output predicted class (argmax) – integer
predicted_class = tf.keras.layers.Lambda(
    lambda t: tf.argmax(t, axis=-1, output_type=tf.int32), 
    name="predicted_class"
)(probabilities)

# Multi‑output model with three outputs
model = tf.keras.Model(
    inputs=inputs,
    outputs={
        "probabilities": probabilities,   # vector of 10 floats
        "embedding": embedding,           # vector of 64 floats (reduced size)
        "predicted_class": predicted_class,  # integer (0-9)
    },
)

# 2. Compilation – only the loss on probabilities contributes to training
model.compile(
    optimizer="adam",
    loss={
        "probabilities": "sparse_categorical_crossentropy",
        "embedding": "mse",               # dummy
        "predicted_class": "mse",         # dummy (non‑differentiable)
    },
    loss_weights={
        "probabilities": 1.0,
        "embedding": 0.0,                 # ignored
        "predicted_class": 0.0,           # ignored
    },
    metrics={"probabilities": "accuracy"},
)

# 3. Fake training
print("Fake training...")
x_train = np.random.random((100, 28, 28, 1)).astype("float32")
y_train = np.random.randint(10, size=(100,))

# Dummy targets for outputs that should not contribute to the loss
# Embedding dimension is now 64 (was 5408)
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

# 4. Export for KServe / TensorFlow Serving
export_path = "./model_repo/1"
model.export(export_path)

print(f"Multi‑output model saved to: {export_path}")
"""
#mock device
inputs = tf.keras.layers.Input(shape=(28, 28, 1), name="input")
x = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)
x = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
embedding = tf.keras.layers.Dense(128, activation="relu", name="embedding")(x) #need emebedding
model = tf.keras.Model(inputs=inputs, outputs=embedding)

# 2. Compilazione
# Anche se è un estrattore, Keras richiede la compilazione per il training fittizio
model.compile(
    optimizer="adam",
    loss="mse" # Loss generica per l'embedding
)

# mock training, weight inizialization
print("Inizializzazione pesi...")
x_train = np.random.random((100, 28, 28, 1)).astype("float32")
y_train_dummy = np.random.random((100, 128)).astype("float32")

model.fit(x_train, y_train_dummy, epochs=1, verbose=0)

# 4. saving for KServe / TensorFlow Serving
export_path = "./model_repo/1"

model.export(export_path)

print(f"\nModello saved into: {export_path}")
"""