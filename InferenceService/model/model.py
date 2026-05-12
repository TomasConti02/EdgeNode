import tensorflow as tf
import numpy as np
import os

# 1. Definizione del modello come Feature Extractor
inputs = tf.keras.layers.Input(shape=(28, 28, 1), name="input")

# Blocco convolutivo 1
x = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)

# Blocco convolutivo 2
x = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)

# Global Average Pooling + embedding compatto
x = tf.keras.layers.GlobalAveragePooling2D()(x)
# Questo è il tuo output finale: un vettore di 128 caratteristiche
embedding = tf.keras.layers.Dense(128, activation="relu", name="embedding")(x)

# Modello a singolo output (solo embedding)
model = tf.keras.Model(inputs=inputs, outputs=embedding)

# 2. Compilazione
# Anche se è un estrattore, Keras richiede la compilazione per il training fittizio
model.compile(
    optimizer="adam",
    loss="mse" # Loss generica per l'embedding
)

# 3. Addestramento fittizio (necessario per inizializzare i pesi prima del salvataggio)
print("Inizializzazione pesi...")
x_train = np.random.random((100, 28, 28, 1)).astype("float32")
# Creiamo target fittizi con la stessa forma dell'embedding (128,)
y_train_dummy = np.random.random((100, 128)).astype("float32")

model.fit(x_train, y_train_dummy, epochs=1, verbose=0)

# 4. Esportazione per KServe / TensorFlow Serving
export_path = "./model_repo/1"
# model.export è il metodo raccomandato nelle versioni recenti di TF 2.13+
# In alternativa si usa model.save(export_path)
model.export(export_path)

print(f"\n✅ Modello estrattore salvato in: {export_path}")
print(f"L'output del modello sarà un tensore di forma (None, 128)")