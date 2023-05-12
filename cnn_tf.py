# Multilayer perceptron made with TensorFlow
import seaborn as sns
import tensorflow as tf
from keras import layers
from matplotlib import pyplot as plt

from utils import load_data, evalute_predictions

sns.set()
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

(matrices_train, matrices_test, matrices_val,
 class_train, class_test, class_val,
 features_train, features_test, features_val, cutoffs,
 true_features_train, true_features_test, true_features_val) = load_data(mode='qgs_gm_pr_v2',
                                                                         test_share=0.2,
                                                                         val_share=0.2)

# Reshape matrices to their original shape (16, 16, 2)
matrices_train = matrices_train.reshape(-1, 16, 16, 2)
matrices_test = matrices_test.reshape(-1, 16, 16, 2)
matrices_val = matrices_val.reshape(-1, 16, 16, 2)

mean = matrices_train.mean(axis=0)
std = matrices_train.std(axis=0)
std[std == 0] = 1  # avoid division by zero
matrices_train = (matrices_train - mean) / std
matrices_test = (matrices_test - mean) / std
matrices_val = (matrices_val - mean) / std

activation = 'relu'
x = model_input = tf.keras.Input(shape=(16, 16, 2))

x = layers.Conv2D(32, (3, 3), activation=activation)(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(64, (3, 3), activation=activation)(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(64, (3, 3), activation=activation)(x)
x = layers.MaxPooling2D((2, 2))(x)

x = layers.Flatten()(x)

# Concatenate extra features
extra_features_input = tf.keras.Input(shape=(2,))
x = layers.Concatenate()([x, extra_features_input])
x = layers.Dense(512, activation=activation)(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(512, activation=activation)(x)
x = layers.Dropout(0.2)(x)

out_layer = layers.Dense(2, activation='softmax')(x)
model = tf.keras.Model(inputs=[model_input, extra_features_input], outputs=[out_layer])
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)

model.summary()
history = model.fit([matrices_train, features_train[:, [5, 6]]],
                    class_train,
                    validation_data=([matrices_val, features_val[:, [5, 6]]], class_val),
                    epochs=500,
                    shuffle=True,
                    verbose=1,
                    batch_size=1024,
                    callbacks=[
                        tf.keras.callbacks.TensorBoard('logs/baseline_cnn'),
                        tf.keras.callbacks.ReduceLROnPlateau(
                            monitor='val_accuracy', factor=0.5, patience=15, verbose=1,
                            min_delta=0.0001, cooldown=0
                        ),
                        tf.keras.callbacks.EarlyStopping(
                            monitor='val_accuracy', min_delta=0, patience=65, verbose=1),
                        tf.keras.callbacks.ModelCheckpoint(
                            'best_weights.h5', monitor='val_accuracy', save_best_only=True, mode='max')
                    ],
                    )
plt.figure(figsize=(15, 5))
plt.title('Loss')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.legend()
plt.show()
plt.figure(figsize=(15, 5))
plt.title('Accuracy')
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.legend()
plt.show()
model.load_weights('best_weights.h5')
model.evaluate(matrices_test, class_test)
preds = model.predict(matrices_test, batch_size=1024).argmax(1)
evalute_predictions(preds, true_features_test)
