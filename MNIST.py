#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow_model_optimization as tfmot
import numpy as np
import os 
from tensorflow_model_optimization.sparsity import keras as sparsity
from tensorflow.keras.models import load_model
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
print('Librery Imported')

# import dataset 
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

sample_index = 1
print(y_train[sample_index])
plt.figure()
plt.imshow(X_train[sample_index], cmap='Greys')
plt.title('Label:'+str(y_train[sample_index]))
plt.savefig('sample.png')
plt.close()


# reshpae the input file
print('Input train shape', X_train.shape)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
print('Input train reshape', X_train.shape)
print('Input test shape', X_test.shape)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
print('Input test reshape', X_test.shape)
input_tensor_shape = (X_train.shape[1], X_train.shape[2],1)
print('input tensor shape:', input_tensor_shape)

# Normalize input 
print('Input normalization')
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255 
X_test /= 255

# Build models
quantize_model = tfmot.quantization.keras.quantize_model

model = keras.Sequential()

model.add(keras.layers.Conv2D(28, kernel_size=(3,3), input_shape=input_tensor_shape ))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation=tf.nn.relu))
model.add(keras.layers.Dense(10, activation=tf.nn.softmax))

print('model summary:',model.summary())
model_q = quantize_model(model)
print('q model:', model_q.summary())

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model_q.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

batch_size = 100
epochs = 5

print('Train Model')
history = model.fit(x=X_train, y=y_train, epochs=epochs, batch_size=batch_size)
history_q = model_q.fit(x=X_train, y=y_train, epochs=epochs, batch_size=batch_size)

print('Export History')
plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history_q.history['accuracy'])
plt.title('model vs q model training acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['model', 'model q'])
plt.savefig('training.png')
plt.close()

print('Model Evaluate')
result = model.evaluate(X_test, y_test)
result_q = model_q.evaluate(X_test, y_test)
print('model has an acc {0:.2f}%'.format(result[1] * 100))
print('model q has an acc {0:.2f}%'.format(result_q[1] * 100))

print('Model convertion and save')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
tflite_model = converter.convert()
open("MNIST_model_no_optimizations.tflite", "wb").write(tflite_model)

# Optimized model size
converter_2 = tf.lite.TFLiteConverter.from_keras_model(model)
converter_2.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
converter_2.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_model_2 = converter_2.convert()
open("MNIST_model_with_size_optim.tflite", "wb").write(tflite_model_2)

# Optimized model performance 
converter_3 = tf.lite.TFLiteConverter.from_keras_model(model)
converter_3.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
converter_3.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model_3 = converter_3.convert()
open("MNIST_model_with_default_optim.tflite", "wb").write(tflite_model_3)


def rep_db_gen():
    for image in X_test:
        arr = np.array(image)
        arr = np.expand_dims(arr, axis=0)
        yield ([arr])


# full quantization 
converter_4 = tf.lite.TFLiteConverter.from_keras_model(model)
converter_4.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter_4.optimizations = [tf.lite.Optimize.DEFAULT]
converter_4.inference_input_type = tf.uint8
converter_4.inference_output_type = tf.uint8
converter_4.representative_dataset = rep_db_gen
tflite_model_4 = converter_4.convert()
open("MNIST_full_quantization.tflite", "wb").write(tflite_model_4)


# Prunning 
end_step = np.ceil(1.0 * X_train.shape[0] / batch_size).astype(np.int32).astype(np.int32) * epochs
print('end step',end_step)

pruning_params = {
    'pruning_schedule': sparsity.PolynomialDecay(initial_sparsity=0.5,
                                                 final_sparsity=0.9,
                                                 begin_step=0,
                                                 end_step=end_step,
                                                 frequency=100
                                                 )
}

new_pruned_model = sparsity.prune_low_magnitude(model, **pruning_params)
print('purine model \n:', new_pruned_model.summary())

new_pruned_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

callbacks = [sparsity.UpdatePruningStep(),sparsity.PruningSummaries(log_dir=os.path.abspath(os.getcwd()), profile_batch=0)]

new_pruned_model.fit(X_train, y_train,
                     batch_size=batch_size,
                     epochs=epochs,
                     verbose=1,
                     callbacks=callbacks)

score = new_pruned_model.evaluate(X_test, y_test, verbose=0)
print("Test loss: {}".format(score[0]))
print("Test accuracy: {}".format(score[1]))

print('new pruned model', new_pruned_model.summary())
best_model_pruned = sparsity.strip_pruning(new_pruned_model)
print('best pruned model',best_model_pruned.summary())

for i, w in enumerate(best_model_pruned.get_weights()):
    print("{} -- Total:{}, Zeros: {:.2f}%".format(
        model.weights[i].name, w.size, np.sum(w == 0) / w.size * 100 ))

# Convert the model 
converter_5 = tf.lite.TFLiteConverter.from_keras_model(best_model_pruned)
converter_5.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter_5.optimizations = [tf.lite.Optimize.DEFAULT]
converter_5.inference_input_type = tf.uint8
converter_5.inference_output_type = tf.uint8
converter_5.representative_dataset = rep_db_gen 
tflite_model_5 = converter_5.convert()
open("MNIST_pruned.tflite", "wb").write(tflite_model_5)
