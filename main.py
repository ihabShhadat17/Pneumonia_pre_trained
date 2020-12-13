import tensorflow as tf

from model.fine_tuning import TransferLearning, plot_history, MODELS, plot_cm

transfer_learning = TransferLearning()
for model in MODELS:
    print(model)
model_name = input('SELECT PRE_TRAINED MODEL:')
base_model, input_shape = transfer_learning.get_fine_tuning_model(model_name)
train_dataset, validation_dataset = transfer_learning.prepare_dataset()
print(train_dataset.class_indices)
print(sum(validation_dataset.labels) / validation_dataset.samples)
print(sum(train_dataset.labels) / train_dataset.samples)

base_model.trainable = False
base_model.summary()
inputs = tf.keras.Input(shape=input_shape)

x = base_model(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(1)(x)
model = tf.keras.Model(inputs, outputs)
base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()
len(model.trainable_variables)
initial_epochs = 50

loss0, accuracy0 = model.evaluate(validation_dataset)
history = model.fit(train_dataset,
                    epochs=initial_epochs,
                    validation_data=validation_dataset)

plot_history(history, model_name)
base_model.trainable = False
# Let's take a look to see how many layers are in the base model
print("Number of layers in the base model: ", len(base_model.layers))

# Fine-tune from this layer onwards
fine_tune_at = 25

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = True
fine_tune_epochs = 250
total_epochs = initial_epochs + fine_tune_epochs

history_fine = model.fit(train_dataset,
                         epochs=total_epochs,
                         initial_epoch=history.epoch[-1],
                         validation_data=validation_dataset)
plot_history(history_fine, model_name + '_FineTuning', initial_epochs=initial_epochs, is_fine_tuning=True)
ypre = model.predict(validation_dataset)

for name, value in zip(model.metrics_names, ypre):
    print(name, ': ', value)

plot_cm(model_name, validation_dataset.labels, ypre)
model.save('models/' + model_name + '.h5')
