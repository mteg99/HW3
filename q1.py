import q1_api as q1
import tensorflow as tf

# Generate datasets
D100_train, L100_train = q1.generate_dataset(100)
D200_train, L200_train = q1.generate_dataset(200)
D500_train, L500_train = q1.generate_dataset(500)
D1k_train, L1k_train = q1.generate_dataset(1000)
D2k_train, L2k_train = q1.generate_dataset(2000)
D5k_train, L5k_train = q1.generate_dataset(5000)
D100k_test, L100k_test = q1.generate_dataset(100000)

# Plot datasets
# q1.plot_dataset(D100_train, L100_train, '100 Training Samples')
# q1.plot_dataset(D200_train, L200_train, '200 Training Samples')
# q1.plot_dataset(D500_train, L500_train, '500 Training Samples')
# q1.plot_dataset(D1k_train, L1k_train, '1000 Training Samples')
# q1.plot_dataset(D2k_train, L2k_train, '2000 Training Samples')
# q1.plot_dataset(D5k_train, L5k_train, '5000 Training Samples')
# q1.plot_dataset(D100k_test, L100k_test, '100000 Training Samples')

# Optimal Classifier
# print('Optimal Pr(error):')
# print(q1.optimal_pr_error(D100k_test, L100k_test))

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(4, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(4, activation=tf.nn.softmax))

model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=[tf.keras.metrics.CategoricalAccuracy()])
model.fit(D5k_train.T, tf.keras.utils.to_categorical(L5k_train), batch_size=32, epochs=100)
loss, accuracy = model.evaluate(D100k_test.T, tf.keras.utils.to_categorical(L100k_test), batch_size=32)
pr_error = 1 - accuracy
