import tensorflow as tf

#Fashion dataset, labels 0-9 for type of clothing, 28x28 pixel image data, 70k images
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

#normalize, leading to simpler data for network -> better accuracy, less time
training_images  = training_images / 255.0
test_images = test_images / 255.0

#first layer should be same shape as data, last layer should be same shape as output
#flatten takes 28x28 image data -> 1D array, needed to simplify shape of data, otherwise we would need 28x28 neurons in first layer
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(), 
                                    tf.keras.layers.Dense(512, activation=tf.nn.relu), 
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

model.compile(optimizer = tf.keras.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=10)
model.evaluate(test_images, test_labels)

classifications = model.predict(test_images)
#classifications[0] is a list of probabilities that test_image[0] is type 0,1,...9, essentially gives probability that image is under which classfication
print(classifications[0])
print(test_labels[0])