import tensorflow
import sklearn

# Load the 8x8 digit dataset
supervised_training_data = sklearn.datasets.load_digits()
image_data = supervised_training_data.images
correct_answers = supervised_training_data.target

# Normalize pixel values to [0, 1]
image_data = image_data / 16.0

# Reshape training data array to be (samples:N, height:8, width:8, channels:1)
image_data = image_data.reshape((-1, 8, 8, 1))

# One-hot encode labels (e.g "2" is converted to [0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
correct_vectors = tensorflow.keras.utils.to_categorical(correct_answers, num_classes=10)

# Split into training and test sets
training_questions, test_questions, training_answers, test_answers = sklearn.model_selection.train_test_split(image_data, correct_vectors, test_size=0.2)

# Build the Neural Network layer by layer starting from the input layer
model = tensorflow.keras.models.Sequential([
    tensorflow.keras.layers.Input(shape=(8, 8, 1)),
    tensorflow.keras.layers.Conv2D(filters=4, activation='relu', kernel_size=(3, 3), padding='same'),
    tensorflow.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tensorflow.keras.layers.Flatten(),
    tensorflow.keras.layers.Dense(units = 20, activation='relu'),
    tensorflow.keras.layers.Dense(units = 10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(training_questions, training_answers, epochs=100, verbose=0, batch_size=16)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_questions, test_answers)  
print('accuracy', test_accuracy)

# Save the model weights and full model
model.save("minimal_cnn100.keras")