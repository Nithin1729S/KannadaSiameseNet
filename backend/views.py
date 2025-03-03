from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Input, Flatten, Dense, Lambda
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam


# Corrected Initializers using TensorFlow
def initialize_bias(shape, dtype=None):
    return tf.keras.initializers.RandomNormal(mean=0.5, stddev=1e-2)(shape)

def initialize_weights(shape, dtype=None):
    return tf.keras.initializers.RandomNormal(mean=0.0, stddev=1e-2)(shape)

def get_siamese_model(input_shape):
    # Define the tensors for the two input images
    left_input = Input(shape=input_shape)
    right_input = Input(shape=input_shape)

    # Shared CNN model
    x = Conv2D(64, (10, 10), activation='relu', kernel_initializer=initialize_weights,
               kernel_regularizer=l2(2e-4))(left_input)
    x = MaxPooling2D()(x)
    x = Conv2D(128, (7, 7), activation='relu', kernel_initializer=initialize_weights,
               bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4))(x)
    x = MaxPooling2D()(x)
    x = Conv2D(128, (4, 4), activation='relu', kernel_initializer=initialize_weights,
               bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4))(x)
    x = MaxPooling2D()(x)
    x = Conv2D(256, (4, 4), activation='relu', kernel_initializer=initialize_weights,
               bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4))(x)
    x = Flatten()(x)
    x = Dense(4096, activation='sigmoid', kernel_regularizer=l2(1e-3),
              kernel_initializer=initialize_weights, bias_initializer=initialize_bias)(x)

    # Shared model definition
    shared_model = Model(left_input, x)

    # Generate the encodings (feature vectors) for the two images
    encoded_l = shared_model(left_input)
    encoded_r = shared_model(right_input)

    # Add a customized layer to compute the absolute difference between encodings
    L1_distance = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))([encoded_l, encoded_r])

    # Add a dense layer with a sigmoid activation for similarity score
    prediction = Dense(1, activation='sigmoid', bias_initializer=initialize_bias)(L1_distance)

    # Connect the inputs with the outputs
    siamese_net = Model(inputs=[left_input, right_input], outputs=prediction)

    return siamese_net

# Example Usage
input_shape = (105, 105, 1)
model = get_siamese_model(input_shape)
model.summary()


optimizer = Adam(learning_rate=0.00006)
model.compile(loss="binary_crossentropy", optimizer=optimizer)

model.load_weights('model/weights.h5')
model.summary()

def similarity_score():
    pass