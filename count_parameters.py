
from keras.models import Sequential
from keras.layers import Conv2D
from keras.models import Sequential
from keras.layers import Dense
from keras.utils.vis_utils import plot_model
import numpy as np

if __name__ == '__main__':
    img_rows, img_cols = 224, 224
    colors = 3
    input_size = img_rows * img_cols * colors
    input_shape = (img_rows, img_cols, colors)
    num_classes = 10

    # number bof parameters = output_size * (input_size + 1 )
    model = Sequential([
        # parameters = 32 * (150528 + 1) = 4816928
        Dense(32, activation='relu', input_shape=(input_size,)),
        # parameters = 64 * ( 32 + 1 ) = 2112
        Dense(64, activation='relu'),
        # parameters = 128 * (64 + 1 ) = 8320
        Dense(128, activation='relu'),
        # parameters = 10 * (128 + 1 ) = 1290
        Dense(num_classes, activation='softmax')
    ])
    model.summary()
    plot_model(model,to_file='model_plot.png',show_shapes=True,show_layer_names=True)

    # --------------convolution layers------------
    # number of parameters = output channels * (input_channels * window_size + 1)
    model = Sequential([
        # number of filters is the number of output channels because it is taking dot
        # product for each filters
        # parameters = 32 * (3* (3*3) +1 ) = 896
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        # parameters = 64 * (32 * (3*3 ) + 1 ) = 18496
        Conv2D(64, (3, 3), activation='relu'),
        Conv2D(128, (3, 3), activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.summary()
    plot_model(model, to_file='model_plot2.png', show_shapes=True, show_layer_names=True)
