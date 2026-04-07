import os
import numpy as np
import tensorflow as tf
import keras
from tensorflow.python.keras.utils import layer_utils
from keras.callbacks import ModelCheckpoint
from keras.metrics import MeanSquaredError
from keras.backend import clear_session
from keras.utils import get_custom_objects
from sklearn.linear_model import LinearRegression
from keras.callbacks import EarlyStopping

def error_bits(error):
    """
    Return a lower bound on the number of bits to encode the errors based on Shannon's source
    coding theorem:
    https://en.wikipedia.org/wiki/Shannon%27s_source_coding_theorem#Source_coding_theorem

    Use a Gaussian approximation to the distribution of errors using the root mean squared error.

    :param error: Vector or list of errors (error = estimate - actual)
    :return: The lower bound number of bits to encode the errors
    """
    rmse = np.mean(error ** 2) ** (1/2)
    if not np.isnan(rmse):
        entropy = max(np.log2(rmse) + 2, 0)
        bits = int(np.ceil(entropy * len(error)))
    else:
        return float('inf')
    return bits


def get_xy(tiff_file):
    import matplotlib.pyplot as plt
    """
    Read TIFF file and convert to data matrix, X and target vector, y.

    :param tiff_file: Image file name (string)
    :return: data matrix 'X' (129600-by-2), target vector 'y' (129600-by-1)
    """

    # Read the image
    image = plt.imread(tiff_file)

    # get the coordinates
    extent = tiff_file.split('_')[1][1:-1].replace(',', '').split(' ')
    extent = [float(ex) for ex in extent]
    lon0, lon1, lat0, lat1 = extent

    # get coordinate system:
    lons = np.linspace(lon0, lon1, image.shape[1])
    lats = np.linspace(lat0, lat1, image.shape[0])
    i, j = np.meshgrid(lats, lons, indexing='ij')

    # reshape row/column indexes into n-by-2 matrix
    x = np.concatenate((i.reshape((-1, 1)), j.reshape((-1, 1))), axis=1).astype('float32')

    # reshape elevations into n-by-1 vector
    y = image.reshape((-1, 1)).astype('float32')

    return x, y, extent


# bogus
def print_error(y_true, y_pred, num_params, name):
    """
    Print performance for this model.

    :param y_true: The correct elevations (in meters)
    :param y_pred: The estimated elevations by the model (in meters)
    :param num_params: The number of trainable parameters in the model
    :param name: The name of the model
    :return: None
    """

    # the error
    e = y_pred - y_true
    # the mean squared error
    mse = np.mean(e ** 2)
    # the lower bound for the number of bits to encode the errors per pixel
    num_pixels = len(e)
    error_bpp = error_bits(e) / num_pixels
    # the number of bits to encode the model per pixel
    desc_bpp = 32 * num_params / num_pixels
    # the total bits for the compressed image per pixel
    total_bpp = desc_bpp + error_bpp
    # comparison to a model that estimated mean(y) for every pixel
    error_bpp_0 = error_bits(y_true - y_true.mean()) / num_pixels
    desc_bpp_0 = 32 / num_pixels
    total_bpp_0 = error_bpp_0 + desc_bpp_0
    # percent improvement
    improvement = 1 - total_bpp/total_bpp_0
    msg = (
        f'{name + ":":12s} {mse:>11.4f} MSE, {error_bpp:>11.4f} error bits/px, {desc_bpp:>11.4f} model bits/px'
        f'{total_bpp:11.4f} total bits/px, {improvement:.2%} improvement'
    )
    print(msg)
    return mse, error_bpp, desc_bpp, error_bpp_0, desc_bpp_0, improvement, msg


# bogus
def num_parameters(model):
    """
    Compute the number of bits to encode the model based on the trainable parameters.
    (We ignore the bits required to encode the architecture for the model.)

    :param model: A keras model
    :return: The number of bits required to encode the model.
    """
    return layer_utils.count_params(model.trainable_weights)


def compare_images(model, x, y, extent, output_path='.'):
    import matplotlib.pyplot as plt
    """
    Use the model to estimate elevation image and show it compared
    to the original.

    :param model: A keras model with 2D input and 1 output
    :param x: The (i, j) coordinates
    :param y: The target vector
    :param output_path: Where to save the figure
    :return: None
    """
    # The model estimates (same shape as 'y')
    y_hat = model(x)
    num_params = num_parameters(model)

    # print some comparisons
    print_error(y, y.mean(), 1, 'Constant')
    print_error(y, LinearRegression().fit(x, y).predict(x), 3, 'Linear')
    mse, error_bpp, desc_bpp, error_bpp_0, desc_bpp_0, improvement, msg = \
        print_error(y, y_hat, num_params, 'Model')
    total_bpp = error_bpp + desc_bpp
    constant_bpp = error_bpp_0 + desc_bpp_0

    # The minimum and maximum elevation over original and estimated image
    vmin = min(y.min(), np.min(y_hat))
    vmax = max(y.max(), np.min(y_hat))

    # close all figure windows
    plt.close('all')

    # create a new figure window with room for 2 adjacent images and a color bar
    fig, ax = plt.subplots(1, 2, sharey='row', figsize=(10, 4))

    # render and label the model estimate
    ax[0].imshow(np.reshape(y_hat, (360, 360)), vmin=vmin, vmax=vmax, extent=extent)
    ax[0].set_title(f'Model Estimate MSE = {mse:.1f}\n'
                    f'{error_bpp:.4f} bpp error, {desc_bpp:.4f} bpp model\n'
                    f'{total_bpp:.4f} tot, '
                    f'{improvement:.2%} improvement')

    # render and label the original
    im1 = ax[1].imshow(y.reshape((360, 360)), vmin=vmin, vmax=vmax, extent=extent)
    ax[1].set_title('Original')
    ax[1].set_title(f'Baseline Estimate MSE = {np.mean((y - y.mean())**2):.1f}\n'
                    f'{error_bpp_0:.4f} bpp error, {desc_bpp_0:.4f} bpp model\n'
                    f'{constant_bpp:.4f} tot, '
                    f'{0:.2%} improvement')

    # add a color bar in a new set of axes
    fig.subplots_adjust(right=0.8)
    ax2 = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im1, cax=ax2)
    plt.savefig(os.path.join(output_path, 'result.png'), bbox_inches='tight')


class Entropy(MeanSquaredError):
    """
    This class provides a custom *metric* for the Bayesian Information Criterion.
    """
    def __init__(self, name='entropy', model_bpp=0, **kwargs):
        super(Entropy, self).__init__(name=name, **kwargs)
        self.model_bpp = model_bpp

    def result(self):
        """
        Compute a Gaussian approximation to entropy from the mean squared error
        :return: Entropy scalar
        """
        mse = super(Entropy, self).result()
        rmse = mse ** (1/2)
        entropy = tf.maximum(tf.math.divide(tf.math.log(rmse), tf.math.log(2.0)) + 2, 0)
        return entropy + self.model_bpp


def setup_model_checkpoints(output_path):
    """
    Setup model checkpoints using the save path and frequency.

    :param output_path: The directory to store the checkpoints in
    :return: a ModelCheckpoint
    """
    os.makedirs(output_path, exist_ok=True)

    model_checkpoint = ModelCheckpoint(
        os.path.join(output_path, 'model.{epoch:05d}_{entropy:.4f}.h5'),
        save_weights_only=False,
        save_freq='epoch',
        save_best_only=True,
        monitor='entropy',
        verbose=1
    )
    return model_checkpoint


def save_model(model, file_name):
    """
    Save a trained model.

    :param model: The keras model
    :param file_name: the name to use to save it (ending in .h5)
    :return: None
    """
    model.save(file_name)


def load_model(file_name):
    """
    Load a saved model optionally including the Entropy custom metric.

    :param file_name: The file name containing the model (ends in .h5)
    :return: A keras model ready for fitting or predicting
    """
    get_custom_objects()['Entropy'] = Entropy
    model = keras.models.load_model(
        file_name,
        compile=True,
        custom_objects={}
    )
    return model


def load_best_model(model_dir):
    import os
    # expecting this format: model.{epoch:05d}_{entropy:.4f}.h5
    min_entropy = float('inf')
    best_file = None
    for model_file in os.listdir(model_dir):
        if model_file.endswith('.h5'):
            entropy = float(model_file.split('_')[-1].split('.')[0])
            if entropy < min_entropy or best_file is None:
                best_file = model_file
    return load_model(model_dir + '/' + best_file)


def plot_history(history, output_path='.'):
    import matplotlib.pyplot as plt

    keys = [k for k in history.history.keys() if not k.startswith('val_')]

    num_cols = int(np.ceil(len(keys) ** (1/2)))
    num_rows = int(np.ceil(len(keys) / num_cols))
    plt.figure()
    for i, k in enumerate(keys):
        ax = plt.subplot(num_rows, num_cols, i+1)
        ax.plot(history.history[k], label=k)
        val_key = f'val_{k}'
        if val_key in history.history:
            ax.plot(history.history[k], label=val_key)
        plt.legend()
    plt.savefig(os.path.join(output_path, 'learning_curve.png'), bbox_inches='tight')

# new example
def example():
    """
    An minimal example training a linear regression model on the elevation data.

    :return: None
    """
    from keras.models import Sequential
    from keras.layers import Input, Dense, Normalization, Lambda
    from keras.optimizers import Adam

    # replace these with your actual TIFF file name.
    problem = 'terrain_(-93.0, -88.0, 44.0, 49.0)_50'
    problem = 'terrain_(-91.5, -89.5, 45.5, 47.5)_20'
    tiff_file = 'terrain/' + problem + '.tiff'

    # change the model name when you change the architecture or hyperparameters.
    model_name = 'linear'
    output_path = f'models/{problem}/{model_name}'

    model_checkpoint = setup_model_checkpoints(output_path)
    x, y, extent = get_xy(tiff_file)

    clear_session()

    # linear regression
    model = Sequential([
        Input(2),
        Dense(units=1, activation='linear'),
    ])
    model_bpp = num_parameters(model) * 32 / len(y)
    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.01), metrics=[Entropy(model_bpp=model_bpp)])
    model.summary()

    print_error(y, y.mean(), 1, 'Constant')

    history = model.fit(x, y, epochs=20, batch_size=1024, verbose=1, callbacks=[model_checkpoint])
    plot_history(history, output_path)

    compare_images(model, x, y, extent, output_path)

    # model = load_model('terrain/linear/model.00001_9.5105.h5 ( or your file)')


def attempt1():
    """
    A minimal example training a linear regression model on the elevation data.

    :return: None
    """
    from keras.models import Sequential
    from keras.layers import Input, Dense, Normalization, Lambda
    from keras.optimizers import Adam

    # replace these with your actual TIFF file name.
    problem = 'terrain_(-100.5, -98.5, 36.5, 38.5)_20'
    # problem = 'terrain_(-90.0, -85.0, 39.0, 44.0)_50'
    # problem = 'terrain_(-76.75, -76.25, 43.25, 43.75)_5'

    tiff_file = 'terrain/' + problem + '.tiff'
    # change the model name when you change the architecture or hyperparameters.
    model_name = 'attempt1'
    output_path = f'models/{problem}/{model_name}'
    os.makedirs(output_path, exist_ok=True)

    model_checkpoint = setup_model_checkpoints(output_path)
    x, y, extent = get_xy(tiff_file)
    clear_session()

    # linear regression
    model = Sequential()
    model.add(Dense(25000, activation='sigmoid', input_shape=(2,)))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.01), metrics=[Entropy()])
    model.summary()

    print_error(y, y.mean(), 1, 'Constant')

    history = model.fit(x, y, epochs=40, batch_size=2048, verbose=1, callbacks=[model_checkpoint])
    history = model.fit(x, y, epochs=20, batch_size=4096, verbose=1, callbacks=[model_checkpoint])
    plot_history(history, output_path)
    save_model(model, problem + '.h5')

    compare_images(model, x, y, extent, output_path)

    # model = load_model('terrain/linear/model.00001_9.5105.h5 ( or your file)')


def attempt2():
    """
    A minimal example training a linear regression model on the elevation data.

    :return: None
    """
    from keras.models import Sequential
    from keras.layers import Input, Dense, Normalization, Lambda
    from keras.optimizers import Adam

    # replace these with your actual TIFF file name.
    # problem = 'terrain_(-100.5, -98.5, 36.5, 38.5)_20'
    # problem = 'terrain_(-90.0, -85.0, 39.0, 44.0)_50'
    problem = 'terrain_(-76.75, -76.25, 43.25, 43.75)_5'

    tiff_file = 'terrain/' + problem + '.tiff'
    # change the model name when you change the architecture or hyperparameters.
    model_name = 'attempt2'
    output_path = f'models/{problem}/{model_name}'
    os.makedirs(output_path, exist_ok=True)

    model_checkpoint = setup_model_checkpoints(output_path)
    x, y, extent = get_xy(tiff_file)
    clear_session()

    # linear regression
    layer = Normalization(axis=1)
    layer.adapt(x)
    model = Sequential([
        Input(2),
        layer,
        Dense(16, activation='tanh'),
        Dense(8, activation='tanh'),
        Dense(2, activation='relu'),
        Dense(1, activation='linear'),
    ])
    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.01), metrics=[Entropy()])
    model.summary()

    print_error(y, y.mean(), 1, 'Constant')

    history = model.fit(x, y, epochs=25, batch_size=1024, verbose=1, callbacks=[model_checkpoint])
    history = model.fit(x, y, epochs=25, batch_size=4096, verbose=1, callbacks=[model_checkpoint])
    history = model.fit(x, y, epochs=30, batch_size=32768, verbose=1, callbacks=[model_checkpoint])
    plot_history(history, output_path)
    save_model(model, problem + '.h5')

    compare_images(model, x, y, extent, output_path)

    # model = load_model('terrain/linear/model.00001_9.5105.h5 ( or your file)')


def attempt3():
    """
    A minimal example training a linear regression model on the elevation data.

    :return: None
    """
    from keras.models import Sequential
    from keras.layers import Input, Dense, Normalization, Lambda
    from keras.optimizers import Adam

    # replace these with your actual TIFF file name.
    # problem = 'terrain_(-100.5, -98.5, 36.5, 38.5)_20'
    # problem = 'terrain_(-90.0, -85.0, 39.0, 44.0)_50'
    # problem = 'terrain_(-76.75, -76.25, 43.25, 43.75)_5'
    problem_array = ['terrain_(-100.5, -98.5, 36.5, 38.5)_20', 'terrain_(-90.0, -85.0, 39.0, 44.0)_50', 'terrain_(-76.75, -76.25, 43.25, 43.75)_5']
    for i in range(len(problem_array)):
        tiff_file = 'terrain/' + problem_array[i] + '.tiff'
        # change the model name when you change the architecture or hyperparameters.
        model_name = 'attempt3'
        output_path = f'models/{problem_array[i]}/{model_name}'
        os.makedirs(output_path, exist_ok=True)

        model_checkpoint = setup_model_checkpoints(output_path)
        x, y, extent = get_xy(tiff_file)
        clear_session()

        # linear regression
        layer = Normalization(axis=1)
        layer.adapt(x)
        model = Sequential([
            Input(2),
            layer,
            Dense(16, activation='tanh'),
            Dense(8, activation='tanh'),
            Dense(2, activation='relu'),
            Dense(1, activation='linear'),
        ])
        model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.01), metrics=[Entropy()])
        model.summary()

        print_error(y, y.mean(), 1, 'Constant')

        model.fit(x, y, epochs=25, batch_size=1024, verbose=0, callbacks=[model_checkpoint])
        model.fit(x, y, epochs=25, batch_size=4096, verbose=0, callbacks=[model_checkpoint])
        model.fit(x, y, epochs=25, batch_size=16384, verbose=0, callbacks=[model_checkpoint])
        history = model.fit(x, y, epochs=40, batch_size=32768, verbose=0, callbacks=[model_checkpoint])
        plot_history(history, output_path)
        save_model(model, problem_array[i] + '.h5')

        compare_images(model, x, y, extent, output_path)

    # model = load_model('terrain/linear/model.00001_9.5105.h5 ( or your file)')


def attempt4():
    """
    A minimal example training a linear regression model on the elevation data.

    :return: None
    """
    from keras.models import Sequential
    from keras.layers import Input, Dense, Normalization, Lambda
    from keras.optimizers import Adam

    # replace these with your actual TIFF file name.
    # problem = 'terrain_(-100.5, -98.5, 36.5, 38.5)_20'
    # problem = 'terrain_(-90.0, -85.0, 39.0, 44.0)_50'
    # problem = 'terrain_(-76.75, -76.25, 43.25, 43.75)_5'
    problem_array = ['terrain_(-100.5, -98.5, 36.5, 38.5)_20', 'terrain_(-90.0, -85.0, 39.0, 44.0)_50', 'terrain_(-76.75, -76.25, 43.25, 43.75)_5']
    for i in range(len(problem_array)):
        tiff_file = 'terrain/' + problem_array[i] + '.tiff'
        # change the model name when you change the architecture or hyperparameters.
        model_name = 'attempt4'
        output_path = f'models/{problem_array[i]}/{model_name}'
        os.makedirs(output_path, exist_ok=True)

        model_checkpoint = setup_model_checkpoints(output_path)
        x, y, extent = get_xy(tiff_file)
        clear_session()

        # linear regression
        layer = Normalization(axis=1)
        layer.adapt(x)
        model = Sequential([
            Input(2),
            layer,
            Dense(12, activation='tanh'),
            Dense(6, activation='tanh'),
            Dense(4, activation='relu'),
            Dense(1, activation='linear'),
        ])
        model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.01), metrics=[Entropy()])
        model.summary()

        print_error(y, y.mean(), 1, 'Constant')

        model.fit(x, y, epochs=25, batch_size=1024, verbose=0, callbacks=[model_checkpoint])
        model.fit(x, y, epochs=25, batch_size=4096, verbose=0, callbacks=[model_checkpoint])
        model.fit(x, y, epochs=25, batch_size=16384, verbose=0, callbacks=[model_checkpoint])
        history = model.fit(x, y, epochs=40, batch_size=32768, verbose=0, callbacks=[model_checkpoint])
        plot_history(history, output_path)
        save_model(model, problem_array[i] + '.h5')

        compare_images(model, x, y, extent, output_path)

    # model = load_model('terrain/linear/model.00001_9.5105.h5 ( or your file)')


def attempt5():
    """
    good for _50 file, got ~17.5%

    :return: None
    """
    from keras.models import Sequential
    from keras.layers import Input, Dense, Normalization, Lambda
    from keras.optimizers import Adam

    problem_array = ['terrain_(-100.5, -98.5, 36.5, 38.5)_20', 'terrain_(-90.0, -85.0, 39.0, 44.0)_50', 'terrain_(-76.75, -76.25, 43.25, 43.75)_5']
    # for i in range(len(problem_array)):
    #     tiff_file = 'terrain/' + problem_array[i] + '.tiff'
    i = 1
    tiff_file = 'terrain/' + problem_array[i] + '.tiff'
    # change the model name when you change the architecture or hyperparameters.
    model_name = 'attempt5'
    output_path = f'models/{problem_array[i]}/{model_name}'

    model_checkpoint = setup_model_checkpoints(output_path)
    x, y, extent = get_xy(tiff_file)

    clear_session()

    layer = Normalization(axis=1)
    layer.adapt(x)
    model = Sequential([
        Input(2),
        layer,
        Dense(16, activation='tanh'),
        Dense(8, activation='tanh'),
        Dense(5, activation='tanh'),
        Dense(3, activation='relu'),
        Dense(1, activation='relu'),
    ])
    model_bpp = num_parameters(model) * 32 / len(y)
    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.01), metrics=[Entropy(model_bpp=model_bpp)])
    model.summary()

    print_error(y, y.mean(), 1, 'Constant')

    history = model.fit(x, y, epochs=25, batch_size=1024, verbose=0, callbacks=[model_checkpoint])
    model.fit(x, y, epochs=25, batch_size=4096, verbose=0, callbacks=[model_checkpoint])
    model.fit(x, y, epochs=25, batch_size=16384, verbose=0, callbacks=[model_checkpoint])
    model.fit(x, y, epochs=40, batch_size=32768, verbose=0, callbacks=[model_checkpoint])
    save_model(model, problem_array[i] + '.h5')
    plot_history(history, output_path)

    compare_images(model, x, y, extent, output_path)

    # model = load_model('terrain/linear/model.00001_9.5105.h5 ( or your file)')


def attempt6():
    """
    not good for the _50 file, gets ~31% for _5 and ~27% for _20

    :return: None
    """
    from keras.models import Sequential
    from keras.layers import Input, Dense, Normalization, Lambda
    from keras.optimizers import Adam

    problem_array = ['terrain_(-100.5, -98.5, 36.5, 38.5)_20', 'terrain_(-90.0, -85.0, 39.0, 44.0)_50', 'terrain_(-76.75, -76.25, 43.25, 43.75)_5']
    for i in range(len(problem_array)):
        tiff_file = 'terrain/' + problem_array[i] + '.tiff'
        # change the model name when you change the architecture or hyperparameters.
        model_name = 'attempt6'
        output_path = f'models/{problem_array[i]}/{model_name}'

        model_checkpoint = setup_model_checkpoints(output_path)
        x, y, extent = get_xy(tiff_file)

        clear_session()

        layer = Normalization(axis=1)
        layer.adapt(x)
        model = Sequential([
            Input(2),
            layer,
            Dense(14, activation='tanh'),
            Dense(5, activation='tanh'),
            Dense(3, activation='relu'),
            Dense(1, activation='relu'),
        ])
        model_bpp = num_parameters(model) * 32 / len(y)
        model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.01), metrics=[Entropy(model_bpp=model_bpp)])
        model.summary()

        print_error(y, y.mean(), 1, 'Constant')

        history = model.fit(x, y, epochs=25, batch_size=1024, verbose=0, callbacks=[model_checkpoint])
        model.fit(x, y, epochs=25, batch_size=4096, verbose=0, callbacks=[model_checkpoint])
        model.fit(x, y, epochs=25, batch_size=16384, verbose=0, callbacks=[model_checkpoint])
        model.fit(x, y, epochs=50, batch_size=32768, verbose=0, callbacks=[model_checkpoint])
        save_model(model, problem_array[i] + '.h5')
        plot_history(history, output_path)

        compare_images(model, x, y, extent, output_path)

        # model = load_model('terrain/linear/model.00001_9.5105.h5 ( or your file)')


def attempt7():
    """
    not good for the _50 file, gets ~31% for _5 and ~27% for _20

    :return: None
    """
    from keras.models import Sequential
    from keras.layers import Input, Dense, Normalization, Lambda
    from keras.optimizers import Adam

    problem_array = ['terrain_(-100.5, -98.5, 36.5, 38.5)_20', 'terrain_(-90.0, -85.0, 39.0, 44.0)_50', 'terrain_(-76.75, -76.25, 43.25, 43.75)_5']
    for i in range(len(problem_array)):
        tiff_file = 'terrain/' + problem_array[i] + '.tiff'
        # change the model name when you change the architecture or hyperparameters.
        model_name = 'attempt7'
        output_path = f'models/{problem_array[i]}/{model_name}'

        model_checkpoint = setup_model_checkpoints(output_path)
        x, y, extent = get_xy(tiff_file)

        clear_session()

        layer = Normalization(axis=1)
        layer.adapt(x)
        model = Sequential([
            Input(2),
            layer,
            Dense(32, activation='tanh'),
            Dense(4, activation='relu'),
            Dense(4, activation='relu'),
            Dense(1, activation='linear'),
        ])
        model_bpp = num_parameters(model) * 32 / len(y)
        model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.01), metrics=[Entropy(model_bpp=model_bpp)])
        model.summary()

        print_error(y, y.mean(), 1, 'Constant')

        history = model.fit(x, y, epochs=25, batch_size=1024, verbose=0, callbacks=[model_checkpoint])
        model.fit(x, y, epochs=25, batch_size=4096, verbose=0, callbacks=[model_checkpoint])
        model.fit(x, y, epochs=25, batch_size=16384, verbose=0, callbacks=[model_checkpoint])
        model.fit(x, y, epochs=50, batch_size=32768, verbose=0, callbacks=[model_checkpoint])
        save_model(model, problem_array[i] + '.h5')
        plot_history(history, output_path)

        compare_images(model, x, y, extent, output_path)

        # model = load_model('terrain/linear/model.00001_9.5105.h5 ( or your file)')

def attempt1_50():
    """
    good for _50 file, got ~17.5%

    :return: None
    """
    from keras.models import Sequential
    from keras.layers import Input, Dense, Normalization, Lambda
    from keras.optimizers import Adam

    problem = 'terrain_(-90.0, -85.0, 39.0, 44.0)_50'
    # for i in range(len(problem_array)):
    #     tiff_file = 'terrain/' + problem_array[i] + '.tiff'
    i = 1
    tiff_file = 'terrain/' + problem + '.tiff'
    # change the model name when you change the architecture or hyperparameters.
    model_name = 'attempt1_50'
    output_path = f'models/{problem}/{model_name}'

    model_checkpoint = setup_model_checkpoints(output_path)
    x, y, extent = get_xy(tiff_file)

    clear_session()

    layer = Normalization(axis=1)
    layer.adapt(x)
    model = Sequential([
        Input(2),
        layer,
        Dense(12, activation='tanh'),
        Dense(12, activation='tanh'),
        Dense(12, activation='tanh'),
        Dense(10, activation='tanh'),
        Dense(8, activation='tanh'),
        Dense(6, activation='tanh'),
        Dense(4, activation='relu'),
        Dense(2, activation='relu'),
        Dense(1, activation='relu'),
    ])
    model_bpp = num_parameters(model) * 32 / len(y)
    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.01), metrics=[Entropy(model_bpp=model_bpp)])
    model.summary()

    print_error(y, y.mean(), 1, 'Constant')

    history = model.fit(x, y, epochs=25, batch_size=1024, verbose=0, callbacks=[model_checkpoint])
    model.fit(x, y, epochs=25, batch_size=4096, verbose=0, callbacks=[model_checkpoint])
    model.fit(x, y, epochs=40, batch_size=32768, verbose=0, callbacks=[model_checkpoint])
    model.fit(x, y, epochs=40, batch_size=32768 * 2, verbose=0, callbacks=[model_checkpoint])
    save_model(model, problem + '.h5')
    plot_history(history, output_path)

    compare_images(model, x, y, extent, output_path)

    # model = load_model('terrain/linear/model.00001_9.5105.h5 ( or your file)')

def attempt2_50():
    """
    good for _50 file, got ~17.5%

    :return: None
    """
    from keras.models import Sequential
    from keras.layers import Input, Dense, Normalization, Lambda
    from keras.optimizers import Adam

    problem = 'terrain_(-90.0, -85.0, 39.0, 44.0)_50'
    # for i in range(len(problem_array)):
    #     tiff_file = 'terrain/' + problem_array[i] + '.tiff'
    i = 1
    tiff_file = 'terrain/' + problem + '.tiff'
    # change the model name when you change the architecture or hyperparameters.
    model_name = 'attempt2_50'
    output_path = f'models/{problem}/{model_name}'

    model_checkpoint = setup_model_checkpoints(output_path)
    x, y, extent = get_xy(tiff_file)

    clear_session()

    layer = Normalization(axis=1)
    layer.adapt(x)
    model = Sequential([
        Input(2),
        layer,
        Dense(12, activation='tanh'),
        Dense(12, activation='tanh'),
        Dense(12, activation='tanh'),
        Dense(10, activation='tanh'),
        Dense(8, activation='tanh'),
        Dense(6, activation='tanh'),
        Dense(4, activation='relu'),
        Dense(2, activation='relu'),
        Dense(1, activation='relu'),
    ])
    model_bpp = num_parameters(model) * 32 / len(y)
    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.01), metrics=[Entropy(model_bpp=model_bpp)])
    model.summary()

    print_error(y, y.mean(), 1, 'Constant')

    history = model.fit(x, y, epochs=25, batch_size=256, verbose=0, callbacks=[model_checkpoint])
    model.fit(x, y, epochs=25, batch_size=1024, verbose=0, callbacks=[model_checkpoint])
    model.fit(x, y, epochs=25, batch_size=4096, verbose=0, callbacks=[model_checkpoint])
    model.fit(x, y, epochs=40, batch_size=32768, verbose=0, callbacks=[model_checkpoint])
    model.fit(x, y, epochs=40, batch_size=32768 * 2, verbose=0, callbacks=[model_checkpoint])
    save_model(model, problem + '.h5')
    plot_history(history, output_path)

    compare_images(model, x, y, extent, output_path)

    # model = load_model('terrain/linear/model.00001_9.5105.h5 ( or your file)')

def attempt1_20():
    """
    not good for the _50 file, gets ~31% for _5 and ~27% for _20

    :return: None
    """
    from keras.models import Sequential
    from keras.layers import Input, Dense, Normalization, Lambda
    from keras.optimizers import Adam

    problem = 'terrain_(-100.5, -98.5, 36.5, 38.5)_20'
    tiff_file = 'terrain/' + problem + '.tiff'
    # change the model name when you change the architecture or hyperparameters.
    model_name = 'attempt1_20'
    output_path = f'models/{problem}/{model_name}'

    model_checkpoint = setup_model_checkpoints(output_path)
    x, y, extent = get_xy(tiff_file)

    clear_session()

    layer = Normalization(axis=1)
    layer.adapt(x)
    model = Sequential([
        Input(2),
        layer,
        Dense(12, activation='tanh'),
        Dense(12, activation='tanh'),
        Dense(12, activation='tanh'),
        Dense(10, activation='tanh'),
        Dense(8, activation='tanh'),
        Dense(6, activation='tanh'),
        Dense(4, activation='relu'),
        Dense(2, activation='relu'),
        Dense(1, activation='relu'),
    ])
    model_bpp = num_parameters(model) * 32 / len(y)
    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.01), metrics=[Entropy(model_bpp=model_bpp)])
    model.summary()

    print_error(y, y.mean(), 1, 'Constant')

    history = model.fit(x, y, epochs=25, batch_size=1024, verbose=0, callbacks=[model_checkpoint])
    model.fit(x, y, epochs=25, batch_size=4096, verbose=0, callbacks=[model_checkpoint])
    model.fit(x, y, epochs=40, batch_size=32768, verbose=0, callbacks=[model_checkpoint])
    model.fit(x, y, epochs=80, batch_size=32768 * 2, verbose=0, callbacks=[model_checkpoint])
    save_model(model, problem + '.h5')
    plot_history(history, output_path)

    compare_images(model, x, y, extent, output_path)

    # model = load_model('terrain/linear/model.00001_9.5105.h5 ( or your file)')

def attempt2_20():
    """
    not good for the _50 file, gets ~31% for _5 and ~27% for _20

    :return: None
    """
    from keras.models import Sequential
    from keras.layers import Input, Dense, Normalization, Lambda
    from keras.optimizers import Adam

    problem = 'terrain_(-100.5, -98.5, 36.5, 38.5)_20'
    tiff_file = 'terrain/' + problem + '.tiff'
    # change the model name when you change the architecture or hyperparameters.
    model_name = 'attempt2_20'
    output_path = f'models/{problem}/{model_name}'

    model_checkpoint = setup_model_checkpoints(output_path)
    x, y, extent = get_xy(tiff_file)

    clear_session()

    layer = Normalization(axis=1)
    layer.adapt(x)
    model = Sequential([
        Input(2),
        layer,
        Dense(12, activation='tanh'),
        Dense(10, activation='tanh'),
        Dense(8, activation='tanh'),
        Dense(6, activation='tanh'),
        Dense(4, activation='relu'),
        Dense(2, activation='relu'),
        Dense(1, activation='relu'),
    ])
    model_bpp = num_parameters(model) * 32 / len(y)
    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.01), metrics=[Entropy(model_bpp=model_bpp)])
    model.summary()

    print_error(y, y.mean(), 1, 'Constant')

    model.fit(x, y, epochs=60, batch_size=512, verbose=0, callbacks=[model_checkpoint])
    model.fit(x, y, epochs=25, batch_size=1024, verbose=0, callbacks=[model_checkpoint])
    model.fit(x, y, epochs=25, batch_size=4096, verbose=0, callbacks=[model_checkpoint])
    model.fit(x, y, epochs=40, batch_size=32768, verbose=0, callbacks=[model_checkpoint])
    history = model.fit(x, y, epochs=80, batch_size=32768 * 2, verbose=0, callbacks=[model_checkpoint])
    save_model(model, problem + '.h5')
    plot_history(history, output_path)

    compare_images(model, x, y, extent, output_path)

    # model = load_model('terrain/linear/model.00001_9.5105.h5 ( or your file)')

def attempt3_20():
    """
    not good for the _50 file, gets ~31% for _5 and ~27% for _20

    :return: None
    """
    from keras.models import Sequential
    from keras.layers import Input, Dense, Normalization, Lambda
    from keras.optimizers import Adam

    problem = 'terrain_(-100.5, -98.5, 36.5, 38.5)_20'
    tiff_file = 'terrain/' + problem + '.tiff'
    # change the model name when you change the architecture or hyperparameters.
    model_name = 'attempt3_20'
    output_path = f'models/{problem}/{model_name}'

    model_checkpoint = setup_model_checkpoints(output_path)
    x, y, extent = get_xy(tiff_file)

    clear_session()

    layer = Normalization(axis=1)
    layer.adapt(x)
    model = Sequential([
        Input(2),
        layer,
        Dense(12, activation='tanh'),
        Dense(12, activation='tanh'),
        Dense(12, activation='tanh'),
        Dense(10, activation='tanh'),
        Dense(8, activation='tanh'),
        Dense(6, activation='tanh'),
        Dense(4, activation='relu'),
        Dense(2, activation='relu'),
        Dense(1, activation='relu'),
    ])
    model_bpp = num_parameters(model) * 32 / len(y)
    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.01), metrics=[Entropy(model_bpp=model_bpp)])
    model.summary()

    print_error(y, y.mean(), 1, 'Constant')

    model.fit(x, y, epochs=60, batch_size=256, verbose=0, callbacks=[model_checkpoint])
    model.fit(x, y, epochs=25, batch_size=1024, verbose=0, callbacks=[model_checkpoint])
    model.fit(x, y, epochs=25, batch_size=4096, verbose=0, callbacks=[model_checkpoint])
    model.fit(x, y, epochs=40, batch_size=32768, verbose=0, callbacks=[model_checkpoint])
    history = model.fit(x, y, epochs=80, batch_size=32768 * 2, verbose=0, callbacks=[model_checkpoint])
    save_model(model, problem + '.h5')
    plot_history(history, output_path)

    compare_images(model, x, y, extent, output_path)

    # model = load_model('terrain/linear/model.00001_9.5105.h5 ( or your file)')

def attempt4_20():
    """
    not good for the _50 file, gets ~31% for _5 and ~27% for _20

    :return: None
    """
    from keras.models import Sequential
    from keras.layers import Input, Dense, Normalization, Lambda
    from keras.optimizers import Adam

    problem = 'terrain_(-100.5, -98.5, 36.5, 38.5)_20'
    tiff_file = 'terrain/' + problem + '.tiff'
    # change the model name when you change the architecture or hyperparameters.
    model_name = 'attempt4_20'
    output_path = f'models/{problem}/{model_name}'

    model_checkpoint = setup_model_checkpoints(output_path)
    x, y, extent = get_xy(tiff_file)

    clear_session()

    layer = Normalization(axis=1)
    layer.adapt(x)
    model = Sequential([
        Input(2),
        layer,
        Dense(8, activation='tanh'),
        Dense(7, activation='tanh'),
        Dense(6, activation='tanh'),
        Dense(5, activation='tanh'),
        Dense(5, activation='tanh'),
        Dense(4, activation='tanh'),
        Dense(3, activation='relu'),
        Dense(2, activation='relu'),
        Dense(1, activation='relu'),
    ])
    model_bpp = num_parameters(model) * 32 / len(y)
    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.01), metrics=[Entropy(model_bpp=model_bpp)])
    model.summary()

    print_error(y, y.mean(), 1, 'Constant')

    model.fit(x, y, epochs=60, batch_size=128, verbose=0, callbacks=[model_checkpoint])
    model.fit(x, y, epochs=60, batch_size=256, verbose=0, callbacks=[model_checkpoint])
    model.fit(x, y, epochs=25, batch_size=1024, verbose=0, callbacks=[model_checkpoint])
    model.fit(x, y, epochs=25, batch_size=1024, verbose=0, callbacks=[model_checkpoint])
    model.fit(x, y, epochs=50, batch_size=4096, verbose=0, callbacks=[model_checkpoint])
    model.fit(x, y, epochs=100, batch_size=32768, verbose=0, callbacks=[model_checkpoint])
    history = model.fit(x, y, epochs=80, batch_size=32768 * 2, verbose=0, callbacks=[model_checkpoint])
    save_model(model, problem + '.h5')
    plot_history(history, output_path)

    compare_images(model, x, y, extent, output_path)

    # model = load_model('terrain/linear/model.00001_9.5105.h5 ( or your file)')

def attempt5_20():
    """
    not good for the _50 file, gets ~31% for _5 and ~27% for _20

    :return: None
    """
    from keras.models import Sequential
    from keras.layers import Input, Dense, Normalization, Lambda
    from keras.optimizers import Adam

    problem = 'terrain_(-100.5, -98.5, 36.5, 38.5)_20'
    tiff_file = 'terrain/' + problem + '.tiff'
    # change the model name when you change the architecture or hyperparameters.
    model_name = 'attempt5_20'
    output_path = f'models/{problem}/{model_name}'

    model_checkpoint = setup_model_checkpoints(output_path)
    x, y, extent = get_xy(tiff_file)

    clear_session()

    layer = Normalization(axis=1)
    layer.adapt(x)
    model = Sequential([
        Input(2),
        layer,
        Dense(8, activation='tanh'),
        Dense(7, activation='tanh'),
        Dense(6, activation='tanh'),
        Dense(5, activation='tanh'),
        Dense(5, activation='tanh'),
        Dense(4, activation='tanh'),
        Dense(3, activation='relu'),
        Dense(2, activation='relu'),
        Dense(1, activation='relu'),
    ])
    model_bpp = num_parameters(model) * 32 / len(y)
    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.01), metrics=[Entropy(model_bpp=model_bpp)])
    model.summary()

    print_error(y, y.mean(), 1, 'Constant')

    model.fit(x, y, epochs=60, batch_size=128, verbose=0, callbacks=[model_checkpoint])
    model.fit(x, y, epochs=60, batch_size=256, verbose=0, callbacks=[model_checkpoint])
    model.fit(x, y, epochs=25, batch_size=1024, verbose=0, callbacks=[model_checkpoint])
    model.fit(x, y, epochs=25, batch_size=1024, verbose=0, callbacks=[model_checkpoint])
    model.fit(x, y, epochs=50, batch_size=4096, verbose=0, callbacks=[model_checkpoint])
    model.fit(x, y, epochs=100, batch_size=32768, verbose=0, callbacks=[model_checkpoint])
    history = model.fit(x, y, epochs=80, batch_size=32768 * 2, verbose=0, callbacks=[model_checkpoint])
    save_model(model, problem + '.h5')
    plot_history(history, output_path)

    compare_images(model, x, y, extent, output_path)

    # model = load_model('terrain/linear/model.00001_9.5105.h5 ( or your file)')

def attempt6_20():
    """
    not good for the _50 file, gets ~31% for _5 and ~27% for _20

    :return: None
    """
    from keras.models import Sequential
    from keras.layers import Input, Dense, Normalization, Lambda
    from keras.optimizers import Adam

    problem = 'terrain_(-100.5, -98.5, 36.5, 38.5)_20'
    tiff_file = 'terrain/' + problem + '.tiff'
    # change the model name when you change the architecture or hyperparameters.
    model_name = 'attempt6_20'
    output_path = f'models/{problem}/{model_name}'

    model_checkpoint = setup_model_checkpoints(output_path)
    x, y, extent = get_xy(tiff_file)

    clear_session()

    layer = Normalization(axis=1)
    layer.adapt(x)
    model = Sequential([
        Input(2),
        layer,
        Dense(8, activation='tanh'),
        Dense(7, activation='tanh'),
        Dense(6, activation='tanh'),
        Dense(5, activation='tanh'),
        Dense(5, activation='tanh'),
        Dense(4, activation='tanh'),
        Dense(3, activation='relu'),
        Dense(2, activation='relu'),
        Dense(1, activation='relu'),
    ])
    model_bpp = num_parameters(model) * 32 / len(y)
    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.01), metrics=[Entropy(model_bpp=model_bpp)])
    model.summary()

    print_error(y, y.mean(), 1, 'Constant')

    model.fit(x, y, epochs=60, batch_size=128, verbose=0, callbacks=[model_checkpoint])
    model.fit(x, y, epochs=60, batch_size=256, verbose=0, callbacks=[model_checkpoint])
    model.fit(x, y, epochs=25, batch_size=1024, verbose=0, callbacks=[model_checkpoint])
    model.fit(x, y, epochs=25, batch_size=1024, verbose=0, callbacks=[model_checkpoint])
    model.fit(x, y, epochs=50, batch_size=4096, verbose=0, callbacks=[model_checkpoint])
    model.fit(x, y, epochs=100, batch_size=32768, verbose=0, callbacks=[model_checkpoint])
    model.fit(x, y, epochs=80, batch_size=32768 * 2, verbose=0, callbacks=[model_checkpoint])

    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.001), metrics=[Entropy(model_bpp=model_bpp)])
    model.fit(x, y, epochs=60, batch_size=128, verbose=0, callbacks=[model_checkpoint])
    model.fit(x, y, epochs=60, batch_size=256, verbose=0, callbacks=[model_checkpoint])
    model.fit(x, y, epochs=25, batch_size=1024, verbose=0, callbacks=[model_checkpoint])
    model.fit(x, y, epochs=25, batch_size=1024, verbose=0, callbacks=[model_checkpoint])
    model.fit(x, y, epochs=50, batch_size=4096, verbose=0, callbacks=[model_checkpoint])
    model.fit(x, y, epochs=100, batch_size=32768, verbose=0, callbacks=[model_checkpoint])
    history = model.fit(x, y, epochs=80, batch_size=32768 * 2, verbose=0, callbacks=[model_checkpoint])

    save_model(model, problem + '.h5')
    plot_history(history, output_path)

    compare_images(model, x, y, extent, output_path)

    # model = load_model('terrain/linear/model.00001_9.5105.h5 ( or your file)')

def attempt7_20():
    """
    not good for the _50 file, gets ~31% for _5 and ~27% for _20

    :return: None
    """
    from keras.models import Sequential
    from keras.layers import Input, Dense, Normalization, Lambda
    from keras.optimizers import Adam

    problem = 'terrain_(-100.5, -98.5, 36.5, 38.5)_20'
    tiff_file = 'terrain/' + problem + '.tiff'
    # change the model name when you change the architecture or hyperparameters.
    model_name = 'attempt7_20'
    output_path = f'models/{problem}/{model_name}'

    model_checkpoint = setup_model_checkpoints(output_path)
    x, y, extent = get_xy(tiff_file)

    clear_session()

    layer = Normalization(axis=1)
    layer.adapt(x)
    model = Sequential([
        Input(2),
        layer,
        Dense(16, activation='tanh'),
        Dense(8, activation='tanh'),
        Dense(6, activation='relu'),
        Dense(2, activation='relu'),
        Dense(1, activation='relu'),
    ])
    model_bpp = num_parameters(model) * 32 / len(y)
    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.1), metrics=[Entropy(model_bpp=model_bpp)])
    model.summary()

    print_error(y, y.mean(), 1, 'Constant')

    model.fit(x, y, epochs=60, batch_size=128, verbose=0, callbacks=[model_checkpoint])
    model.fit(x, y, epochs=60, batch_size=256, verbose=0, callbacks=[model_checkpoint])
    model.fit(x, y, epochs=25, batch_size=1024, verbose=0, callbacks=[model_checkpoint])
    model.fit(x, y, epochs=25, batch_size=1024, verbose=0, callbacks=[model_checkpoint])
    model.fit(x, y, epochs=50, batch_size=4096, verbose=0, callbacks=[model_checkpoint])
    model.fit(x, y, epochs=100, batch_size=32768, verbose=0, callbacks=[model_checkpoint])
    model.fit(x, y, epochs=80, batch_size=32768 * 2, verbose=0, callbacks=[model_checkpoint])

    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.01), metrics=[Entropy(model_bpp=model_bpp)])
    model.fit(x, y, epochs=60, batch_size=128, verbose=0, callbacks=[model_checkpoint])
    model.fit(x, y, epochs=60, batch_size=256, verbose=0, callbacks=[model_checkpoint])
    model.fit(x, y, epochs=25, batch_size=1024, verbose=0, callbacks=[model_checkpoint])
    model.fit(x, y, epochs=25, batch_size=1024, verbose=0, callbacks=[model_checkpoint])
    model.fit(x, y, epochs=50, batch_size=4096, verbose=0, callbacks=[model_checkpoint])
    model.fit(x, y, epochs=100, batch_size=32768, verbose=0, callbacks=[model_checkpoint])
    history = model.fit(x, y, epochs=80, batch_size=32768 * 2, verbose=0, callbacks=[model_checkpoint])

    save_model(model, problem + '.h5')
    plot_history(history, output_path)

    compare_images(model, x, y, extent, output_path)

    # model = load_model('terrain/linear/model.00001_9.5105.h5 ( or your file)')

def attempt1_5():
    """
    not good for the _50 file, gets ~31% for _5 and ~27% for _20

    :return: None
    """
    from keras.models import Sequential
    from keras.layers import Input, Dense, Normalization, Lambda
    from keras.optimizers import Adam

    problem = 'terrain_(-76.75, -76.25, 43.25, 43.75)_5'
    tiff_file = 'terrain/' + problem + '.tiff'
    # change the model name when you change the architecture or hyperparameters.
    model_name = 'attempt1_5'
    output_path = f'models/{problem}/{model_name}'

    model_checkpoint = setup_model_checkpoints(output_path)
    x, y, extent = get_xy(tiff_file)

    clear_session()

    layer = Normalization(axis=1)
    layer.adapt(x)
    model = Sequential([
        Input(2),
        layer,
        Dense(12, activation='tanh'),
        Dense(12, activation='tanh'),
        Dense(12, activation='tanh'),
        Dense(10, activation='tanh'),
        Dense(8, activation='tanh'),
        Dense(6, activation='tanh'),
        Dense(4, activation='relu'),
        Dense(2, activation='relu'),
        Dense(1, activation='relu'),
    ])
    model_bpp = num_parameters(model) * 32 / len(y)
    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.01), metrics=[Entropy(model_bpp=model_bpp)])
    model.summary()

    print_error(y, y.mean(), 1, 'Constant')

    history = model.fit(x, y, epochs=25, batch_size=1024, verbose=0, callbacks=[model_checkpoint])
    model.fit(x, y, epochs=25, batch_size=4096, verbose=0, callbacks=[model_checkpoint])
    model.fit(x, y, epochs=40, batch_size=32768, verbose=0, callbacks=[model_checkpoint])
    model.fit(x, y, epochs=80, batch_size=32768 * 2, verbose=0, callbacks=[model_checkpoint])
    save_model(model, problem + '.h5')
    plot_history(history, output_path)

    compare_images(model, x, y, extent, output_path)

    # model = load_model('terrain/linear/model.00001_9.5105.h5 ( or your file)')

def attempt2_5():
    """
    not good for the _50 file, gets ~31% for _5 and ~27% for _20

    :return: None
    """
    from keras.models import Sequential
    from keras.layers import Input, Dense, Normalization, Lambda
    from keras.optimizers import Adam

    problem = 'terrain_(-76.75, -76.25, 43.25, 43.75)_5'
    tiff_file = 'terrain/' + problem + '.tiff'
    # change the model name when you change the architecture or hyperparameters.
    model_name = 'attempt2_5'
    output_path = f'models/{problem}/{model_name}'

    model_checkpoint = setup_model_checkpoints(output_path)
    x, y, extent = get_xy(tiff_file)

    clear_session()

    layer = Normalization(axis=1)
    layer.adapt(x)
    model = Sequential([
        Input(2),
        layer,
        Dense(16, activation='swish'),
        Dense(16, activation='swish'),
        Dense(12, activation='swish'),
        Dense(12, activation='tanh'),
        Dense(12, activation='tanh'),
        Dense(10, activation='tanh'),
        Dense(8, activation='tanh'),
        Dense(6, activation='tanh'),
        Dense(4, activation='tanh'),
        Dense(2, activation='swish'),
        Dense(1, activation='relu'),
    ])
    model_bpp = num_parameters(model) * 32 / len(y)
    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.01), metrics=[Entropy(model_bpp=model_bpp)])
    model.summary()

    print_error(y, y.mean(), 1, 'Constant')

    history = model.fit(x, y, epochs=50, batch_size=128, verbose=0, callbacks=[model_checkpoint])
    history = model.fit(x, y, epochs=50, batch_size=256, verbose=0, callbacks=[model_checkpoint])
    model.fit(x, y, epochs=25, batch_size=1024, verbose=0, callbacks=[model_checkpoint])
    model.fit(x, y, epochs=25, batch_size=4096, verbose=0, callbacks=[model_checkpoint])
    model.fit(x, y, epochs=40, batch_size=32768, verbose=0, callbacks=[model_checkpoint])
    model.fit(x, y, epochs=80, batch_size=32768 * 2, verbose=0, callbacks=[model_checkpoint])
    save_model(model, problem + '.h5')
    plot_history(history, output_path)

    compare_images(model, x, y, extent, output_path)

    # model = load_model('terrain/linear/model.00001_9.5105.h5 ( or your file)')


if __name__ == '__main__':
    # attempt2_5()
    # attempt2_50()
    attempt6_20()
