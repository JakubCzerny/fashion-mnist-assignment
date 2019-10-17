from utils import mnist_reader

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import Model, regularizers, backend
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.layers import InputLayer, Lambda, LeakyReLU, ReLU, GlobalAveragePooling2D, BatchNormalization, Flatten, Dense, MaxPool2D, AveragePooling2D, Conv2D, Dropout, MaxPooling2D
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator


train_generator_args = dict(
  data_format = 'channels_last',
  rescale=1./255,
  width_shift_range=3,
  height_shift_range=3,
  horizontal_flip=True,
  cval=0,
  fill_mode='constant',
)

val_test_generator_args = dict(
  data_format = 'channels_last',
  rescale=1./255,
)


def train_random_splits_fn(params):
    """
    Function perform random split of data and model training - called in a loop

    @param params: dictionary 'build_fn', 'batch_size', 'augment', 'patience', 'model', 'iter', 'epochs'
    @return: trained model
    """

    print ('Passed parameters: ', params)

    model = params['build_fn'](params)
    batch_size = params['batch_size']

    # Load & split data
    X, y = mnist_reader.load_mnist(params['data_path'], kind='train')
    X_test, y_test = mnist_reader.load_mnist(params['data_path'], kind='t10k')
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=int(X_test.shape[0]))

    # Add channel dimension to data
    X_train = X_train.reshape(X_train.shape[0], params['image_size'], params['image_size'], 1)
    X_test = X_test.reshape(X_test.shape[0], params['image_size'], params['image_size'], 1)
    X_val = X_val.reshape(X_val.shape[0], params['image_size'], params['image_size'], 1)

    # One hot encode the data
    y_train_encoded = to_categorical(y_train, num_classes=params['num_classes'], dtype='float32')
    y_test_encoded = to_categorical(y_test, num_classes=params['num_classes'], dtype='float32')
    y_val_encoded = to_categorical(y_val, num_classes=params['num_classes'], dtype='float32')

    print("Train {:}, Validate {:}, Test {:}".format(X_train.shape[0], X_val.shape[0], X_test.shape[0]))

    if params['augment']:
        train_datagen = ImageDataGenerator(**train_generator_args)
    else:
        train_datagen = ImageDataGenerator(**val_test_generator_args)

    test_datagen = ImageDataGenerator(**val_test_generator_args)
    val_datagen = ImageDataGenerator(**val_test_generator_args)

    train_datagen.fit(X_train)
    test_datagen.fit(X_test)
    val_datagen.fit(X_val)

    train_generator = train_datagen.flow(
      X_train,
      y_train_encoded,
      batch_size=batch_size,
      shuffle=True
    )

    test_generator = test_datagen.flow(
      X_test,
      y_test_encoded,
      batch_size=batch_size,
      shuffle=False
    )

    val_generator = val_datagen.flow(
      X_val,
      y_val_encoded,
      batch_size=batch_size,
      shuffle=False
    )

    earlystop_callback = EarlyStopping(monitor='val_acc',
                                     mode='max',
                                     patience=params['patience']
                                    )

    checkpoint = ModelCheckpoint('models/model_{:}_{:}.h5'.format(params['model'], params['iter']),
                               monitor='val_loss',
                               verbose=1,
                               save_best_only=True,
                               mode='min',
                               save_freq='epoch',
                              )

    step_size_train=train_generator.n//train_generator.batch_size

    model.fit_generator(generator=train_generator,
                      validation_data=val_generator,
                      steps_per_epoch=step_size_train,
                      epochs=params['epochs'],
                      verbose=1,
                      callbacks=[earlystop_callback, checkpoint]
                      )

    results = model.evaluate_generator(test_generator)

    print("Test accuracy: {:.3f} and loss: {:.3f}".format(results[1], results[0]))

    return model


def hyper_tuning_fn(params):
    """
    Function used in hyperparameter optimization with hyperopt library

    @param params: dictionary {'build_fn', 'batch_size', 'augment', 'patience', 'epochs'}
    @return: dictionary {'loss', 'acc', 'status'}
    """

    print ('Passed parameters: ', params)

    model = params['build_fn'](params)
    batch_size = params['batch_size']

    # Load & split data
    X, y = mnist_reader.load_mnist(params['data_path'], kind='train')
    X_test, y_test = mnist_reader.load_mnist(params['data_path'], kind='t10k')
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=int(X_test.shape[0]))

    # Add channel dimension to data
    X_train = X_train.reshape(X_train.shape[0], params['image_size'], params['image_size'], 1)
    X_test = X_test.reshape(X_test.shape[0], params['image_size'], params['image_size'], 1)
    X_val = X_val.reshape(X_val.shape[0], params['image_size'], params['image_size'], 1)

    # One hot encode the data
    y_train_encoded = to_categorical(y_train, num_classes=params['num_classes'], dtype='float32')
    y_test_encoded = to_categorical(y_test, num_classes=params['num_classes'], dtype='float32')
    y_val_encoded = to_categorical(y_val, num_classes=params['num_classes'], dtype='float32')

    print("Train {:}, Validate {:}, Test {:}".format(X_train.shape[0], X_val.shape[0], X_test.shape[0]))

    if params['augment']:
        train_datagen = ImageDataGenerator(**train_generator_args)
    else:
        train_datagen = ImageDataGenerator(**val_test_generator_args)

    test_datagen = ImageDataGenerator(**val_test_generator_args)
    val_datagen = ImageDataGenerator(**val_test_generator_args)

    train_datagen.fit(X_train)
    test_datagen.fit(X_test)
    val_datagen.fit(X_val)

    train_generator = train_datagen.flow(
      X_train,
      y_train_encoded,
      batch_size=batch_size,
      shuffle=True
    )

    test_generator = test_datagen.flow(
      X_test,
      y_test_encoded,
      batch_size=batch_size,
      shuffle=False
    )

    val_generator = val_datagen.flow(
      X_val,
      y_val_encoded,
      batch_size=batch_size,
      shuffle=False
    )

    earlystop_callback = EarlyStopping(monitor='val_acc',
                                     mode='max',
                                     patience=params['patience']
                                    )

    step_size_train=train_generator.n//train_generator.batch_size

    model.fit_generator(generator=train_generator,
                      validation_data=val_generator,
                      steps_per_epoch=step_size_train,
                      epochs=params['epochs'],
                      verbose=2,
                      callbacks=[earlystop_callback]
                      )

    results = model.evaluate_generator(test_generator)

    print("Test accuracy: {:.3f} and loss: {:.3f}".format(results[1], results[0]))

    return {'loss':results[0], 'acc':results[1], 'status': STATUS_OK}


def final_training_fn(params):
    """
    Function does model training without validation subset - instead 60k datapoints are used for training

    @param params: dictionary {'build_fn', 'batch_size', 'augment', 'patience', 'epochs'}
    @return: list [history, model]
    """

    print ('Passed parameters: ', params)

    model = params['build_fn'](params)
    batch_size = params['batch_size']

    # Load & split data
    X_train, y_train = mnist_reader.load_mnist(params['data_path'], kind='train')
    X_test, y_test = mnist_reader.load_mnist(params['data_path'], kind='t10k')

    # Add channel dimension to data
    X_train = X_train.reshape(X_train.shape[0], params['image_size'], params['image_size'], 1)
    X_test = X_test.reshape(X_test.shape[0], params['image_size'], params['image_size'], 1)

    # One hot encode the data
    y_train_encoded = to_categorical(y_train, num_classes=params['num_classes'], dtype='float32')
    y_test_encoded = to_categorical(y_test, num_classes=params['num_classes'], dtype='float32')

    print("Train {:}, Test {:}".format(X_train.shape[0], X_test.shape[0]))

    if params['augment']:
        train_datagen = ImageDataGenerator(**train_generator_args)
    else:
        train_datagen = ImageDataGenerator(**val_test_generator_args)

    test_datagen = ImageDataGenerator(**val_test_generator_args)

    train_datagen.fit(X_train)
    test_datagen.fit(X_test)

    train_generator = train_datagen.flow(
      X_train,
      y_train_encoded,
      batch_size=batch_size,
      shuffle=True
    )

    test_generator = test_datagen.flow(
      X_test,
      y_test_encoded,
      batch_size=batch_size,
      shuffle=False
    )

    step_size_train=train_generator.n//train_generator.batch_size

    history = model.fit_generator(generator=train_generator,
                      validation_data=test_generator,
                      steps_per_epoch=step_size_train,
                      use_multiprocessing=True,
                      epochs=params['epochs'],
                      verbose=1
                      )

    results = model.evaluate_generator(test_generator)

    print("Test accuracy: {:.3f} and loss: {:.3f}".format(results[1], results[0]))

    return history, model
