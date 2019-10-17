import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import Model, regularizers, backend
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.layers import InputLayer, Lambda, LeakyReLU, ReLU, GlobalAveragePooling2D, BatchNormalization, Flatten, Dense, MaxPool2D, AveragePooling2D, Conv2D, Dropout, MaxPooling2D
from tensorflow.keras.optimizers import Adam, SGD

def build_model_v1(params):
    """
    Network remind simplified VGG

    @param params: dictionary 'conv', 'dense', 'lr', 'optimizer', 'num_classes'
    @return: compiled model - ready to train
    """

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2D(filters=params['conv'], kernel_size=(3,3), padding='same', activation='relu', input_shape=(params['image_size'],params['image_size'],1)))
    model.add(tf.keras.layers.Conv2D(filters=params['conv'], kernel_size=(3,3), padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

    model.add(tf.keras.layers.Conv2D(filters=2*params['conv'], kernel_size=(3,3), padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv2D(filters=2*params['conv'], kernel_size=(3,3), padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

    model.add(tf.keras.layers.Conv2D(filters=3*params['conv'], kernel_size=(3,3), padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv2D(filters=3*params['conv'], kernel_size=(3,3), padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(params['dense'], activation='relu'))
    model.add(tf.keras.layers.Dense(params['num_classes'], activation='softmax'))

    optimizer = params['optimizer'](learning_rate=params['lr'])

    model.compile(optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    print(model.summary())

    return model


def build_model_v2(params):
    """
    Slightly modified version of model v1

    - 1 more dense layer
    - dropout before softmax
    - more filter in 2 last convolutional layers

    @param params: dictionary 'conv', 'dense', 'lr', 'optimizer', 'num_classes'
    @return: compiled model - ready to train
    """

    model = tf.keras.Sequential()

    model.add(Conv2D(filters=params['conv'], kernel_size=(3,3), padding='same', activation='relu', input_shape=(params['image_size'],params['image_size'],1)))
    model.add(Conv2D(filters=params['conv'], kernel_size=(3,3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(filters=2*params['conv'], kernel_size=(3,3), padding='same', activation='relu'))
    model.add(Conv2D(filters=2*params['conv'], kernel_size=(3,3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(filters=4*params['conv'], kernel_size=(3,3), padding='same', activation='relu'))
    model.add(Conv2D(filters=4*params['conv'], kernel_size=(3,3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(params['dense'], activation='relu'))
    model.add(tf.keras.layers.Dense(params['dense'], activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(params['num_classes'], activation='softmax'))

    optimizer = params['optimizer'](learning_rate=params['lr'])

    model.compile(optimizer=optimizer,
                loss=params['loss'],
                metrics=['accuracy'])
    print(model.summary())

    return model


def build_model_v3(params):
    """
    Slightly modified version of model v2

    - ReLU replaced with LeakyReLU
    - last MaxPooling 2x2 --> 3x3
    - added one more dropout

    @param params: dictionary 'conv', 'dense', 'lr', 'optimizer', 'num_classes'
    @return: compiled model - ready to train
    """
    model = tf.keras.Sequential()

    model.add(Conv2D(filters=params['conv'], kernel_size=(3,3), padding='same', input_shape=(params['image_size'],params['image_size'],1)))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Conv2D(filters=params['conv'], kernel_size=(3,3), padding='same'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(filters=2*params['conv'], kernel_size=(3,3), padding='same'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Conv2D(filters=2*params['conv'], kernel_size=(3,3), padding='same'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(filters=4*params['conv'], kernel_size=(3,3), padding='same'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Conv2D(filters=4*params['conv'], kernel_size=(3,3), padding='valid'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(MaxPooling2D(pool_size=(3,3)))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(params['dense']))
    model.add(LeakyReLU(alpha=0.01))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(params['dense']))
    model.add(LeakyReLU(alpha=0.01))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(params['num_classes'], activation='softmax'))

    optimizer = params['optimizer'](learning_rate=params['lr'])

    model.compile(optimizer=optimizer,
                loss=params['loss'],
                metrics=['accuracy'])
    print(model.summary())

    return model


def build_model_v4(params):
    """
    Slightly modified version of model v3

    - back to MaxPooling 2x2
    - even more dropout

    @param params: dictionary 'conv', 'dense', 'lr', 'optimizer', 'num_classes'
    @return: compiled model - ready to train
    """
    model = tf.keras.Sequential()

    model.add(Conv2D(filters=params['conv'], kernel_size=(3,3), padding='same', input_shape=(params['image_size'],params['image_size'],1)))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Conv2D(filters=params['conv'], kernel_size=(3,3), padding='same'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(filters=2*params['conv'], kernel_size=(3,3), padding='same'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Conv2D(filters=2*params['conv'], kernel_size=(3,3), padding='same'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(filters=4*params['conv'], kernel_size=(3,3), padding='same'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Conv2D(filters=4*params['conv'], kernel_size=(3,3), padding='same'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Flatten())
    model.add(BatchNormalization())
    model.add(tf.keras.layers.Dense(params['dense']))
    model.add(LeakyReLU(alpha=0.01))
    model.add(tf.keras.layers.Dropout(0.5))

    model.add(tf.keras.layers.Dense(params['dense']))
    model.add(LeakyReLU(alpha=0.01))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(params['num_classes'], activation='softmax'))

    optimizer = params['optimizer'](learning_rate=params['lr'])

    model.compile(optimizer=optimizer,
                loss=params['loss'],
                metrics=['accuracy'])
    print(model.summary())

    return model


def build_model_v5(params):
    """
    Custom model - obtained after multiple iterations of training - more capacity - more regularization - cycles

    @param params: dictionary 'conv', 'dense', 'lr', 'optimizer', 'num_classes'
    @return: compiled model - ready to train
    """
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2D(filters=params['conv'], kernel_size=(3,3), padding='same', activation='relu', input_shape=(params['image_size'],params['image_size'],1)))
    model.add(BatchNormalization())
    model.add(tf.keras.layers.Conv2D(filters=2*params['conv'], kernel_size=(3,3), padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(BatchNormalization())
    model.add(tf.keras.layers.Conv2D(filters=params['conv'], kernel_size=(3,3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(tf.keras.layers.Conv2D(filters=2*params['conv'], kernel_size=(3,3), padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(BatchNormalization())
    model.add(tf.keras.layers.Conv2D(filters=params['conv'], kernel_size=(3,3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(tf.keras.layers.Conv2D(filters=2*params['conv'], kernel_size=(3,3), padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Flatten())
    model.add(BatchNormalization())
    model.add(tf.keras.layers.Dense(params['dense'], activation='relu', kernel_regularizer=regularizers.l2(params['reg_l2'])))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(params['dense'], activation='relu', kernel_regularizer=regularizers.l2(params['reg_l2'])))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(params['num_classes'], activation='softmax', kernel_regularizer=regularizers.l2(params['reg_l2'])))

    optimizer = params['optimizer'](learning_rate=params['lr'])

    model.compile(optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    print(model.summary())

    return model


class ResBlockBottleneck(tf.keras.Model):
    '''
    Basic module of bottlenect residual Network
    Composed of 3 convolution layers:

    1) 1x1 conv - keep spatial size but change number of filters, commonly smaller than input
    2) 3x3 conv - classical convolution operation, commonly the same number of filters as above
    3) 1x1 conv - remap the data into original size - number of channels - allows easy residual connection
    '''
    def __init__(self, filters, padding='same'):
        super(ResBlockBottleneck, self).__init__(name='')

        self.conv1 = Conv2D(filters=filters[0], kernel_size=(1,1), padding='same')
        self.bn1 = BatchNormalization()

        self.conv2 = Conv2D(filters=filters[1], kernel_size=(3,3), padding='same')
        self.bn2 = BatchNormalization()

        self.conv3 = Conv2D(filters=filters[2], kernel_size=(1,1), padding=padding)
        self.bn3 = BatchNormalization()


    def call(self, input_tensor, training=False):
        x = self.conv1(input_tensor)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv3(x)
        x = self.bn3(x, training=training)

        x += input_tensor
        x = tf.nn.relu(x)

        return x


def build_model_v6(params):
    """
    Model strongly inspired by residual networks with bottlenect architecture.
    Idea behind it is forcing the network to learn more compact - thus better - representation

    @param params: dictionary 'conv', 'dense', 'lr', 'optimizer', 'reg_l2', 'num_classes'
    @return: compiled model - ready to train
    """
    model = tf.keras.Sequential()
    model.add(Conv2D(filters=params['conv'][2], kernel_size=(3,3), padding='same', activation='relu', input_shape=(params['image_size'],params['image_size'],1)))

    model.add(ResBlockBottleneck(params['conv']))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=2*params['conv'][2], kernel_size=(1,1), padding='same', activation='relu'))
    model.add(ResBlockBottleneck(2*np.array(params['conv'])))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=3*params['conv'][2], kernel_size=(1,1), padding='same', activation='relu'))
    model.add(ResBlockBottleneck(3*np.array(params['conv']),'valid'))
    model.add(MaxPooling2D(pool_size=(3,3)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(params['dense'], activation='relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Dense(params['dense'], activation='relu'))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(params['num_classes'], activation='softmax', kernel_regularizer=regularizers.l2(params['reg_l2'])))

    optimizer = params['optimizer'](learning_rate=params['lr'])

    model.compile(optimizer=optimizer,
                loss=params['loss'],
                metrics=['accuracy'])

    print(model.summary())

    return model


class ResBlock(tf.keras.Model):
    '''
    Basic module of simple residual Network
    Composed of 2 convolution layers:

    Both layers have the same numer of filter, so that the input to the first one is compatible with the output of the second one, allowing easy residual connection
    '''
    def __init__(self, filters):
        super(ResBlock, self).__init__(name='')

        self.conv1 = Conv2D(filters=filters, kernel_size=(3,3), padding='same')
        self.bn1 = BatchNormalization()

        self.conv2 = Conv2D(filters=filters, kernel_size=(3,3), padding='same')
        self.bn2 = BatchNormalization()


    def call(self, input_tensor, training=False):
        x = self.conv1(input_tensor)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)

        x += input_tensor
        x = tf.nn.relu(x)

        return x

def build_model_v7(params):
    """
    It's basically VGG architecture with residual connections

    @param params: dictionary 'conv', 'dense', 'lr', 'optimizer', 'reg_l2', 'num_classes'
    @return: compiled model - ready to train
    """
    model = tf.keras.Sequential()

    model.add(Conv2D(filters=params['conv'], kernel_size=(3,3), padding='same', activation='relu', input_shape=(params['image_size'],params['image_size'],1)))
    model.add(ResBlock(params['conv']))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(BatchNormalization())
    model.add(Conv2D(filters=2*params['conv'], kernel_size=(3,3), padding='same'))
    model.add(ReLU())
    model.add(ResBlock(2*params['conv']))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(BatchNormalization())
    model.add(Conv2D(filters=4*params['conv'], kernel_size=(3,3), padding='same'))
    model.add(ReLU())
    model.add(ResBlock(4*params['conv']))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(params['dense']))
    model.add(LeakyReLU(0.01))
    model.add(Dropout(0.5))
    model.add(tf.keras.layers.Dense(params['dense']//2))
    model.add(LeakyReLU(0.01))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(tf.keras.layers.Dense(params['num_classes'], activation='softmax', kernel_regularizer=regularizers.l2(params['reg_l2'])))

    optimizer = params['optimizer'](learning_rate=params['lr'])

    model.compile(optimizer=optimizer,
                loss=params['loss'],
                metrics=['accuracy'])

    print(model.summary())

    return model
