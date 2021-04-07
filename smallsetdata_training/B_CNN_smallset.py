import keras
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D
from keras.layers import Input
from scipy import io

batch_size = 40                           # number of samples per gradient update (for mini-batch gradient-based algorithms)
epochs = 3                               # number of times you go through your whole training set
loss = keras.losses.mean_squared_error    # loss function
optimizers = keras.optimizers.Adadelta()  # optimization algorithm,
                                          # an Adaptive Learning Rate Method is used here, so no need to set training rate ;-)
metrics=['mean_squared_error']            # metrics:    a function that is used to judge the performance of your model,
                                          # mean squared error is used here

######## load train and development data for training ########
mat_contents = io.loadmat('../Data/X_train_smallset.mat')
X_train = mat_contents['X_train']
mat_contents = io.loadmat('../Data/Y_train_smallset.mat')
Y_train = mat_contents['Y_train']
mat_contents = io.loadmat('../Data/X_dev_smallset.mat')
X_dev = mat_contents['X_dev']
mat_contents = io.loadmat('../Data/Y_dev_smallset.mat')
Y_dev = mat_contents['Y_dev']

train_datasize, patch_rows, patch_cols = X_train.shape[0], X_train.shape[1], X_train.shape[2]
dev_datasize = X_dev.shape[0]
X_train = X_train.reshape(train_datasize, patch_rows, patch_cols, 1)
X_dev = X_dev.reshape(dev_datasize, patch_rows, patch_cols, 1)
Y_train = Y_train.reshape(train_datasize, patch_rows, patch_cols, 1)
Y_dev = Y_dev.reshape(dev_datasize, patch_rows, patch_cols, 1)
print('X_train shape:', X_train.shape)
print('Y_train shape:', Y_train.shape)

######## specify the CNN architecture ########
# step1: Feature extraction: Extracts a set of features directly from the missing data (2 * 32 * (3, 3) convolution)
# step2: Non-linear mapping: Maps the features representing missing data to complete data with several fully connected layers (3 * 20 * (3, 3) convolution)
# step3: Reconstruction: reconstruct the complete data from its features (2 * 32 * (3, 3) deconvolution + 1 * (1, 1) deconvolution)
model = Sequential()
input_data = Input(shape=(patch_rows, patch_cols, 1))

####### feature extraction
conv1 = Conv2D(32,
               kernel_size=(3, 3),
               padding='same',
               activation='relu',
               kernel_initializer='glorot_uniform')(input_data)
# conv1 = BatchNormalization()(conv1)
conv2 = Conv2D(32,
               kernel_size=(3, 3),
               padding='same',
               activation='relu',
               kernel_initializer='glorot_uniform')(conv1)
# conv2 = BatchNormalization()(conv2)
# conv2 = Dropout(0.25)(conv2)

####### non-linear mapping
# conv3 = Dense(180, activation='relu')(conv2)
# conv3 = Dropout(0.25)(conv3)
# conv3 = Dense(180, activation='relu')(conv3)
# conv3 = Dropout(0.25)(conv3)
# conv3 = Dense(180, activation='relu')(conv3)
# conv3 = Dropout(0.25)(conv3)
# conv3 = Dense(180, activation='relu')(conv3)

conv3 = Conv2D(20,
               kernel_size=(3, 3),
               padding='same',
               activation='relu',
               kernel_initializer='glorot_uniform')(conv2)
conv3 = BatchNormalization()(conv3)
conv3 = Conv2D(20,
               kernel_size=(3, 3),
               padding='same',
               activation='relu',
               kernel_initializer='glorot_uniform')(conv3)
conv3 = BatchNormalization()(conv3)
conv3 = Conv2D(20,
               kernel_size=(3, 3),
               padding='same',
               activation='relu',
               kernel_initializer='glorot_uniform')(conv3)
conv3 = BatchNormalization()(conv3)
conv3 = Conv2D(20,
               kernel_size=(3, 3),
               padding='same',
               activation='relu',
               kernel_initializer='glorot_uniform')(conv3)
conv3 = BatchNormalization()(conv3)

####### reconstruction
deconv1 = Conv2DTranspose(32,
               kernel_size=(3, 3),
               padding='same',
               activation='relu',
               kernel_initializer='glorot_uniform')(conv3)
# deconv1 = BatchNormalization()(deconv1)
deconv2 = Conv2DTranspose(32,
               kernel_size=(3, 3),
               padding='same',
               activation='relu',
               kernel_initializer='glorot_uniform')(deconv1)
# deconv2 = BatchNormalization()(deconv2)
# deconv2 = Dropout(0.25)(deconv2)
output_data = Conv2DTranspose(1,
               kernel_size=(1, 1),
               padding='same',
               activation='tanh',
               kernel_initializer='glorot_uniform')(deconv2)
model = Model(inputs=input_data, outputs=output_data)
print(model.summary())

######## specify the inversion algorithm ########
model.compile(loss=loss,
              optimizer=optimizers)

######## training ########
model.fit(X_train, Y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_dev, Y_dev))
model.save('../Data/trained_model_smallset.h5')

