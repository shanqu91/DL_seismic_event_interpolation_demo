import keras
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend
import numpy as np
import matplotlib.pyplot as plt
from scipy import io

def plot_feature_maps(feature_maps, num):
    height, width, depth = feature_maps.shape
    plt.figure(num=num)
    for i in range(depth):
        plt.subplot(2, np.ceil(depth/2), i+1)
        plt.axis('off')
        plt.imshow(feature_maps[:,:,i], cmap='gray')
    plt.subplots_adjust(left=0.01, bottom=0.01, right=0.9, top=0.9, wspace=0.01, hspace=0.01)


mat_contents = io.loadmat('../Data/X_test.mat')
X_test = mat_contents['X_test']

batch_size = 40

test_datasize = 1
patch_rows, patch_cols = X_test.shape[0], X_test.shape[1]
X_test = X_test.reshape(test_datasize, patch_rows, patch_cols, 1)

# load trained model
model = load_model('../Data/trained_model_full.h5')

# prediction
Y_test = model.predict(X_test, batch_size=batch_size, verbose=1)
io.savemat('../Data/predicted_test_data_full.mat', {'Y_test':Y_test})

######## visualize CNN layers ########
layer_names = [layer.name for layer in model.layers]
print(layer_names)

plt.figure(1)
plt.subplot(121)
plt.imshow(X_test[0,:,:,0], cmap='gray')
plt.subplot(122)
plt.imshow(Y_test[0,:,:,0], cmap='gray')
plt.title('with 8000 full data including steep events, test data (left->input, right->predicted)')

interlayer = Model(inputs=model.input, outputs=model.get_layer(layer_names[1]).output)
features = interlayer.predict(X_test)
plot_feature_maps(features[0], 3)
plt.title('intermediate layer after feature extraction')

interlayer = Model(inputs=model.input, outputs=model.get_layer(layer_names[-3]).output)
features = interlayer.predict(X_test)
plot_feature_maps(features[0], 4)
plt.title('intermediate layer after non-linear mapping')

plt.show()
