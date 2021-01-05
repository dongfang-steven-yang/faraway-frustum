from tensorflow import keras
import pickle
import numpy as np
import tensorflow as tf

from models.models import BEV2dCentroid

import matplotlib.pyplot as plt

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

tf.compat.v1.disable_eager_execution()

# see the coordinates convention: http://www.cvlibs.net/datasets/kitti/setup.php
# all the coordinates follow the cam 0's coordinates in the above link

# data format:
# - dim 0: num. of frustums
# - dim 1: num. of points
# - dim 2: x, y, z, intensity, useless

# labels format:
# - dim 0: num. of frustums
# - dim 1: x, y, z, height, width, length, rotation_y

data = pickle.load(open('data/samples.p', 'rb'))
labels = pickle.load(open('data/labels.p', 'rb'))
assert len(data) == len(labels)

# create 2D grid. X and Y indices are the same with x and z from the data. The value in each cell in the grid is
# equal to the number of 3D points that falls into that cell.

# initiliaze Bird's Eye View (BEV) grid. WidthxHeight, resolution is defined explicitly below.
# Take only x and z coordinates of the pointcloud.
width_meters = 40
height_meters = 70
resoultion = 0.1# meter per pixel

width = int(width_meters*(1/resoultion))
height = int(height_meters*(1/resoultion))
# initiliaze grid. widhtxheight, resolution is in decimeters
input_data = np.zeros([len(data), width, height, 1], dtype=np.int8)
target_data = np.zeros([len(labels), 2])

for counter, data_point in enumerate(data):
    # initiliaze grid. 1200x1200, resolution is in decimeters

    data_x = data_point[:, 0]
    data_z = data_point[:, 2]

    data_grid_2D, edgesX, edgesZ = np.histogram2d(data_x, data_z, bins=[width, height],
                                                  range=[[int(-width_meters/2), int(width_meters/2)], [0, int(height_meters)]])

    #fig = plt.figure()
    #plt.imshow(data_grid_2D)

    data_grid_2D = np.reshape(data_grid_2D, (width, height, 1))
    input_data[counter, :, :, :] = data_grid_2D
    target_data[counter] = [labels[counter][0]+75, labels[counter][2]] # we shift the X data by 75 meters to positive.
    # This is neccessary, because we only want positive numbers (or only negative) for the output of NN (because of RELU)
    # This can be corrected after training. During test, simply substract 75 from the X dimension of the output of NN.

training_input_data = input_data[:-448, :, :, :]
training_target_data = target_data[:-448, :]
validation_input_data = input_data[-448:, :, :, :]
validation_target_data = target_data[-448:, :]

es = keras.callbacks.EarlyStopping(monitor='val_loss',mode='min',patience=10,verbose=1,restore_best_weights=True)

model = BEV2dCentroid([width, height, 1]).model
loss_fn = keras.losses.MeanAbsoluteError()
metric = keras.metrics.MeanAbsoluteError()

model.compile(loss=loss_fn, optimizer='adam', metrics=[metric])
model.fit(training_input_data, training_target_data, validation_data=(validation_input_data,validation_target_data),
          epochs=250, batch_size=16, callbacks=[es])

test_target = validation_target_data[0]
test_data = validation_input_data[0]
test_data = np.reshape(test_data, (1, width, height, 1))
test_data = tf.cast(test_data, tf.float32)
#test_data.astype('float64')
predicted_labes = model.predict(test_data, batch_size=1)

print('Test label X:',test_target[0]-75, 'Test label Y:',test_target[1],'\n','Pred. label X:',
      predicted_labes[0,0]-75, 'Pred. label Y:',predicted_labes[0,1])

model.save('saved_model_mobilenet_ES_1')
