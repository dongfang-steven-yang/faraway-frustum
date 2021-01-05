from tensorflow import keras
import pickle
import numpy as np
import tensorflow as tf

from models.models import BEV2dCentroid

def test():

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    #data = pickle.load(open('data/samples.p', 'rb'))
    #labels = pickle.load(open('data/labels.p', 'rb'))

    data = pickle.load(open('data/data_pcdet_val/samples.p', 'rb'))
    labels = pickle.load(open('data/data_pcdet_val/labels.p', 'rb'))

    assert len(data) == len(labels)

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

    #training_input_data = input_data[:-448, :, :, :]
    #training_target_data = target_data[:-448, :]
    validation_input_data = input_data
    validation_target_data = target_data

    model = keras.models.load_model('saved_model_resnet50_1')

    test_target = validation_target_data[8]
    test_data = validation_input_data[8]
    test_data = np.reshape(test_data, (1, width, height, 1))
    test_data = tf.cast(test_data, tf.float32)
    # test_data.astype('float64')
    predicted_labes = model.predict(test_data, batch_size=1)

    print('Test label X:', test_target[0] - 75, 'Test label Z:', test_target[1], '\n', 'Pred. label X:',
          predicted_labes[0, 0] - 75, 'Pred. label Z:', predicted_labes[0, 1])

if __name__ == '__main__':
    test()
