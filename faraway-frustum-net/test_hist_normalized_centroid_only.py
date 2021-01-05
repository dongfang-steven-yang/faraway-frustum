from tensorflow import keras
import pickle
import numpy as np
import tensorflow as tf
from sklearn import mixture

#from models.models import BEV2dCentroid
def get_nonZero_indices(list_data):
    indices = []

    for i, data_sample in enumerate(list_data):
        if len(data_sample) == 0:
            continue
        indices.append(i)
    return indices


def pointcloud_clustering(clusters, method='histogram'):
    clf = mixture.GaussianMixture(n_components=2, covariance_type='full')

    positions = []
    # for i in range(len(boxes_img)):
    for cluster in clusters:
        if len(cluster) == 0:
            positions.append(None)
        else:
            if method == 'average':
                pos = [cluster[:, 0].mean(), cluster[:, 1].mean(), cluster[:, 2].mean()]
            elif method == 'mix_gaussian':
                if len(cluster[:, [0, 2]]) < 3:
                    pos = [cluster[:, 0].mean(), cluster[:, 1].mean(), cluster[:, 2].mean()]
                else:
                    clf.fit(cluster[:, [0, 2]])
                    k = np.argmax(np.argsort(clf.covariances_[:, 0, 0]) + np.argsort(clf.covariances_[:, 1, 1]))
                    pos = [clf.means_[k, 0], None, clf.means_[k, 1]]
            elif method == 'histogram':
                pos = []
                for j in range(3):
                    hist = np.histogram(cluster[:, j])
                    k = np.argmax(hist[0])
                    pos.append((hist[1][k] + hist[1][k + 1]) / 2)
            else:
                raise Exception('Invalid definition of method.')
            positions.append(tuple(pos))

    return positions

def preprocess_data(pointcloud_frustum_data, bb_label_data, centroids=[], width_meters=15, height_meters=20, resolution=0.1, bb_shift=75):

    # Preprocesses data for neural network training.

    # First, convert raw frustum pointcloud (PC) data -> A 2D bird's eye view frustum PC image with a fixed width and height.
    # Origin of this new frustum PC image is clustering centroids. Follow these steps:
    # 1 - find the histogram (cluster) centroids (X and Z) in the raw frustum point cloud data.
    # 2 - convert the frustum raw pointcloud coordinates into a new coordinate system whose origin is histogram centroids
    # 3 - Now, make a grid image via 2D histograming (different from step 1!). The value in each cell in the grid is
    #     equal to the number of 3D points that falls into that cell. The range of the 2D histogram is defined by
    #     -width_meters/2 to width_meters/2 and 0 to height_meters, number of bins depend on resolution.
    #     If you face memory issues for preallocating NN weights, try to increase resolution (it will make your image smaller)

    # Then, preprocess the label data. We want to do 2 things here:
    # 1- Make sure everything is positive (the final output of the NN will be positive only). This is very straight
    #    forward: Just add a big positive number (e.g 75) to the bounding box center coordinates. We do this via bb_shift
    # 2- Change the coordinate system of the labels to a new coordinate system whose origin is histogram
    #    centroids of input data (yes, the same step as step 1 for input data processing)

    # initiliaze Bird's Eye View (BEV) grid. WidthxHeight, resolution is defined explicitly below.
    # Take only x and z coordinates of the pointcloud.

    data = pointcloud_frustum_data
    labels = bb_label_data
    #centroids = pointcloud_clustering(data)

    width_meters_2 = width_meters
    height_meters_2 = height_meters
    resolution_2 = resolution  # meter per pixel

    width_2 = int(width_meters_2 * (1 / resolution_2))
    height_2 = int(height_meters_2 * (1 / resolution_2))

    # initiliaze grid. widhtxheight, resolution is in decimeters
    input_data = np.zeros([len(data), width_2, height_2, 1], dtype=np.int8)
    target_data = np.zeros([len(labels), 3])

    for counter, (data_point, centroid) in enumerate(zip(data, centroids)):

        #if len(data_point)==0:
        #continue
        data_x = data_point[:, 0]
        data_z = data_point[:, 2]

        centroid_x = centroid[0]
        # print(counter)
        centroid_z = centroid[2]

        data_x_new_coordinate = data_x - centroid_x
        data_z_new_coordinate = data_z - centroid_z

        # Here we get the centroid of the pointcloud frustum

        data_grid_2D_new, edgesX_new, edgesZ_new = np.histogram2d(data_x_new_coordinate, data_z_new_coordinate,
                                                                  bins=[width_2, height_2],
                                                                  range=[[int((-width_meters_2 / 2) - centroid_x),
                                                                          int((width_meters_2 / 2) - centroid_x)],
                                                                         [(0 - centroid_z),
                                                                          int(height_meters_2 - centroid_z)]])


        #fig = plt.figure()
        #plt.imshow(data_grid_2D_new)

        data_grid_2D_new = np.reshape(data_grid_2D_new, (width_2, height_2, 1))
        input_data[counter, :, :, :] = data_grid_2D_new

        # target_data[counter] = [labels[counter][0]+75, labels[counter][2]]
        target_data[counter] = [labels[counter][0] + bb_shift - centroid_x, labels[counter][2] +
                                bb_shift - centroid_z, labels[counter][6] + 3.14]
        # we shift the X and Z data by bb_shift meters to positive.
        # This is neccessary, because we only want positive numbers (or only negative) for the output of NN (because of RELU)
        # This can be corrected after training. During test, simply substract bb_shift from the X and Z dimension of the output of NN.

    return input_data, target_data


def test():
    bb_shift = 75
    validation_data = pickle.load(
        open('/localdata/yurtsever.2/Data/faraway_frustum/data_split01_val_Pedestrian/samples.p', 'rb'))
    validation_labels = pickle.load(
        open('/localdata/yurtsever.2/Data/faraway_frustum/data_split01_val_Pedestrian/labels.p', 'rb'))

    # validation_labels_Z = [data_point[2] for data_point in validation_labels]
    # sorted(validation_labels_Z,reverse=True)
    faraway_indices = []
    for i, val_label in enumerate(validation_labels):
        if val_label[2] > 60:
            faraway_indices.append(i)

    assert len(validation_data) == len(validation_labels)
    validation_indices = get_nonZero_indices(validation_data)
    validation_data = [validation_data[i] for i in validation_indices]
    validation_labels = [validation_labels[i] for i in validation_indices]

    validation_centroids = pointcloud_clustering(validation_data)
    validation_input_data, validation_target_data = preprocess_data(validation_data, validation_labels,
                                                                    validation_centroids)


    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    model = keras.models.load_model('saved_models/saved_model_hist_norm_centroid_1')

    test_target = validation_labels[faraway_indices[5]] # ground truth untouched
    test_centroid = validation_centroids[faraway_indices[5]] # need to change input shape of val data to use the trained NN
    test_data = validation_input_data[faraway_indices[5]]

    test_data = np.reshape(test_data, (1, test_data.shape[0], test_data.shape[1], 1))
    test_data = tf.cast(test_data, tf.float32)
    # test_data.astype('float64')
    predicted_labels = model.predict(test_data, batch_size=1)

    #validation_labels[8][0] = validation_target_data[8][0] - bb_shift + validation_centroids[8][0]

    #predicted_labels has 1 extra dimension because of tensorflow, ignore it.

    print('Test label X:', test_target[0] , 'Test label Y:', test_target[1], 'Test label Z:', test_target[2],
          'Test label REST:', test_target[3:],  '\n', 'Pred. label X:', predicted_labels[0, 0] - bb_shift + test_centroid[0],
          'Pred. label Y: - ', 'Pred. label Z:', predicted_labels[0, 1] - bb_shift + test_centroid[2],
          'Pred. label REST: -,-,-,', predicted_labels[0,2] - 3.14)
    print(test_centroid)
if __name__ == '__main__':
    test()
