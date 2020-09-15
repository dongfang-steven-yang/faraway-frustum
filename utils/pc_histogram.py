from sklearn import mixture
import numpy as np


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