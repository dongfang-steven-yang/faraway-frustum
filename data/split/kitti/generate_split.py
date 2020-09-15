n_kitti_train = 7481
n_kitti_test = 7518
# n_kitti_train = 20
# n_kitti_test = 7518

with open('training.txt', 'w') as f:
    for i in range(n_kitti_train):
        if i == 0:
            f.write('%06d' % i)
        else:
            f.write('\n%06d' % i)

with open('eval.txt', 'w') as f:
    for i in range(n_kitti_train):
        if i == 0:
            f.write('%06d' % i)
        else:
            f.write('\n%06d' % i)

with open('testing.txt', 'w') as f:
    for i in range(n_kitti_test):
        if i == 0:
            f.write('%06d' % i)
        else:
            f.write('\n%06d' % i)