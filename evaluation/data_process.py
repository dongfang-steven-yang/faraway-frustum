import os
import numpy as np
import pandas as pd
from pathlib import Path
import shutil

def fuse_result(path_in1,path_in2,path_out,name_sample,distance1,distance2):
    path_in1 = path_in1 + '/{}.txt'.format(name_sample)
    path_in2 = path_in2 + '/{}.txt'.format(name_sample)
    path_out = path_out + '/{}.txt'.format(name_sample)
    try:
        dt1 = pd.read_csv(str(Path(path_in1)), header=None, sep=' ')
        dt1.columns = ['type', 'truncated', 'occluded', 'alpha', 'bbox_left', 'bbox_top',
                      'bbox_right', 'bbox_bottom', 'height', 'width', 'length', 'pos_x', 'pos_y', 'pos_z', 'rot_y','score']
        rowNum1 = dt1.shape[0]
        dt2 = pd.read_csv(str(Path(path_in2)), header=None, sep=' ')
        dt2.columns = ['type', 'truncated', 'occluded', 'alpha', 'bbox_left', 'bbox_top',
                      'bbox_right', 'bbox_bottom', 'height', 'width', 'length', 'pos_x', 'pos_y', 'pos_z',
                      'rot_y', 'score']
        rowNum2 = dt2.shape[0]
        rowNum=rowNum1+rowNum2

        with open(path_out,'w') as resulttxt:
            # resulttxt.truncate()
            for i in range(0,rowNum1):
                d = dt1.loc[i]
                distance_his = d['pos_z']
                classes_his = d['type']
                if int(distance_his) <= int(distance1) and classes_his == 'Pedestrian' :
                        resulttxt.write(str(d['type']) + ' ')
                        resulttxt.write(str(d['truncated']) + ' ')
                        resulttxt.write(str(d['occluded']) + ' ')
                        resulttxt.write(str(d['alpha']) + ' ')
                        resulttxt.write(str(d['bbox_left']) + ' ')
                        resulttxt.write(str(d['bbox_top']) + ' ')
                        resulttxt.write(str(d['bbox_right']) + ' ')
                        resulttxt.write(str(d['bbox_bottom']) + ' ')
                        # resulttxt.write(str(1.75) + ' ')
                        resulttxt.write(str(d['height']) + ' ')
                        resulttxt.write(str(d['width']) + ' ')
                        resulttxt.write(str(d['length']) + ' ')
                        resulttxt.write(str(d['pos_x']) + ' ')
                        resulttxt.write(str(d['pos_y']) + ' ')
                        # resulttxt.write(str(1.30) + ' ')
                        resulttxt.write(str(d['pos_z']) + ' ')
                        resulttxt.write(str(d['rot_y']) + ' ')
                        resulttxt.write(str(d['score']) + '\n')

                elif int(distance_his) <= int(distance2) and classes_his == 'Car' :
                        resulttxt.write(str(d['type']) + ' ')
                        resulttxt.write(str(d['truncated']) + ' ')
                        resulttxt.write(str(d['occluded']) + ' ')
                        resulttxt.write(str(d['alpha']) + ' ')
                        resulttxt.write(str(d['bbox_left']) + ' ')
                        resulttxt.write(str(d['bbox_top']) + ' ')
                        resulttxt.write(str(d['bbox_right']) + ' ')
                        resulttxt.write(str(d['bbox_bottom']) + ' ')
                        # resulttxt.write(str(1.75) + ' ')
                        resulttxt.write(str(d['height']) + ' ')
                        resulttxt.write(str(d['width']) + ' ')
                        resulttxt.write(str(d['length']) + ' ')
                        resulttxt.write(str(d['pos_x']) + ' ')
                        resulttxt.write(str(d['pos_y']) + ' ')
                        # resulttxt.write(str(1.30) + ' ')
                        resulttxt.write(str(d['pos_z']) + ' ')
                        resulttxt.write(str(d['rot_y']) + ' ')
                        resulttxt.write(str(d['score']) + '\n')

                elif classes_his == 'Cyclist' :
                        resulttxt.write(str(d['type']) + ' ')
                        resulttxt.write(str(d['truncated']) + ' ')
                        resulttxt.write(str(d['occluded']) + ' ')
                        resulttxt.write(str(d['alpha']) + ' ')
                        resulttxt.write(str(d['bbox_left']) + ' ')
                        resulttxt.write(str(d['bbox_top']) + ' ')
                        resulttxt.write(str(d['bbox_right']) + ' ')
                        resulttxt.write(str(d['bbox_bottom']) + ' ')
                        # resulttxt.write(str(1.75) + ' ')
                        resulttxt.write(str(d['height']) + ' ')
                        resulttxt.write(str(d['width']) + ' ')
                        resulttxt.write(str(d['length']) + ' ')
                        resulttxt.write(str(d['pos_x']) + ' ')
                        resulttxt.write(str(d['pos_y']) + ' ')
                        # resulttxt.write(str(1.30) + ' ')
                        resulttxt.write(str(d['pos_z']) + ' ')
                        resulttxt.write(str(d['rot_y']) + ' ')
                        resulttxt.write(str(d['score']) + '\n')

                else:
                        resulttxt.write(str('DontCare') + ' ')
                        resulttxt.write(str(-1.0) + ' ')
                        resulttxt.write(str(-1) + ' ')
                        resulttxt.write(str(-10.0) + ' ')
                        resulttxt.write(str(d['bbox_left']) + ' ')  # boxes
                        resulttxt.write(str(d['bbox_top']) + ' ')
                        resulttxt.write(str(d['bbox_right']) + ' ')
                        resulttxt.write(str(d['bbox_bottom']) + ' ')
                        resulttxt.write(str(-1.0) + ' ')
                        resulttxt.write(str(-1.0) + ' ')
                        resulttxt.write(str(-1.0) + ' ')
                        resulttxt.write(str(-1000.0) + ' ')
                        resulttxt.write(str(-1000.0) + ' ')
                        resulttxt.write(str(-1000.0) + ' ')
                        resulttxt.write(str(-10.0) + ' ')
                        resulttxt.write(str(d['score']) + '\n')

            for j in range(0, rowNum2):
                dd= dt2.loc[j]
                distance_his = dd['pos_z']
                classes_his = dd['type']
                if int(distance_his) >= int(distance1) and classes_his == 'Pedestrian':
                    resulttxt.write(str(dd['type']) + ' ')
                    resulttxt.write(str(dd['truncated']) + ' ')
                    resulttxt.write(str(dd['occluded']) + ' ')
                    resulttxt.write(str(dd['alpha']) + ' ')
                    resulttxt.write(str(dd['bbox_left']) + ' ')
                    resulttxt.write(str(dd['bbox_top']) + ' ')
                    resulttxt.write(str(dd['bbox_right']) + ' ')
                    resulttxt.write(str(dd['bbox_bottom']) + ' ')
                    # resulttxt.write(str(1.75) + ' ')
                    resulttxt.write(str(dd['height']) + ' ')
                    resulttxt.write(str(dd['width']) + ' ')
                    resulttxt.write(str(dd['length']) + ' ')
                    resulttxt.write(str(dd['pos_x']) + ' ')
                    resulttxt.write(str(dd['pos_y']) + ' ')
                    # resulttxt.write(str(1.30) + ' ')
                    resulttxt.write(str(dd['pos_z']) + ' ')
                    resulttxt.write(str(dd['rot_y']) + ' ')
                    resulttxt.write(str(dd['score']) + '\n')

                elif int(distance_his) >= int(distance2) and classes_his == 'Car':
                    resulttxt.write(str(dd['type']) + ' ')
                    resulttxt.write(str(dd['truncated']) + ' ')
                    resulttxt.write(str(dd['occluded']) + ' ')
                    resulttxt.write(str(dd['alpha']) + ' ')
                    resulttxt.write(str(dd['bbox_left']) + ' ')
                    resulttxt.write(str(dd['bbox_top']) + ' ')
                    resulttxt.write(str(dd['bbox_right']) + ' ')
                    resulttxt.write(str(dd['bbox_bottom']) + ' ')
                    # resulttxt.write(str(1.75) + ' ')
                    resulttxt.write(str(dd['height']) + ' ')
                    resulttxt.write(str(dd['width']) + ' ')
                    resulttxt.write(str(dd['length']) + ' ')
                    resulttxt.write(str(dd['pos_x']) + ' ')
                    resulttxt.write(str(dd['pos_y']) + ' ')
                    # resulttxt.write(str(1.30) + ' ')
                    resulttxt.write(str(dd['pos_z']) + ' ')
                    resulttxt.write(str(dd['rot_y']) + ' ')
                    resulttxt.write(str(dd['score']) + '\n')

                else:
                    pass
    except:
        with open(path_out, 'w') as resulttxt:
            resulttxt.write(str('DontCare') + ' ')
            resulttxt.write(str(-1.0) + ' ')
            resulttxt.write(str(-1) + ' ')
            resulttxt.write(str(-10.0) + ' ')
            resulttxt.write(str(503.00) + ' ')  # boxes
            resulttxt.write(str(154.00) + ' ')
            resulttxt.write(str(544.00) + ' ')
            resulttxt.write(str(201.00) + ' ')
            resulttxt.write(str(-1.0) + ' ')
            resulttxt.write(str(-1.0) + ' ')
            resulttxt.write(str(-1.0) + ' ')
            resulttxt.write(str(-1000.0) + ' ')
            resulttxt.write(str(-1000.0) + ' ')
            resulttxt.write(str(-1000.0) + ' ')
            resulttxt.write(str(-10.0) + ' ')
            resulttxt.write(str(0.99) + '\n')

def reduce(path_in,path_out,name_sample,distance):
    path_in = path_in + '/{}.txt'.format(name_sample)
    path_out = path_out + '/{}.txt'.format(name_sample)
    try:
        dt = pd.read_csv(str(Path(path_in)), header=None, sep=' ')
        dt.columns = ['type', 'truncated', 'occluded', 'alpha', 'bbox_left', 'bbox_top',
                      'bbox_right', 'bbox_bottom', 'height', 'width', 'length', 'pos_x', 'pos_y', 'pos_z', 'rot_y','score']
        rowNum = dt.shape[0]
        with open(path_out,'w') as resulttxt:
            # resulttxt.truncate()
            for i in range(0,rowNum):
                d = dt.loc[i]
                distance_his = d['pos_z']
                classes_his = d['type']
                if int(distance_his) >= int(distance) and classes_his == 'Pedestrian' :
                        resulttxt.write(str(d['type']) + ' ')
                        resulttxt.write(str(d['truncated']) + ' ')
                        resulttxt.write(str(d['occluded']) + ' ')
                        resulttxt.write(str(d['alpha']) + ' ')
                        resulttxt.write(str(d['bbox_left']) + ' ')
                        resulttxt.write(str(d['bbox_top']) + ' ')
                        resulttxt.write(str(d['bbox_right']) + ' ')
                        resulttxt.write(str(d['bbox_bottom']) + ' ')
                        # resulttxt.write(str(1.75) + ' ')
                        resulttxt.write(str(d['height']) + ' ')
                        resulttxt.write(str(d['width']) + ' ')
                        resulttxt.write(str(d['length']) + ' ')
                        resulttxt.write(str(d['pos_x']) + ' ')
                        resulttxt.write(str(d['pos_y']) + ' ')
                        # resulttxt.write(str(1.30) + ' ')
                        resulttxt.write(str(d['pos_z']) + ' ')
                        resulttxt.write(str(d['rot_y']) + ' ')
                        resulttxt.write(str(d['score']) + '\n')

                elif int(distance_his) >= int(distance) and classes_his == 'Car' :
                        resulttxt.write(str(d['type']) + ' ')
                        resulttxt.write(str(d['truncated']) + ' ')
                        resulttxt.write(str(d['occluded']) + ' ')
                        resulttxt.write(str(d['alpha']) + ' ')
                        resulttxt.write(str(d['bbox_left']) + ' ')
                        resulttxt.write(str(d['bbox_top']) + ' ')
                        resulttxt.write(str(d['bbox_right']) + ' ')
                        resulttxt.write(str(d['bbox_bottom']) + ' ')
                        # resulttxt.write(str(1.75) + ' ')
                        resulttxt.write(str(d['height']) + ' ')
                        resulttxt.write(str(d['width']) + ' ')
                        resulttxt.write(str(d['length']) + ' ')
                        resulttxt.write(str(d['pos_x']) + ' ')
                        resulttxt.write(str(d['pos_y']) + ' ')
                        # resulttxt.write(str(1.30) + ' ')
                        resulttxt.write(str(d['pos_z']) + ' ')
                        resulttxt.write(str(d['rot_y']) + ' ')
                        resulttxt.write(str(d['score']) + '\n')
                elif int(distance_his) >= int(distance) and classes_his == 'Cyclist' :
                        resulttxt.write(str(d['type']) + ' ')
                        resulttxt.write(str(d['truncated']) + ' ')
                        resulttxt.write(str(d['occluded']) + ' ')
                        resulttxt.write(str(d['alpha']) + ' ')
                        resulttxt.write(str(d['bbox_left']) + ' ')
                        resulttxt.write(str(d['bbox_top']) + ' ')
                        resulttxt.write(str(d['bbox_right']) + ' ')
                        resulttxt.write(str(d['bbox_bottom']) + ' ')
                        # resulttxt.write(str(1.75) + ' ')
                        resulttxt.write(str(d['height']) + ' ')
                        resulttxt.write(str(d['width']) + ' ')
                        resulttxt.write(str(d['length']) + ' ')
                        resulttxt.write(str(d['pos_x']) + ' ')
                        resulttxt.write(str(d['pos_y']) + ' ')
                        # resulttxt.write(str(1.30) + ' ')
                        resulttxt.write(str(d['pos_z']) + ' ')
                        resulttxt.write(str(d['rot_y']) + ' ')
                        resulttxt.write(str(d['score']) + '\n')

                else:
                        resulttxt.write(str('DontCare') + ' ')
                        resulttxt.write(str(-1.0) + ' ')
                        resulttxt.write(str(-1) + ' ')
                        resulttxt.write(str(-10.0) + ' ')
                        resulttxt.write(str(d['bbox_left']) + ' ')  # boxes
                        resulttxt.write(str(d['bbox_top']) + ' ')
                        resulttxt.write(str(d['bbox_right']) + ' ')
                        resulttxt.write(str(d['bbox_bottom']) + ' ')
                        resulttxt.write(str(-1.0) + ' ')
                        resulttxt.write(str(-1.0) + ' ')
                        resulttxt.write(str(-1.0) + ' ')
                        resulttxt.write(str(-1000.0) + ' ')
                        resulttxt.write(str(-1000.0) + ' ')
                        resulttxt.write(str(-1000.0) + ' ')
                        resulttxt.write(str(-10.0) + ' ')
                        resulttxt.write(str(d['score']) + '\n')
    except:
        with open(path_out, 'w') as resulttxt:
            resulttxt.write(str('DontCare') + ' ')
            resulttxt.write(str(-1.0) + ' ')
            resulttxt.write(str(-1) + ' ')
            resulttxt.write(str(-10.0) + ' ')
            resulttxt.write(str(503.00) + ' ')  # boxes
            resulttxt.write(str(154.00) + ' ')
            resulttxt.write(str(544.00) + ' ')
            resulttxt.write(str(201.00) + ' ')
            resulttxt.write(str(-1.0) + ' ')
            resulttxt.write(str(-1.0) + ' ')
            resulttxt.write(str(-1.0) + ' ')
            resulttxt.write(str(-1000.0) + ' ')
            resulttxt.write(str(-1000.0) + ' ')
            resulttxt.write(str(-1000.0) + ' ')
            resulttxt.write(str(-10.0) + ' ')
            resulttxt.write(str(0.99) + '\n')

def reduce_label(path_in,path_out,name_sample,distance):
    path_in = path_in + '/{}.txt'.format(name_sample)
    path_out = path_out + '/{}.txt'.format(name_sample)
    try:
        dt = pd.read_csv(str(Path(path_in)), header=None, sep=' ')
        dt.columns = ['type', 'truncated', 'occluded', 'alpha', 'bbox_left', 'bbox_top',
                      'bbox_right', 'bbox_bottom', 'height', 'width', 'length', 'pos_x', 'pos_y', 'pos_z', 'rot_y']
        rowNum = dt.shape[0]
        with open(path_out,'w') as resulttxt:
            # resulttxt.truncate()
            for i in range(0,rowNum):
                d = dt.loc[i]
                distance_his = d['pos_z']
                classes_his = d['type']
                if int(distance_his) >= int(distance) and classes_his == 'Pedestrian' :
                        resulttxt.write(str(d['type']) + ' ')
                        resulttxt.write(str(d['truncated']) + ' ')
                        resulttxt.write(str(d['occluded']) + ' ')
                        resulttxt.write(str(d['alpha']) + ' ')
                        resulttxt.write(str(d['bbox_left']) + ' ')
                        resulttxt.write(str(d['bbox_top']) + ' ')
                        resulttxt.write(str(d['bbox_right']) + ' ')
                        resulttxt.write(str(d['bbox_bottom']) + ' ')
                        # resulttxt.write(str(1.75) + ' ')
                        resulttxt.write(str(d['height']) + ' ')
                        resulttxt.write(str(d['width']) + ' ')
                        resulttxt.write(str(d['length']) + ' ')
                        resulttxt.write(str(d['pos_x']) + ' ')
                        resulttxt.write(str(d['pos_y']) + ' ')
                        # resulttxt.write(str(1.30) + ' ')
                        resulttxt.write(str(d['pos_z']) + ' ')
                        resulttxt.write(str(d['rot_y']) + '\n')

                elif int(distance_his) >= int(distance) and classes_his == 'Car' :
                        resulttxt.write(str(d['type']) + ' ')
                        resulttxt.write(str(d['truncated']) + ' ')
                        resulttxt.write(str(d['occluded']) + ' ')
                        resulttxt.write(str(d['alpha']) + ' ')
                        resulttxt.write(str(d['bbox_left']) + ' ')
                        resulttxt.write(str(d['bbox_top']) + ' ')
                        resulttxt.write(str(d['bbox_right']) + ' ')
                        resulttxt.write(str(d['bbox_bottom']) + ' ')
                        # resulttxt.write(str(1.75) + ' ')
                        resulttxt.write(str(d['height']) + ' ')
                        resulttxt.write(str(d['width']) + ' ')
                        resulttxt.write(str(d['length']) + ' ')
                        resulttxt.write(str(d['pos_x']) + ' ')
                        resulttxt.write(str(d['pos_y']) + ' ')
                        # resulttxt.write(str(1.30) + ' ')
                        resulttxt.write(str(d['pos_z']) + ' ')
                        resulttxt.write(str(d['rot_y']) + '\n')
                elif int(distance_his) >= int(distance) and classes_his == 'Cyclist' :
                        resulttxt.write(str(d['type']) + ' ')
                        resulttxt.write(str(d['truncated']) + ' ')
                        resulttxt.write(str(d['occluded']) + ' ')
                        resulttxt.write(str(d['alpha']) + ' ')
                        resulttxt.write(str(d['bbox_left']) + ' ')
                        resulttxt.write(str(d['bbox_top']) + ' ')
                        resulttxt.write(str(d['bbox_right']) + ' ')
                        resulttxt.write(str(d['bbox_bottom']) + ' ')
                        # resulttxt.write(str(1.75) + ' ')
                        resulttxt.write(str(d['height']) + ' ')
                        resulttxt.write(str(d['width']) + ' ')
                        resulttxt.write(str(d['length']) + ' ')
                        resulttxt.write(str(d['pos_x']) + ' ')
                        resulttxt.write(str(d['pos_y']) + ' ')
                        # resulttxt.write(str(1.30) + ' ')
                        resulttxt.write(str(d['pos_z']) + ' ')
                        resulttxt.write(str(d['rot_y']) + '\n')

                else:
                        resulttxt.write(str('DontCare') + ' ')
                        resulttxt.write(str(-1.0) + ' ')
                        resulttxt.write(str(-1) + ' ')
                        resulttxt.write(str(-10.0) + ' ')
                        resulttxt.write(str(d['bbox_left']) + ' ')  # boxes
                        resulttxt.write(str(d['bbox_top']) + ' ')
                        resulttxt.write(str(d['bbox_right']) + ' ')
                        resulttxt.write(str(d['bbox_bottom']) + ' ')
                        resulttxt.write(str(-1.0) + ' ')
                        resulttxt.write(str(-1.0) + ' ')
                        resulttxt.write(str(-1.0) + ' ')
                        resulttxt.write(str(-1000.0) + ' ')
                        resulttxt.write(str(-1000.0) + ' ')
                        resulttxt.write(str(-1000.0) + ' ')
                        resulttxt.write(str(-10.0) + '\n')

    except:
        with open(path_out, 'w') as resulttxt:
            resulttxt.write(str('DontCare') + ' ')
            resulttxt.write(str(-1.0) + ' ')
            resulttxt.write(str(-1) + ' ')
            resulttxt.write(str(-10.0) + ' ')
            resulttxt.write(str(503.00) + ' ')  # boxes
            resulttxt.write(str(154.00) + ' ')
            resulttxt.write(str(544.00) + ' ')
            resulttxt.write(str(201.00) + ' ')
            resulttxt.write(str(-1.0) + ' ')
            resulttxt.write(str(-1.0) + ' ')
            resulttxt.write(str(-1.0) + ' ')
            resulttxt.write(str(-1000.0) + ' ')
            resulttxt.write(str(-1000.0) + ' ')
            resulttxt.write(str(-1000.0) + ' ')
            resulttxt.write(str(-10.0) + '\n')


def check_subset(path_in,path_out,name_sample):
    path_in= path_in + '/{}.txt'.format(name_sample)
    path_out = path_out+ '/{}.txt'.format(name_sample)
    if os.path.isfile(path_in):
        shutil.copyfile(path_in, path_out)
        print(f'copy {name_sample} to subset{name_sample}')

def re2se(path_in,path_out,name_sample,sequence):
    path_in= path_in + '/{}.txt'.format(name_sample)
    path_out = path_out+ '/{}.txt'.format(sequence)
    if os.path.isfile(path_in):
        # mymovefile(path_in,path_out)
        shutil.copyfile(path_in, path_out)
        print(f'copy {name_sample} to subset{sequence}')


def main():
    function = 'eval_val'
    # fuse_result: fuse our faraway results with SOTA detector's results
    # eval_val: results process for evaluate val set(easy,mod,hard)
    # eval_sub: results process for evaluting subdataset (pd>60m or car>75m)
    if function == 'fuse_result':
        print(f'===============================>start processing for fusing results!')
        names_sample = []
        with open('./split/val.txt', 'r') as f:  # data split file for val
            for line in f:
                names_sample.append(list(line.strip('\n').split(',')))
        for i, name_sample in enumerate(names_sample):
            print('==> running sample ' + str(name_sample[0]) + ', index=%d' % i)
            fuse_result(path_in1='',#SOTA original val results
                    path_in2='',#Our original val results
                    path_out='', # Path of output
                    name_sample=str(name_sample), distance1=60, distance2=75)  # Ours: fuse our faraway results with SOTA detector's results

    elif function == 'eval_val':
        print(f'===============================>start processing for val evaluation data!')
        names_sample = []
        with open('./split/val.txt', 'r') as f:  #data split file for val
            for line in f:
                names_sample.append(list(line.strip('\n').split(',')))
        sequences=[]
        with open('./split/sequence.txt', 'r') as f: #data split file (sequence)
            for line in f:
                sequences.append(list(line.strip('\n').split(',')))

        for i, name_sample in enumerate(names_sample):
            sequence=str(sequences[i][0])
            print('==> running sample ' + str(name_sample[0]) + ', index=%d' % i)
            re2se(path_in='', #path of original val results (or val labels)
                      path_out='', #path of output
                      name_sample=str(name_sample[0]),sequence=sequence)  #change results to sequences

    elif function == 'eval_sub':
        print(f'===============================>start processing for subdataset evaluation data!')
        names_sample = []
        with open('./split/val_ped_60.txt.txt', 'r') as f:  # data split file for subset
            # use 'val_ped_60' for pd>60m or 'val_car_75' for car>75m
            for line in f:
                names_sample.append(list(line.strip('\n').split(',')))
        sequences = []
        with open('./split/sequence.txt', 'r') as f:  # data split file (sequence)
            for line in f:
                sequences.append(list(line.strip('\n').split(',')))
        for i, name_sample in enumerate(names_sample):
            sequence = str(sequences[i][0])
            print('==> running sample ' + str(name_sample[0]) + ', index=%d' % i)

            ###### if processing detection results, using the function below.
            reduce(path_in='', # path of detection results
                   path_out='', # path of output1 (detection results (reducing close objects) of subset))
                   name_sample=str(name_sample[0]),distance=60) #

            ###### if processing labels files, using the function below.
            # reduce_label(path_in='', # path of labels
            #       path_out='',  # path of output1 (labels (reducing close objects) of subset))
            #       name_sample=str(name_sample[0]),distance=60)

            re2se(path_in='',  # path of output1
                  path_out='',  # path of output2 (sequences of results (or labels) of subset)
                  name_sample=str(name_sample[0]), sequence=sequence)
    else:
        print('error function input!')

if __name__ == '__main__':
    main()