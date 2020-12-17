import numpy as np
import os

def calculate_score_2D():
    score_Pd_2D_easy =0
    score_Car_2D_easy = 0
    score_Cyc_2D_easy = 0
    score_Pd_2D_mod =0
    score_Car_2D_mod = 0
    score_Cyc_2D_mod = 0
    score_Pd_2D_hard =0
    score_Car_2D_hard = 0
    score_Cyc_2D_hard = 0
    path = os.getcwd()
    Pd_2D_data= np.loadtxt(path +'/results/dt/plot/pedestrian_detection.txt')
    Car_2D_data = np.loadtxt(path +'/results/dt/plot/car_detection.txt')
    Cyc_2D_data = np.loadtxt(path +'/results/dt/plot/cyclist_detection.txt')
    for i in range(0,11):
        score_Pd_2D_easy=score_Pd_2D_easy + Pd_2D_data[i][1]
        score_Pd_2D_mod = score_Pd_2D_mod + Pd_2D_data[i][2]
        score_Pd_2D_hard = score_Pd_2D_hard + Pd_2D_data[i][3]
        score_Car_2D_easy=score_Car_2D_easy + Car_2D_data[i][1]
        score_Car_2D_mod = score_Car_2D_mod + Car_2D_data[i][2]
        score_Car_2D_hard = score_Car_2D_hard + Car_2D_data[i][3]
        score_Cyc_2D_easy=score_Cyc_2D_easy + Cyc_2D_data[i][1]
        score_Cyc_2D_mod = score_Cyc_2D_mod + Cyc_2D_data[i][2]
        score_Cyc_2D_hard = score_Cyc_2D_hard + Cyc_2D_data[i][3]
    score_Pd_2D_easy = score_Pd_2D_easy/11
    score_Pd_2D_mod = score_Pd_2D_mod/11
    score_Pd_2D_hard = score_Pd_2D_hard/11
    score_Car_2D_easy =score_Car_2D_easy/11
    score_Car_2D_mod = score_Car_2D_mod / 11
    score_Car_2D_hard = score_Car_2D_hard / 11
    score_Cyc_2D_easy =score_Cyc_2D_easy/11
    score_Cyc_2D_mod = score_Cyc_2D_mod / 11
    score_Cyc_2D_hard = score_Cyc_2D_hard / 11
    with open(path +'/results/score_2D.txt', 'w') as resulttxt:
        resulttxt.truncate()
        resulttxt.write('2D Pedestrian Detection(easy,mod,hard) : ' + str(score_Pd_2D_easy) +' '+str(score_Pd_2D_mod) + ' '+ str(score_Pd_2D_hard)+'\n')
        resulttxt.write('2D Car Detection(easy,mod,hard) : ' + str(score_Car_2D_easy) +' '+str(score_Car_2D_mod) + ' '+ str(score_Car_2D_hard) +'\n')
        resulttxt.write('2D Cyclist(easy,mod,hard) : ' + str(score_Cyc_2D_easy) +' '+str(score_Cyc_2D_mod) + ' '+ str(score_Cyc_2D_hard) +'\n')
    print('2D score calculation done')


def calculate_score_BEV():
    score_Pd_BEV_easy =0
    score_Car_BEV_easy = 0
    score_Cyc_BEV_easy = 0
    score_Pd_BEV_mod =0
    score_Car_BEV_mod = 0
    score_Cyc_BEV_mod = 0
    score_Pd_BEV_hard =0
    score_Car_BEV_hard = 0
    score_Cyc_BEV_hard = 0
    path = os.getcwd()
    try:
        Pd_BEV_data= np.loadtxt(path + '/results/dt/plot/pedestrian_detection_ground.txt')
        print('==========>can calculate PD BEV')
        for i in range(0,11):
            score_Pd_BEV_easy=score_Pd_BEV_easy + Pd_BEV_data[i][1]
            score_Pd_BEV_mod = score_Pd_BEV_mod + Pd_BEV_data[i][2]
            score_Pd_BEV_hard = score_Pd_BEV_hard + Pd_BEV_data[i][3]
        score_Pd_BEV_easy = score_Pd_BEV_easy/11
        score_Pd_BEV_mod = score_Pd_BEV_mod/11
        score_Pd_BEV_hard = score_Pd_BEV_hard/11
        print(f'BEV Pedestrian Detection: {str(score_Pd_BEV_easy * 100)}')
    except:
        print('==========>cannot calculate PD BEV')
    try:
        Car_BEV_data = np.loadtxt(path + '/results/dt/plot/car_detection_ground.txt')
        print('==========>can calculate CAR BEV')
        for i in range(0,11):
            score_Car_BEV_easy=score_Car_BEV_easy + Car_BEV_data[i][1]
            score_Car_BEV_mod = score_Car_BEV_mod + Car_BEV_data[i][2]
            score_Car_BEV_hard = score_Car_BEV_hard + Car_BEV_data[i][3]
        score_Car_BEV_easy = score_Car_BEV_easy / 11
        score_Car_BEV_mod = score_Car_BEV_mod / 11
        score_Car_BEV_hard = score_Car_BEV_hard / 11
        print(f'BEV Car Detection: {str(score_Car_BEV_easy * 100)}')
    except:
        print('==========>cannot calculate CAR BEV')

    try:
        Cyc_BEV_data = np.loadtxt(path + '/results/dt/plot/cyclist_detection_ground.txt')
        print('==========>can calculate CYC BEV')
        for i in range(0,11):
            score_Cyc_BEV_easy=score_Cyc_BEV_easy + Cyc_BEV_data[i][1]
            score_Cyc_BEV_mod = score_Cyc_BEV_mod + Cyc_BEV_data[i][2]
            score_Cyc_BEV_hard = score_Cyc_BEV_hard + Cyc_BEV_data[i][3]
        score_Cyc_BEV_easy = score_Cyc_BEV_easy / 11
        score_Cyc_BEV_mod = score_Cyc_BEV_mod / 11
        score_Cyc_BEV_hard = score_Cyc_BEV_hard / 11
        print(f'BEV Cyclist Detection: {str(score_Cyc_BEV_easy * 100)} ')
    except:
        print('==========>cannot calculate CYC BEV')
    print('BEV score calculation done')


def calculate_socre_3D():
    score_Pd_3d_easy =0
    score_Car_3d_easy = 0
    score_Cyc_3d_easy = 0
    score_Pd_3d_mod =0
    score_Car_3d_mod = 0
    score_Cyc_3d_mod = 0
    score_Pd_3d_hard =0
    score_Car_3d_hard = 0
    score_Cyc_3d_hard = 0
    path = os.getcwd()
    try:
        Pd_3d_data= np.loadtxt(path + '/results/dt/plot/pedestrian_detection_3d.txt')
        print('==========>can calculate PD 3d')
        for i in range(0, 11):
            score_Pd_3d_easy = score_Pd_3d_easy + Pd_3d_data[i][1]
            score_Pd_3d_mod = score_Pd_3d_mod + Pd_3d_data[i][2]
            score_Pd_3d_hard = score_Pd_3d_hard + Pd_3d_data[i][3]
        score_Pd_3d_easy = score_Pd_3d_easy / 11
        score_Pd_3d_mod = score_Pd_3d_mod / 11
        score_Pd_3d_hard = score_Pd_3d_hard / 11
        print(f'3d Pedestrian Detection: {str(score_Pd_3d_easy*100)}')
    except:
        print('==========>cannot calculate PD 3D')

    try:
        Car_3d_data= np.loadtxt(path + '/results/dt/plot/car_detection_3d.txt')
        print('==========>can calculate CAR 3d')
        for i in range(0, 11):
            score_Car_3d_easy = score_Car_3d_easy + Car_3d_data[i][1]
            score_Car_3d_mod = score_Car_3d_mod + Car_3d_data[i][2]
            score_Car_3d_hard = score_Car_3d_hard + Car_3d_data[i][3]
        score_Car_3d_easy = score_Car_3d_easy / 11
        score_Car_3d_mod = score_Car_3d_mod / 11
        score_Car_3d_hard = score_Car_3d_hard / 11
        print(f'3d Car Detection: {str(score_Car_3d_easy*100)}')
    except:
        print('==========>cannot calculate CAR 3D')


    try:
        Cyc_3d_data= np.loadtxt(path + '/results/dt/plot/cyclist_detection_3d.txt')
        print('==========>can calculate Cyc 3d')
        for i in range(0, 11):
            score_Cyc_3d_easy = score_Cyc_3d_easy + Cyc_3d_data[i][1]
            score_Cyc_3d_mod = score_Cyc_3d_mod + Cyc_3d_data[i][2]
            score_Cyc_3d_hard = score_Cyc_3d_hard + Cyc_3d_data[i][3]
        score_Cyc_3d_easy = score_Cyc_3d_easy / 11
        score_Cyc_3d_mod = score_Cyc_3d_mod / 11
        score_Cyc_3d_hard = score_Cyc_3d_hard / 11
        print(f'3d Cyclist Detection: {str(score_Cyc_3d_easy*100)}')
    except:
        print('==========>cannot calculate Cyc 3D')

    print('3d score calculation done')


def main():
    calculate_score_BEV()
    calculate_socre_3D()
    # calculate_score_2D()

if __name__ == '__main__':
    main()
