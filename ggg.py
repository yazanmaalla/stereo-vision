import cv2
import numpy as np
import scipy.io as sio
from scipy.linalg import null_space
import matplotlib.pyplot as plt
import sys
from sklearn.neighbors import NearestNeighbors
from mpl_toolkits.mplot3d import Axes3D


NN_RATIO = 0.7
RANSAC_N = 1000
SIFT_SIZE = 7
def find_match1(img1, img2):

    sift = cv2.xfeatures2d.SIFT_create()
    im,x = [], [] 
    im.append(img1)
    im.append(img2)
    for j in range(2):
        tem_kp, tem_des = sift.detectAndCompute(im[j], None)
        tar_kp, tar_des = sift.detectAndCompute(im[1-j], None)
        # Nearest neighbour matching  
        model = NearestNeighbors(n_neighbors=2).fit(tar_des)
        dist, indices = model.kneighbors(tem_des)
        # Ratio culling
        u,v =[], []
        # For each kp in img1 if nearest neighbour distance ratio < NN_RATIO
        for i in range(len(tem_kp)):
            point1 = tem_kp[i].pt
            point2 = tar_kp[indices[i][0]].pt
            d1, d2 = dist[i]
            if (d1 / d2) <= NN_RATIO:
                u.append(point1)
                v.append(point2)
            
        u,v  = np.asarray(u), np.asarray(v)
        x.append(u)
        x.append(v)
        print('{} my ratio {}'.format(len(x[j+1]), NN_RATIO))
    #print(x)
    f_dict = {}
    x1_fo, x2_fo, x1_ba, x2_ba =x[0],x[1],x[2],x[3] 
    #print(x1_fo)
    x1_fo, x2_fo, x1_ba, x2_ba =np.asarray(x1_fo) ,np.asarray(x2_fo) ,np.asarray(x1_ba) ,np.asarray(x2_ba) 
    #print(x1_fo)
    for x1, x2 in zip(x1_fo,x2_fo):
        f_dict[tuple(x1)] = tuple(x2)
    b_dict = {}
    print(f_dict)
    print(b_dict)
    for x1, x2 in zip( x1_ba , x2_ba):
        b_dict[tuple(x2)] = tuple(x1)
    x1_f, x2_f = [], [] 
    for x1, x2 in zip( x1_fo , x2_fo ):
        try:
            if b_dict[f_dict[tuple(x1)]] == tuple(x1):
                x1_f.append(x1)
                x2_f.append(x2)
        except KeyError:
            pass
    x1_f, x2_f = np.asarray(x1_f), np.asarray(x2_f)       
    x1 , x2 =x1_f , x2_f
    print('{} my consistency check'.format(len(x1)))

    return x1, x2

def find_match_from_template_to_target(_template, _target):
    # SIFT features (descriptors) extraction
    sift = cv2.xfeatures2d.SIFT_create()
    template_kps, template_descriptors = sift.detectAndCompute(_template, None)
    target_kps, target_descriptors = sift.detectAndCompute(_target, None)
    # Nearest neighbour matching
    model = NearestNeighbors(n_neighbors=2).fit(target_descriptors)
    distances, indices = model.kneighbors(template_descriptors)
    # Ratio culling
    x1, x2, x1_all, x2_all = [], [], [], []
    # For each kp in img1 if nearest neighbour distance ratio < NN_RATIO
    for i in range(len(template_kps)):
        point1 = template_kps[i].pt
        point2 = target_kps[indices[i][0]].pt
        d1, d2 = distances[i]
        if (d1 / d2) <= NN_RATIO:
            x1.append(point1)
            x2.append(point2)
        x1_all.append(point1)
        x2_all.append(point2)

    x1, x2, x1_all, x2_all = np.asarray(x1), np.asarray(x2), np.asarray(x1_all), np.asarray(x2_all)

    print('{} SIFT feature matches'.format(len(x1_all)))
    # visualize_find_match(_target, _template, x1_all, x2_all)
    print('{} SIFT feature matches with filtering ratio {}'.format(len(x1), NN_RATIO))
    return x1, x2

def find_match(img1, img2):
    x1_forward, x2_forward = find_match_from_template_to_target(img1, img2)
    # visualize_find_match(img1, img2, x1_forward, x2_forward)
    x2_backward, x1_backward = find_match_from_template_to_target(img2, img1)
    # visualize_find_match(img1, img2, x1_backward, x2_backward)
    forward_dict = {}
    for x1, x2 in zip(x1_forward, x2_forward):
        forward_dict[tuple(x1)] = tuple(x2)
    backward_dict = {}
    for x1, x2 in zip(x1_backward, x2_backward):
        backward_dict[tuple(x2)] = tuple(x1)
    x1_final, x2_final = [], []
    for x1, x2 in zip(x1_forward, x2_forward):
        try:
            if backward_dict[forward_dict[tuple(x1)]] == tuple(x1):
                x1_final.append(x1)
                x2_final.append(x2)
        except KeyError:
            pass
    x1_final, x2_final = np.asarray(x1_final), np.asarray(x2_final)
    print('{} SIFT feature matches with bi-directional consistency check'.format(len(x1_final)))
    return x1_final, x2_final

img_left = cv2.imread('left.bmp', 1)
img_right = cv2.imread('right.bmp', 1)

#pts1, pts2 = find_match(img_left, img_right)
p1, p2 = find_match1(img_left, img_right)

#x=pts1.shape 
#print(sum(x))
print(p1)

cv2.waitKey(0)
cv2.destroyAlWindows()
