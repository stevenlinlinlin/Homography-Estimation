import sys
import numpy as np
import cv2 as cv
from utils import *


def get_sift_correspondences(img1, img2):
    '''
    Input:
        img1: numpy array of the first image
        img2: numpy array of the second image

    Return:
        points1: numpy array [N, 2], N is the number of correspondences
        points2: numpy array [N, 2], N is the number of correspondences
    '''
    #sift = cv.xfeatures2d.SIFT_create()# opencv-python and opencv-contrib-python version == 3.4.2.16 or enable nonfree
    sift = cv.SIFT_create()             # opencv-python==4.5.1.48
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    matcher = cv.BFMatcher()
    matches = matcher.knnMatch(des1, des2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    good_matches = sorted(good_matches, key=lambda x: x.distance)
    
    if sys.argv[2] == 'images/1-1.png':
        select_list = [1373, 165, 779, 100, 248, 312, 22, 223, 1368, 568, 283, 1418, 336, 402, 1367, 1365, 564, 801, 1131, 399]
    elif sys.argv[2] == 'images/1-2.png':
        select_list = [16, 20, 36, 43, 24, 28, 31, 34, 7, 8, 11, 12, 17, 18, 19, 22, 27, 33, 35, 48]
    good_matches = [good_matches[i] for i in select_list]

    
    points1 = np.array([kp1[m.queryIdx].pt for m in good_matches])
    points2 = np.array([kp2[m.trainIdx].pt for m in good_matches])
    
    img_draw_match = cv.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv.imshow('match', img_draw_match)
    cv.waitKey(0)
    return points1, points2

if __name__ == '__main__':
    img1 = cv.imread(sys.argv[1]) # 'images/1-0.png'
    img2 = cv.imread(sys.argv[2]) # 'images/1-1.png' or 'images/1-2.png'
    gt_correspondences = np.load(sys.argv[3]) # 'groundtruth_correspondences/correspondence_01.npy' or 'groundtruth_correspondences/correspondence_02.npy'
    method = sys.argv[4] # 'DLT': Direct Linear Transformer or 'NDLT': Normalized Direct Linear Transformer
    
    print('Method: ', method)
    print(f'Anchor: {sys.argv[1]}, Target: {sys.argv[2]}, Groundtruth: {sys.argv[3]}')
    points1, points2 = get_sift_correspondences(img1, img2)
    p_s = gt_correspondences[0]
    p_t = gt_correspondences[1]
    
    # sample 4, 8, 20 correspondences
    # samples = {}
    # if sys.argv[2] == 'images/1-1.png':
    #     select_list = [1373, 165, 779, 100, 248, 312, 22, 223, 1368, 568, 283, 1418, 336, 402, 1367, 1365, 564, 801, 1131, 399]
    #     samples = {4: np.array(select_list[:4]),
    #                8: np.array(select_list[:8]),
    #                20: np.array(select_list)}
        
    # elif sys.argv[2] == 'images/1-2.png':
    #     select_list = [16, 20, 36, 43, 24, 28, 31, 34, 7, 8, 11, 12, 17, 18, 19, 22, 27, 33, 35, 48]
    #     samples = {4: np.array(select_list[:4]),
    #                8: np.array(select_list[:8]),
    #                20: np.array(select_list)}
        
    
    for k in [4, 8, 20]:
        print('-----Top k =', k)
        # random_selected_rows = np.random.choice(points1.shape[0], 4, replace=False)
        # print('Randomly selected rows: ', random_selected_rows)
        if method == 'DLT':
            # print('Direct Linear Transformer-----------------')
            H = direct_linear_transformer(
                points1[:k], points2[:k])
            error = compute_reprojection_error(H, p_s, p_t)
            print('Reprojection error: ', error)
            
        elif method == 'NDLT':
            # print('Normalized Direct Linear Transformer-----------------')
            H = normalized_direct_linear_transformer(points1[:k], points2[:k])
            error = compute_reprojection_error(H, p_s, p_t)
            print('Reprojection error: ', error)
