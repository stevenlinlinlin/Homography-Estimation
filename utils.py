import numpy as np

def direct_linear_transformer(points1, points2):
    '''
    Input:
        points1: numpy array [N, 2], N is the number of correspondences
        points2: numpy array [N, 2], N is the number of correspondences
    Return:
        H: numpy array [3, 3], the homography
    '''
    # build A matrix
    for i in range(points1.shape[0]):
        A = np.array([[0, 0, 0, -points1[i][0], -points1[i][1], -1, points2[i][1]*points1[i][0], points2[i][1]*points1[i][1], points2[i][1]],
                      [points1[i][0], points1[i][1], 1, 0, 0, 0, -points2[i][0]*points1[i][0], -points2[i][0]*points1[i][1], -points2[i][0]]])
        if i == 0:
            A_matrix = A
        else:
            A_matrix = np.concatenate((A_matrix, A), axis=0)
    # SVD
    U, S, V = np.linalg.svd(A_matrix)
    # get H matrix
    H = (V[-1]/V[-1, -1]).reshape(3, 3)

    return H


def normalization(points):
    '''
    Input:
        points: numpy array [N, 2], N is the number of correspondences

    Return:
        normalized_points: numpy array [N, 2], N is the number of correspondences
        T : numpy array [3, 3], the normalization matrix
    '''
    mean = np.mean(points, axis=0)
    points_c = points - mean
    mean_dist = np.mean(np.sqrt(np.sum(points_c**2, axis=1)))
    scale = np.sqrt(2) / mean_dist
    T = np.array([[scale, 0, -scale*mean[0]],
                  [0, scale, -scale*mean[1]],
                  [0, 0, 1]])
    normalized_points = np.dot(T, np.concatenate(
        (points.T, np.ones((1, points.shape[0]))), axis=0))
    normalized_points = normalized_points[:2] / normalized_points[2]
    normalized_points = normalized_points.T
    return normalized_points, T


def normalized_direct_linear_transformer(points1, points2):
    '''
    Input:
        points1: numpy array [N, 2], N is the number of correspondences
        points2: numpy array [N, 2], N is the number of correspondences
    Return:
        H: numpy array [3, 3], the homography
    '''
    # Normalization
    normalized_points1, T1 = normalization(points1)  # points1
    normalized_points2, T2 = normalization(points2)  # points2
    # DLT
    H_hat = direct_linear_transformer(normalized_points1, normalized_points2)
    # Denormalization
    H = np.dot(np.dot(np.linalg.inv(T2), H_hat), T1)

    return H


def compute_reprojection_error(H, p_s, p_t):
    '''
    Input:
        H: numpy array [3, 3], the homography
        p_s: numpy array [N, 2], N is the number of correspondences
        p_t: numpy array [N, 2], N is the number of correspondences
    Return:
        error: float, mean of the reprojection error
    '''
    p_t_hat = np.dot(H, np.concatenate(
        (p_s.T, np.ones((1, p_s.shape[0]))), axis=0))
    error = np.mean(
        np.sqrt(np.sum((p_t_hat[:2]/p_t_hat[2] - p_t.T)**2, axis=0)))

    return error
