import cv2
import numpy as np
import scipy.io as sio
from numpy import dot as dot
from scipy.linalg import null_space
import matplotlib.pyplot as plt
import sys
from sklearn.neighbors import NearestNeighbors
from mpl_toolkits.mplot3d import Axes3D

RANSAC_N = 1000
SIFT_SIZE = 7


def find_match(img1, img2):
    sift = cv2.xfeatures2d.SIFT_create()
    im,x = [], [] 
    im.append(img1)
    im.append(img2)
    ##### do the matching first from 1st img to 2nd one, then the other side
    for j in range(2):  
        tem_kp, tem_des = sift.detectAndCompute(im[j], None)
        tar_kp, tar_des = sift.detectAndCompute(im[1-j], None)
       
        model = NearestNeighbors(n_neighbors=2).fit(tar_des)
        dist, indices = model.kneighbors(tem_des)
        u,v   =[], []
        uu,vv=[],[]
        for i in range(len(tem_kp)):
            point1 = tem_kp[i].pt
            point2 = tar_kp[indices[i][0]].pt
            d1, d2 = dist[i]
            if (d1 / d2) <= 0.705 :
                u.append(point1)
                v.append(point2)
            uu.append(point1)
            vv.append(point2)
        uu=uu[::8]
        vv=vv[::8]    
        u,v  = np.asarray(u), np.asarray(v)
        uu,vv  = np.asarray(uu), np.asarray(vv)
        x.append(u)
        x.append(v)
       
        visualize_find_match(im[j], im[1-j], uu, vv, j)
        visualize_find_match(im[j], im[1-j], u, v, j)
        print('the total features are :{},  and {} after  filtering with a Ratio {} '.format(len(tem_kp),len(u), 0.7))
       
    x1_fo, x2_fo, x1_ba, x2_ba =x[0],x[1],x[2],x[3]

    f_dict = {}
    for x1, x2 in zip(x1_fo,x2_fo):
        f_dict[tuple(x1)] = tuple(x2)

    b_dict = {}
    for x1, x2 in zip( x2_ba , x1_ba):
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
    print('{} SIFT feature matches with bi-directional check'.format(len(x1)))

    return x1, x2
def compute_F(pts1, pts2):
    assert pts1.shape == pts2.shape
    n, nn = pts1.shape
    indices = np.arange(n)
    min_loss = None
    best_F = None
    for nn in range(1500):
        np.random.shuffle(indices)
        first_eight_indices = indices[:8]
        # Compute tentative F using null space of 8 points matrix
        ps1,ps2= pts1[first_eight_indices], pts2[first_eight_indices]
        assert ps1.shape == ps2.shape == (8, 2)
        A = np.zeros((8, 9))
        for i, (u, v) in enumerate(zip(ps1, ps2)):
            # print(i, u, v)
            A[i, 0], A[i, 1], A[i, 2], A[i, 3],= u[0] * v[0],u[1] * v[0],v[0], u[0] * v[1]
            A[i, 4] , A[i, 5], A[i, 6], A[i, 7], A[i, 8]= u[1] * v[1], v[1],u[0],u[1],1
        F = null_space(A)
        # Take only the first solution to null space
        F = F[:, 0]
        F_tentative = F.reshape(3, 3)
        #F_tentative = compute_F_by_8_point_algo(pts1[first_eight_indices], pts2[first_eight_indices])
        # Do SVD cleanup

       # F_cleaned = do_svd_cleapup(F_tentative)
        u, d, vt = np.linalg.svd(F_tentative)
        d[2] = 0
        F_cleaned = np.dot(u * d, vt)

        loss = []
        for pt1, pt2 in zip(pts1, pts2):
            u1 = np.asarray([pt1[0], pt1[1], 1])
            v1 = np.asarray([pt2[0], pt2[1], 1])
            per_point_loss = np.dot(np.matmul(v1, F_cleaned), u1)
            # print(v, F_cleaned, u, per_point_loss)
            loss.append(per_point_loss)
        loss = np.asarray(loss)
        loss = np.sum(loss ** 2)

        # Compute loss
        #loss = compute_RANSAC_loss(pts1, pts2, F_cleaned)
        # print(loss)
        if min_loss is None or loss < min_loss:
            min_loss = loss
            best_F = F_cleaned
    print('Min loss = {} for RANSAC iterations = {}'.format(min_loss, RANSAC_N))
    #print('Best fundamental matrix = {}'.format(best_F))
    return best_F
def triangulation(P1, P2, pts1, pts2):
    pts3D = []
    for pt1, pt2 in zip(pts1, pts2):
        pt1_3d = list(pt1) + [1]
        pt2_3d = list(pt2) + [1]
        pt1_skew_symmteric =np.asarray([
            [0, -pt1_3d[2], pt1_3d[1]],
            [pt1_3d[2], 0, -pt1_3d[0]],
            [-pt1_3d[1], pt1_3d[0], 0], ])

        pt2_skew_symmteric =np.asarray([
            [0, -pt2_3d[2], pt2_3d[1]],
            [pt2_3d[2], 0, -pt2_3d[0]],
            [-pt2_3d[1], pt2_3d[0], 0], ])

        pt1_cross_P1 =np.dot(pt1_skew_symmteric, P1)
        pt2_cross_P2 =np.dot(pt2_skew_symmteric, P2)
        A = np.vstack((pt1_cross_P1[:2], pt2_cross_P2[:2]))
        X = null_space(A, rcond=1e-1)
        # Take the first null space entry
        X = X[:, 0]
        # Divide by w
        X = X / X[3]
        pts3D.append(X[:3])
    pts3D = np.asarray(pts3D)
    return pts3D
def disambiguate_pose(Rs, Cs, points_3D_sets):
    best_i = 0
    bestValid = 0
    for i, (r, c, points_3D_set) in enumerate(zip(Rs, Cs, points_3D_sets)):
        numValid = 0
        c = c.reshape(-1) 
        r3 = r[2, :]
        for x in points_3D_set:
            view_1 = (x - c)[2]
            view_2 = np.dot(x - c, r3)
            if view_1 > 0 and view_2 > 0:
                # Both cameras looks towards this point
                numValid += 1
        if numValid > bestValid:
            bestValid = numValid
            best_i = i
        print(numValid)
    return Rs[best_i], Cs[best_i], points_3D_sets[best_i]
def compute_rectification(K, R, C):
    C1 = np.zeros(3)
    R1 = np.identity(3)
    C2 = C.reshape(-1)
    R2 = R
    RrectX = C2 / np.linalg.norm(C2)
    R1Z = R1[2, :]
    RrectZ = R1Z - RrectX * np.dot(R1Z, RrectX)
    RrectZ = RrectZ / np.linalg.norm(RrectZ)
    RrectY = np.cross(RrectZ, RrectX)
    Rrect = np.asarray([
        RrectX,
        RrectY,
        RrectZ
    ])
    H1= np.dot(dot(K, Rrect),np.linalg.inv(K))
    H2= np.dot(dot(K, Rrect),np.dot(R2.T,np.linalg.inv(K)))
    return H1, H2

def dense_match(img1, img2):
    assert img1.shape == img2.shape
    sift = cv2.xfeatures2d.SIFT_create() 
    im=[]
    im.append(img1)
    im.append(img2)
    desc=[]
    for b in range(2):
        h, w = im[b].shape
        kp = []
        for i in range(h):
            for j in range(w):
                kp.append(cv2.KeyPoint(x=j, y=i, _size=7))  ##make pixels as key-points
        kps, des = sift.compute(im[b], kp)   ## compute descreptors
        des = np.asarray(des).reshape((h, w, 128))
        desc.append(des)

    dense1 = desc[0]
    dense2 = desc[1]
    disparity = np.ones(img1.shape) ##initializing disparity
    h, w = img1.shape

    for i in range(h):
        for j in range( w):
            if img1[i, j] == 0:
                continue    ##ignoring background
            d1_d2_dists = []
            d1 = dense1[i, j]  ## 1st point's descriptor
            for k in range(0, j + 1): ##slipping on a row
                d2 = dense2[i, k]  
                d1_d2_dists.append(np.linalg.norm(d1 - d2))
            disparity[i, j] = np.abs(np.argmin(d1_d2_dists) - j)
    return disparity

# PROVIDED functions
def compute_camera_pose(F, K):
    #E = K.T @ F @ K
    E = np.dot(np.dot(K.T , F) , K)
    R_1, R_2, t = cv2.decomposeEssentialMat(E)
    # 4 cases
    R1, t1 = R_1, t
    R2, t2 = R_1, -t
    R3, t3 = R_2, t
    R4, t4 = R_2, -t

    Rs = [R1, R2, R3, R4]
    ts = [t1, t2, t3, t4]
    Cs = []
    for i in range(4):
        #Cs.append(-Rs[i].T @ ts[i])
        Cs.append(np.dot(-Rs[i].T , ts[i]))

    return Rs, Cs
def visualize_img_pair(img1, img2):
    img = np.hstack((img1, img2))
    if img1.ndim == 3:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()
def visualize_find_match(img1, img2, pts1, pts2, c):
    assert pts1.shape == pts2.shape, 'x1 and x2 should have same shape!'
    img_h = img1.shape[0]
    scale_factor1 = img_h/img1.shape[0]
    scale_factor2 = img_h/img2.shape[0]
    img1_resized = cv2.resize(img1, None, fx=scale_factor1, fy=scale_factor1)
    img2_resized = cv2.resize(img2, None, fx=scale_factor2, fy=scale_factor2)
    pts1 = pts1 * scale_factor1
    pts2 = pts2 * scale_factor2
    pts2[:, 0] += img1_resized.shape[1]
    img = np.hstack((img1_resized, img2_resized))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if c==0:
        for i in range(pts1.shape[0]):
            plt.plot([pts1[i, 0], pts2[i, 0]], [pts1[i, 1], pts2[i, 1]], 'g.-', linewidth=0.5, markersize=5)
        plt.axis('off')
        plt.show()
    if c==1:
        for i in range(pts1.shape[0]):
            plt.plot([pts1[i, 0], pts2[i, 0]], [pts1[i, 1], pts2[i, 1]], 'r.-', linewidth=0.5, markersize=5)
        plt.axis('off')
        plt.show()    
    if c==2:
        for i in range(pts1.shape[0]):
            plt.plot([pts1[i, 0], pts2[i, 0]], [pts1[i, 1], pts2[i, 1]], 'b.-', linewidth=0.5, markersize=5)
        plt.axis('off')
        plt.show()
def visualize_epipolar_lines(F, pts1, pts2, img1, img2):
    assert pts1.shape == pts2.shape, 'x1 and x2 should have same shape!'
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    ax1.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    ax2.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))

    for i in range(pts1.shape[0]):
        x1, y1 = int(pts1[i][0] + 0.5), int(pts1[i][1] + 0.5)
        ax1.scatter(x1, y1, s=5)
        p1, p2 = find_epipolar_line_end_points(img2, F, (x1, y1))
        ax2.plot([p1[0], p2[0]], [p1[1], p2[1]], linewidth=0.5)

    for i in range(pts2.shape[0]):
        x2, y2 = int(pts2[i][0] + 0.5), int(pts2[i][1] + 0.5)
        ax2.scatter(x2, y2, s=5)
        p1, p2 = find_epipolar_line_end_points(img1, F.T, (x2, y2))
        ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], linewidth=0.5)

    ax1.axis('off')
    ax2.axis('off')
    plt.show()
def find_epipolar_line_end_points(img, F, p):
    img_width = img.shape[1]
    el = np.dot(F, np.array([p[0], p[1], 1]).reshape(3, 1))
    p1, p2 = (0, -el[2] / el[1]), (img.shape[1], (-img_width * el[0] - el[2]) / el[1])
    _, p1, p2 = cv2.clipLine((0, 0, img.shape[1], img.shape[0]), p1, p2)
    return p1, p2
def visualize_camera_poses(Rs, Cs):
    assert(len(Rs) == len(Cs) == 4)
    fig = plt.figure()
    R1, C1 = np.eye(3), np.zeros((3, 1))
    for i in range(4):
        R2, C2 = Rs[i], Cs[i]
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        draw_camera(ax, R1, C1)
        draw_camera(ax, R2, C2)
        set_axes_equal(ax)
        ax.set_xlabel('x axis')
        ax.set_ylabel('y axis')
        ax.set_zlabel('z axis')
        ax.view_init(azim=-90, elev=0)
    fig.tight_layout()
    plt.show()
def visualize_camera_poses_with_pts(Rs, Cs, pts3Ds):
    assert(len(Rs) == len(Cs) == 4)
    fig = plt.figure()
    R1, C1 = np.eye(3), np.zeros((3, 1))
    for i in range(4):
        R2, C2, pts3D = Rs[i], Cs[i], pts3Ds[i]
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        draw_camera(ax, R1, C1, 5)
        draw_camera(ax, R2, C2, 5)
        ax.plot(pts3D[:, 0], pts3D[:, 1], pts3D[:, 2], 'b.')
        set_axes_equal(ax)
        ax.set_xlabel('x axis')
        ax.set_ylabel('y axis')
        ax.set_zlabel('z axis')
        ax.view_init(azim=-90, elev=0)
    fig.tight_layout()
    plt.show()
def draw_camera(ax, R, C, scale=0.2):
    axis_end_points = C + scale * R.T  # (3, 3)
    #vertices = C + scale * R.T @ np.array([[1, 1, 1], [-1, 1, 1], [-1, -1, 1], [1, -1, 1]]).T  # (3, 4)
    vertices = C + scale * np.dot(R.T , np.array([[1, 1, 1], [-1, 1, 1], [-1, -1, 1], [1, -1, 1]]).T ) # (3, 4)
    vertices_ = np.hstack((vertices, vertices[:, :1]))  # (3, 5)

    # draw coordinate system of camera
    ax.plot([C[0], axis_end_points[0, 0]], [C[1], axis_end_points[1, 0]], [C[2], axis_end_points[2, 0]], 'r-')
    ax.plot([C[0], axis_end_points[0, 1]], [C[1], axis_end_points[1, 1]], [C[2], axis_end_points[2, 1]], 'g-')
    ax.plot([C[0], axis_end_points[0, 2]], [C[1], axis_end_points[1, 2]], [C[2], axis_end_points[2, 2]], 'b-')

    # draw square window and lines connecting it to camera center
    ax.plot(vertices_[0, :], vertices_[1, :], vertices_[2, :], 'k-')
    ax.plot([C[0], vertices[0, 0]], [C[1], vertices[1, 0]], [C[2], vertices[2, 0]], 'k-')
    ax.plot([C[0], vertices[0, 1]], [C[1], vertices[1, 1]], [C[2], vertices[2, 1]], 'k-')
    ax.plot([C[0], vertices[0, 2]], [C[1], vertices[1, 2]], [C[2], vertices[2, 2]], 'k-')
    ax.plot([C[0], vertices[0, 3]], [C[1], vertices[1, 3]], [C[2], vertices[2, 3]], 'k-')
def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range, x_middle = abs(x_limits[1] - x_limits[0]), np.mean(x_limits)
    y_range, y_middle = abs(y_limits[1] - y_limits[0]), np.mean(y_limits)
    z_range, z_middle = abs(z_limits[1] - z_limits[0]), np.mean(z_limits)

    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
def visualize_disparity_map(disparity):
    plt.imshow(disparity, cmap='jet')
    plt.show()


if __name__ == '__main__':
    # read in left and right images as RGB images
    img_left = cv2.imread('left.bmp', 1)
    img_right = cv2.imread('right.bmp', 1)
   # visualize_img_pair(img_left, img_right)

    # Step 1: find correspondences between image pair
    pts1, pts2 = find_match(img_left, img_right)
    visualize_find_match(img_left, img_right, pts1, pts2,2)

    # Step 2: compute fundamental matrix
    F = compute_F(pts1, pts2)
    visualize_epipolar_lines(F, pts1, pts2, img_left, img_right)

    # Step 3: computes four sets of camera poses
    K = np.array([[350, 0, 960/2], [0, 350, 540/2], [0, 0, 1]])
    
    Rs, Cs = compute_camera_pose(F, K)
    visualize_camera_poses(Rs, Cs)

    # Step 4: triangulation
    pts3Ds = []
    P1 = np.dot(K , np.hstack((np.eye(3), np.zeros((3, 1)))))
    for i in range(len(Rs)):
        P2 = np.dot(K , np.hstack((Rs[i], np.dot(-Rs[i],Cs[i]))))
        pts3D= triangulation(P1, P2, pts1, pts2)
        pts3Ds.append(pts3D)
    visualize_camera_poses_with_pts(Rs, Cs, pts3Ds)

    ### Step 5: disambiguate camera poses
    R, C, pts3D = disambiguate_pose(Rs, Cs, pts3Ds)
   # visualize_camera_poses_with_pts(R, C, pts3D) 

    # Step 6: rectification
    H1, H2 = compute_rectification(K, R, C)
    img_left_w = cv2.warpPerspective(img_left, H1, (img_left.shape[1], img_left.shape[0]))
    img_right_w = cv2.warpPerspective(img_right, H2, (img_right.shape[1], img_right.shape[0]))
    visualize_img_pair(img_left_w, img_right_w)

    # Step 7: generate disparity map
    img_left_w = cv2.resize(img_left_w, (int(img_left_w.shape[1] / 2), int(img_left_w.shape[0] / 2)))  # resize image for speed
    img_right_w = cv2.resize(img_right_w, (int(img_right_w.shape[1] / 2), int(img_right_w.shape[0] / 2)))
    img_left_w = cv2.cvtColor(img_left_w, cv2.COLOR_BGR2GRAY)  # convert to gray scale
    img_right_w = cv2.cvtColor(img_right_w, cv2.COLOR_BGR2GRAY)
    disparity = dense_match(img_left_w, img_right_w)
    visualize_disparity_map(disparity)
 

    # save to mat
    sio.savemat('stereo.mat', mdict={'pts1': pts1, 'pts2': pts2, 'F': F, 'pts3D': pts3D, 'H1': H1, 'H2': H2,
                                     'img_left_w': img_left_w, 'img_right_w': img_right_w, 'disparity': disparity})
    cv2.waitKey(0)
    cv2.destroyAllWindows()