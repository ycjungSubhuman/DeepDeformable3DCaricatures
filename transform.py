import numpy as np
import scipy
from math import sin, cos, atan2
from scipy.spatial.transform import Rotation

def align_3d_to_2d(pt3d, pt2d):
    """
    Get pose of 3D pointset based on its 2D projections
    
    Kemelmacher-Shlizerman & Seitz. Face Reconstruction in the Wild. ICCV. 2011.
    
    https://grail.cs.washington.edu/3dfaces/paper.pdf
    """
    q = pt2d.copy().T
    q_bar = np.mean(q, axis=1)
    p = (q.T - q_bar.T).T
    
    Q = pt3d.copy().T # Use mean face with features for pose optimization
    Q_bar = np.mean(Q, axis=1)
    P = (Q.T - Q_bar.T).T
    
    PPTinv = scipy.linalg.inv(P@P.T)
    A = p@P.T@PPTinv
    
    # translation
    t = q_bar - A@Q_bar
    
    row_third = np.cross(A[0].T, A[1].T)
    A_prime = np.vstack([A, row_third.T])
    
    # rotation
    u, s, vh = np.linalg.svd(A_prime)
    R = u @ vh
    # scale
    s = 0.5*(np.linalg.norm(A[0]) + np.linalg.norm(A[1]))
    t = np.array([t[0], t[1], 0.0])

    return R, s, t

def ralign(X,Y):
    """
    Rigid alignment between two pointsets using Umeyama algorithm

    X           (n x m) points (i.e., m=3 or 2)
    Y           (n x m) points
    """

    X = X.T
    Y = Y.T

    m, n = X.shape

    mx = X.mean(1)
    my = Y.mean(1)
    Xc =  X - np.tile(mx, (n, 1)).T
    Yc =  Y - np.tile(my, (n, 1)).T

    sx = np.mean(np.sum(Xc*Xc, 0))

    Sxy = np.dot(Yc, Xc.T) / n

    U,D,V = np.linalg.svd(Sxy,full_matrices=True,compute_uv=True)
    V=V.T.copy()

    S = np.eye(m)

    R = np.dot( np.dot(U, S ), V.T)

    s = np.trace(np.dot(np.diag(D), S)) / sx
    t = my - s * np.dot(R, mx)

    return R,s,t


def rigid_registeration(src_array, tar_array, index = None):
	'''
	apply rigid registeration between 2 meshes.
	'''
	if index is not None:
		src_array = src_array[index]
		tar_array = tar_array[index]
	M = np.matmul((src_array - np.mean(src_array, axis=0)).T, (tar_array - np.mean(tar_array, axis=0)))
	u,s,v= np.linalg.svd(M)
	# print(np.dot(np.dot(u,np.diag(s)), v))
	sig = np.ones(s.shape)
	sig[-1] = np.linalg.det(np.dot(u,v))
	R = np.matmul(np.matmul(v.T, np.diag(sig)), u.T)
	t = np.mean(tar_array, axis=0) - np.matmul(R,np.mean(src_array, axis=0))
	return R, t


def apply_trans(X, params):
    return params['s'] * X @ params['R'].T + params['t']

def rot2angle(R):
    """
    Convert 2x2 rotation matrix to angle (radians)
    """
    return atan2(R[1, 0], R[0, 0])


class Affine2D:
    """
    2D affine transformation
    (similarity)
    """
    LENGTH=1+1+2

    def __init__(self, arr):
        # scale
        self.s = arr[0]
        # rotation
        self.theta = arr[1]
        # translation
        self.tx = arr[2]
        self.ty = arr[3]

    @staticmethod
    def identity():
        return Affine2D([1.0, 0.0, 0.0, 0.0])

    @staticmethod
    def optimize(X, Y):
        """
        X           (N x 2) source points
        Y           (N x 2) target points

        Returns)
        A transformation applied to X.
        """
        R, s, t = ralign(X, Y)
        theta = rot2angle(R)

        return Affine2D([s, theta, t[0], t[1]])

    def as_matrix(self):
        """
        Return 3X3 matrix for a 2D point in homogeneous coordinate
        """
        return np.array([
            [self.s*cos(self.theta), -self.s*sin(self.theta), self.tx],
            [self.s*sin(self.theta), self.s*cos(self.theta), self.ty],
            [0.0, 0.0, 1.0],
        ])

    def as_array(self):
        return np.array([self.s, self.theta, self.tx, self.ty])

    def apply(self, pts):
        """
        pts             input 2D points (N x 2)
        
        Returns)
        Transformed points (N x 2)
        """
        M = self.as_matrix()
        R = M[:2, :2]
        return (R @ pts.T).T + M[:2, 2] 
    
    def __repr__(self):
        return 'Affine2D: (s={}, theta={} rad, tx={}, ty={})'.format(self.s, self.theta, self.tx, self.ty)


class Affine3D:
    """
    2D affine transformation
    (similarity)
    """
    LENGTH=1+3+3

    def __init__(self, arr):
        # scale
        self.s = arr[0]
        # rotation
        self.angles = arr[1:4]
        # translation
        self.t = arr[4:7]

    @staticmethod
    def identity():
        return Affine3D([
            1.0, 
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0])

    @staticmethod
    def optimize(X, Y):
        """
        X           (N x 3) source points
        Y           (N x 3) target points

        Returns)
        A transformation applied to X.
        """
        R, s, t = ralign(X, Y)
        angles = Rotation.from_matrix(R).as_euler('XYZ')

        return Affine3D(np.concatenate([[s], angles, t]))

    @staticmethod
    def optimize_2d(X, Y):
        """
        X           (N x 3) source points
        Y           (N x 2) target points

        Returns)
        A transformation applied to X.
        """
        assert X.shape[1] == 3 and Y.shape[1] == 2
        R, s, t = align_3d_to_2d(X, Y)
        angles = Rotation.from_matrix(R).as_euler('XYZ')

        return Affine3D(np.concatenate([[s], angles, t]))

    def as_matrix(self):
        """
        Return 4X4 matrix for a 3D point in homogeneous coordinate
        """
        R = self.s * Rotation.from_euler('XYZ', self.angles).as_matrix()
        result = np.eye(4)
        result[:3,:3] = R
        result[:3,3] = self.t

        return result

    def as_array(self):
        return np.concatenate([[self.s], self.angles, self.t])

    def apply(self, pts):
        """
        pts             input 3D points (N x 3)
        
        Returns)
        Transformed points (N x 3)
        """
        M = self.as_matrix()
        R = M[:3, :3]
        return (R @ pts.T).T + M[:3, 3] 
    
    def __repr__(self):
        return 'Affine3D: (s={}, angles={} rad, t={})'.format(self.s, self.angles, self.t)
