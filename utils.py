import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2
import math
import re
import torch
import collections


def bdb2d_iou(bdb1, bdb2):
    bdbc_min = np.maximum(bdb1[:2], bdb2[:2])
    bdbc_max = np.minimum(bdb1[2:], bdb2[2:])
    if (bdbc_max<=bdbc_min).any():
        return 0
    bdb1_area = (bdb1[2:]-bdb1[:2]).prod()
    bdb2_area = (bdb2[2:]-bdb2[:2]).prod()
    bdbc_area = (bdbc_max-bdbc_min).prod()
    return bdbc_area/(bdb1_area+bdb2_area-bdbc_area)

def get_bdb3d_corners(p1, p2):
    corners = np.zeros((8,3))
    corners[0, :] = (p1[0], p1[1], p1[2])
    corners[1, :] = (p1[0], p1[1], p2[2])
    corners[2, :] = (p1[0], p2[1], p2[2])
    corners[3, :] = (p1[0], p2[1], p1[2])
    corners[4, :] = (p2[0], p1[1], p1[2])
    corners[5, :] = (p2[0], p1[1], p2[2])
    corners[6, :] = (p2[0], p2[1], p2[2])
    corners[7, :] = (p2[0], p2[1], p1[2])
    return corners

def get_bdb3d_corners_from_basis_scale(basis3d, scale3d):
    edges3d = basis3d[:,:] * scale3d[:, None]
    corners = np.zeros((8,3))
    corners[0, :] = [0,0,0]
    corners[1, :] = edges3d[0,:]
    corners[2, :] = edges3d[0,:] + edges3d[1,:]
    corners[3, :] = edges3d[1,:]
    corners[4, :] = edges3d[2,:] + [0,0,0]
    corners[5, :] = edges3d[2,:] + edges3d[0,:]
    corners[6, :] = edges3d[2,:] + edges3d[0,:]+edges3d[1,:]
    corners[7, :] = edges3d[2,:] + edges3d[1,:]
    return corners-edges3d[:,:].sum(0)/2

def get_bdb3d_corners_from_centre_basis_scale(centre_3d, basis3d, scale3d):
    return centre_3d + get_bdb3d_corners_from_basis_scale(basis3d, scale3d)

def plt_line(ax, p1, p2, color='k'):
    ax.add_line(matplotlib.lines.Line2D([p1[0],p2[0]],[p1[1],p2[1]],linestyle='dashed',color=color))

def plt_cuboid2d(ax, points):
    # Points: 8,2
    color=np.random.rand(3)
    plt_line(ax, points[0,:], points[1,:], color=color)
    plt_line(ax, points[1,:], points[2,:], color=color)
    plt_line(ax, points[2,:], points[3,:], color=color)
    plt_line(ax, points[3,:], points[0,:], color=color)
    plt_line(ax, points[4,:], points[5,:], color=color)
    plt_line(ax, points[5,:], points[6,:], color=color)
    plt_line(ax, points[6,:], points[7,:], color=color)
    plt_line(ax, points[7,:], points[4,:], color=color)
    plt_line(ax, points[0,:], points[4,:], color=color)
    plt_line(ax, points[1,:], points[5,:], color=color)
    plt_line(ax, points[2,:], points[6,:], color=color)
    plt_line(ax, points[3,:], points[7,:], color=color)

def plt_cuboids2d(rgb, corners2d, output='show'):
    fig = plt.figure('3dbbox_proj')
    ax = plt.axes()
    ax.imshow(rgb)
    for corners in corners2d:
        plt_cuboid2d(ax, corners)
    if output == 'show':
        plt.show()
    elif output is None:
        pass
    else:
        plt.savefig(output)
    return fig

def plot_bbox2d(ax, bdb2d):
    # Points: 8,2
    color=np.random.rand(3)
    p0 = bdb2d[[0,1]]
    p1 = bdb2d[[0,3]]
    p2 = bdb2d[[2,3]]
    p3 = bdb2d[[2,1]]
    plt_line(ax, p0, p1, color=color)
    plt_line(ax, p1, p2, color=color)
    plt_line(ax, p2, p3, color=color)
    plt_line(ax, p3, p0, color=color)

def plot_bbox2ds(rgb, bdb2d_list, output='show'):
    fig = plt.figure('2dbbox')
    ax = plt.axes()
    ax.imshow(rgb)
    for bdb in bdb2d_list:
        plot_bbox2d(ax, bdb)
    if output == 'show':
        plt.show()
    elif output is None:
        pass
    else:
        plt.savefig(output)
    return fig

def read_matrix_file(fname):
    with open(fname) as f:
        content = f.readlines()
    content = [[float(f) for f in x.strip().split()] for x in content]
    return np.array(content)

def _is_char_digit(c):
    return 49 <= ord(c) <= 57
def _is_char_az(c):
    return 97 <= ord(c) <= 122
def _is_char_AZ(c):
    return 65 <= ord(c) <= 90
def _is_char_alphabet(c):
    return _is_char_az(c) or _is_char_AZ(c)
def _process_string(s):
    try:
        return int(s)
    except:
        try:
            return float(s)
        except:
            try: # Array of float
                return np.array([float(x.strip()) for x in s.split()])
            except:
                return s

def read_info_file(fname):
    # Processes text files where each lines is a variable assignemnt.
    # For example:
    # ```
    # length = 123
    # pi = 3.14
    # name = Shubham
    # list = 0 0.1 0.2 0.3
    # ```
    # Returns a dictionary:
    # {
    #   'length': 123,
    #   'pi': 3.14,
    #   'name': 'Shubham',
    #   'list': np.array([0.0, 0.1, 0.2, 0.3])
    # }
    with open(fname) as f:
        content = f.readlines()
    content = [ x.strip().split('=') for x in content]
    content = [[x.strip() for x in l] for l in content]
    keys = [l[0] for l in content]
    values = [_process_string(l[1]) for l in content]
    return dict(zip(keys, values))

def homo2eucl(X, first_axis=False):
    # Numpy and Torch friendly
    if first_axis:  # X: (4, ...)
        assert(X.shape[0] in [3,4])
        return X[ :-1, ... ]/X[ -1: , ... ]
    else:           # X: (... , 4)
        assert(X.shape[-1] in [3,4])
        return X[ ... , :-1]/X[ ... , -1: ]
homo2euclNumpy = homo2eucl
homo2euclTorch = homo2eucl

def eucl2homo(X, first_axis=False):
    # Numpy and Torch friendly
    if torch.is_tensor(X):
        return eucl2homoTorch(X,first_axis=first_axis)
    else:
        return eucl2homoNumpy(X,first_axis=first_axis)
def eucl2homoNumpy(X, first_axis=False):
    if first_axis:  # X: (3, ...)
        assert(X.shape[0] in [2,3])
        return np.concatenate((X,np.ones_like(X[ -1:, ... ])), axis=0)
    else:           # X: (... , 3)
        assert(X.shape[-1] in [2,3])
        return np.concatenate((X,np.ones_like(X[ ... , -1:])), axis=-1)
def eucl2homoTorch(X, first_axis=False):
    if first_axis:  # X: (3, ...)
        assert(X.shape[0] in [2,3])
        return torch.cat((X,torch.ones_like(X[ -1:, ... ])), dim=0)
    else:           # X: (... , 3)
        assert(X.shape[-1] in [2,3])
        return torch.cat((X,torch.ones_like(X[ ... , -1:])), dim=-1)

def transformND(X, M, first_axis=False):
    # Numpy and Torch friendly
    if first_axis:  # X: (3, ...)
        assert(X.shape[0] in [3,4])
        return homo2eucl(M.dot(eucl2homo(X, first_axis=True)), first_axis=True)
    else:           # X: (... , 3)
        assert(X.shape[-1] in [3,4])
        Mt = M.t() if torch.is_tensor(M) else M.T
        return homo2eucl(eucl2homo(X, first_axis=False).dot(Mt), first_axis=False)

def projectND(X, P, first_axis=False):
    # Numpy and Torch friendly
    if first_axis:  # X: (3, ...)
        assert(X.shape[0] in [3,4])
        return homo2eucl(P.dot(X), first_axis=True)
    else:           # X: (... , 3)
        assert(X.shape[-1] in [3,4])
        Pt = P.t() if torch.is_tensor(P) else P.T
        return homo2eucl(X.dot(Pt), first_axis=False)

# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R, atol=1e-4) :
    I = np.eye(3, dtype = R.dtype)
    return np.isclose(I, R.T @ R, atol=atol).all()
def rotationMatrixToEulerAngles(R) :
    assert(isRotationMatrix(R))
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    return np.array([x, y, z])
def eulerXToRotationMatrix(theta):
    R_x = np.array([[1,         0,               0                ],
                    [0,         math.cos(theta), -math.sin(theta) ],
                    [0,         math.sin(theta), math.cos(theta)  ]
                    ])
    return R_x
def eulerYToRotationMatrix(theta):
    R_y = np.array([[math.cos(theta),    0,      math.sin(theta)  ],
                    [0,                  1,      0                ],
                    [-math.sin(theta),   0,      math.cos(theta)  ]
                    ])
    return R_y
def eulerZToRotationMatrix(theta):
    R_z = np.array([[math.cos(theta),    -math.sin(theta),    0],
                    [math.sin(theta),    math.cos(theta),     0],
                    [0,                  0,                   1]
                    ])
    return R_z
def eulerAnglesToRotationMatrix(theta) :
    R_x = eulerXToRotationMatrix(theta[0])
    R_y = eulerYToRotationMatrix(theta[1])
    R_z = eulerZToRotationMatrix(theta[2])
    return R_z.dot(R_y.dot(R_x))
def axisAngleToRotationMatrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def cameraPoseToPitchRoll(cam2world, worldZ=np.array([0,0,1]), camAxisDepth = np.array([0,0,1]), camAxisWidth = np.array([1,0,0])):
    # Assume worldZ is upright vector in world coordinates
    # Assume camera uses right handed coordinate system with X along image width, Y along height
    # Angle between optical axes and world-Z gives pitch
    # Angle between width axes (after fixing pitch) and world-Z gives roll
    cam2world = cam2world[:3,:3]
    worldAxisDepth = cam2world.dot(camAxisDepth)
    pitch = math.pi/2 - math.acos(worldAxisDepth.dot(worldZ))

    pitch_align_rot_axis = np.cross(worldZ, worldAxisDepth)
    pitch_align_rot_angle = -pitch
    worldToWorldNewR = axisAngleToRotationMatrix(pitch_align_rot_axis, pitch_align_rot_angle)

    worldNewAxisWidth = worldToWorldNewR.dot(cam2world.dot(camAxisWidth))
    roll = math.pi/2 - math.acos(worldNewAxisWidth.dot(worldZ))

    return pitch, roll

def read_pgm(filename, byteorder='>', shift=None):
    """Return image data from a raw PGM file as numpy array.
    Format specification: http://netpbm.sourceforge.net/doc/pgm.html
    """
    with open(filename, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    pgm = np.frombuffer(buffer,
                            dtype='u1' if int(maxval) < 256 else byteorder+'u2',
                            count=int(width)*int(height),
                            offset=len(header)
                            ).reshape((int(height), int(width)))
    if shift is None:
        return pgm
    else:
        return pgm/shift

def read_file_as_list(fname):
    with open(fname, 'r') as f:
        ll = f.readlines()
    return [l.strip() for l in ll]

IMG_MEAN = [0.485, 0.456, 0.406]
IMG_DEV  = [0.229, 0.224, 0.225]
def normalize(img):
    if torch.is_tensor(img):
        assert(img.dim() in [3,4])
        if img.dim()==3:
            assert(img.shape[0] == 3)
            img = (img - torch.tensor(IMG_MEAN, dtype=img.dtype, device=img.device)[:,None,None])/torch.tensor(IMG_DEV, dtype=img.dtype, device=img.device)[:,None,None]
        elif img.dim()==4:
            assert(img.shape[1] == 3)
            img = (img - torch.tensor(IMG_MEAN, dtype=img.dtype, device=img.device)[None,:,None,None])/torch.tensor(IMG_DEV, dtype=img.dtype, device=img.device)[None,:,None,None]
    else:
        assert(img.ndim == 3)
        assert(img.shape[2] == 3)
        img = (img - np.array(IMG_MEAN)[None,None,:])/np.array(IMG_DEV)[None,None,:]
    return img

def unnormalize(img):
    if torch.is_tensor(img):
        assert(img.dim() in [3,4])
        if img.dim()==3:
            assert(img.shape[0] == 3)
            img = (img * torch.tensor(IMG_DEV, dtype=img.dtype, device=img.device)[:,None,None]) + torch.tensor(IMG_MEAN, dtype=img.dtype, device=img.device)[:,None,None]
        elif img.dim()==4:
            assert(img.shape[1] == 3)
            img = (img * torch.tensor(IMG_DEV, dtype=img.dtype, device=img.device)[None,:,None,None]) + torch.tensor(IMG_MEAN, dtype=img.dtype, device=img.device)[None,:,None,None]
    else:
        assert(img.ndim == 3)
        assert(img.shape[2] == 3)
        img = (img * np.array(IMG_DEV)[None,None,:]) + np.array(IMG_MEAN)[None,None,:]
    return img

def np2torch(img):
    assert(img.ndim in [2,3])
    img = torch.from_numpy(img)
    if img.dim()==3:
        assert(img.shape[2] in [1,3])
        img = img.permute((2,0,1))
    return img

def torch2np(img):
    assert(img.dim() in [2,3])
    img = img.numpy()
    if img.ndim==3:
        assert(img.shape[0] in [1,3])
        img = np.transpose(img, (1,2,0))
    return img

def np2torch_recursive(elem):
    if torch.is_tensor(elem):
        return elem
    elif type(elem).__module__ == 'numpy':
        if elem.size == 0:
            return torch.zeros(elem.shape).type(torch.DoubleTensor)
        elif elem.dtype.type in [np.unicode_, np.string_]:
            return elem
        else:
            return torch.from_numpy(elem)
    elif isinstance(elem, int):
        return torch.LongTensor([elem])
    elif isinstance(elem, float):
        return torch.DoubleTensor([elem])
    elif isinstance(elem, str):
        return elem
    elif isinstance(elem, collections.Mapping):
        return {key: np2torch_recursive(elem[key]) for key in elem}
    elif isinstance(elem, collections.Sequence):
        return [np2torch_recursive(samples) for samples in elem]
    else:
        return elem
