import numpy as np
import trimesh
import utils
import matplotlib.pyplot as plt
import math
import numbers
from raster import render_scene as render_scene_concrete

class Box:
    # Box Domain representing interval [lb, ub]
    def __init__(self, lb, ub=None):
        ub = ub if ub else lb
        assert(isinstance(lb, numbers.Number))
        assert(isinstance(ub, numbers.Number))
        self.lb = min(lb, ub)
        self.ub = max(lb, ub)
    def __str__(self):
        return f'({self.lb:8.4f}, {self.ub:8.4f})'
    def __repr__(self):
        return f'({self.lb:8.4f}, {self.ub:8.4f})'
    def __bool__(self):
        # TODO: Fix
        return (self==0)==False
    def __lt__(self,other):
        other = Box.toBox(other)
        return self.ub < other.lb
    def __le__(self,other):
        other = Box.toBox(other)
        return self.ub <= other.lb
    def __gt__(self,other):
        other = Box.toBox(other)
        return self.lb > other.ub
    def __ge__(self,other):
        other = Box.toBox(other)
        return self.lb >= other.ub
    def __eq__(self,other):
        other = Box.toBox(other)
        return (self.lb == other.lb) and (self.ub == other.ub)
    def __ne__(self,other):
        other = Box.toBox(other)
        return (self<other) or (self>other)
    def __isfinite__(self):
        return np.isfinite([self.lb, self.ub]).all()
    def __add__(self, other):
        other = Box.toBox(other)
        return Box(self.lb + other.lb, self.ub + other.ub)
    def __sub__(self, other):
        other = Box.toBox(other)
        return Box(self.lb - other.lb, self.ub - other.ub)
    def __rsub__(self, other):
        other = Box.toBox(other)
        return Box(other.lb - self.lb, other.ub - self.ub)
    def __mul__(self, other):
        other = Box.toBox(other)
        v1 = self.lb * other.lb
        v2 = self.lb * other.ub
        v3 = self.ub * other.lb
        v4 = self.ub * other.ub
        lb = min(v1,v2,v3,v4)
        ub = max(v1,v2,v3,v4)
        return Box(lb, ub)
    def __truediv__(self, other):
        other = Box.toBox(other)
        if other.lb <= 0 <= other.ub:
            raise NotImplementedError
        else:
            v1 = self.lb / other.lb
            v2 = self.lb / other.ub
            v3 = self.ub / other.lb
            v4 = self.ub / other.ub
            lb = min(v1,v2,v3,v4)
            ub = max(v1,v2,v3,v4)
            return Box(lb, ub)
    def __radd__(self, other):
        return self + other
    def __rmul__(self, other):
        return self * other
    def __abs__(self):
        ub = max(abs(self.ub), abs(self.lb))
        lb = min(abs(self.ub), abs(self.lb))
        if self.lb <= 0 <= self.ub:
            return Box(0,ub)
        else:
            return Box(lb,ub)

    @staticmethod
    def isclose(A, B, rtol=1e-05, atol=1e-08):
        # Returns Box(1,1) if all possible A,B are close
        # Returns Box(0,0) if all possible A,B are far
        # Returns Box(0,1) otherwise
        diff = abs(A-B)
        tol = atol + rtol*abs(B)
        return np.where(diff <= tol,
                            Box(1,1),
                        np.where(diff >= tol,
                            Box(0,0), Box(0,1)))

    @staticmethod
    def toBox(x):
        if isinstance(x, Box):
            return x
        else:
            return Box(x)

def area(x1, y1, x2, y2, x3, y3):
    return np.abs((x1 * (y2 - y3) + x2 * (y3 - y1)
                + x3 * (y1 - y2)) / 2.0)

# A function to check whether point P(x, y)
# lies inside the triangle formed by
# A(x1, y1), B(x2, y2) and C(x3, y3)
def isInside(x1, y1, x2, y2, x3, y3, x, y):
    # TODO: Use better method that is abstraction-friendly
    A = area(x1, y1, x2, y2, x3, y3)
    A1 = area(x, y, x2, y2, x3, y3)
    A1 = A1 + area(x1, y1, x, y, x3, y3)
    A1 = A1 + area(x1, y1, x2, y2, x, y)
    return Box.isclose(A, A1, rtol=1e-3)

def lies_in_triangles(pixel, triangles2d):
    # pixel: (..., j < img_w, i < img_h)
    assert(pixel.ndim >= 2)
    num_d = pixel.ndim - 1
    num_t = triangles2d.shape[0]
    # x = np.full((num_t,), pixel[0])
    # y = np.full((num_t,), pixel[1])
    x = pixel[ ... , None, 0]
    y = pixel[ ... , None, 1]
    triangles2d = triangles2d.reshape((1,)*num_d + triangles2d.shape)
    x1 = triangles2d[ ... , 0,0]
    y1 = triangles2d[ ... , 0,1]
    x2 = triangles2d[ ... , 1,0]
    y2 = triangles2d[ ... , 1,1]
    x3 = triangles2d[ ... , 2,0]
    y3 = triangles2d[ ... , 2,1]
    xxx = isInside(x1, y1, x2, y2, x3, y3, x, y)
    return xxx.any(axis=-1) # TODO: Implement union

def render_scene_abstract(vertices, faces, img_h, img_w, R, t):
    # Apply transformation R, t to vertices
    # Assume camera pose is np.eye(4)
    vertices = np.vectorize(Box)(vertices)
    vertices = vertices.dot(R.T) + t[None,:]


    N = vertices.shape[0]
    M = faces.shape[0]

    triangles3d = vertices[faces.reshape(-1), :].reshape(M,3,3)
    camera_intr = np.array([[img_w  , 0 , img_w/2],
                            [0  , img_h , img_h/2],
                            [0  , 0 , 1]])


    vertices2d = utils.projectND(vertices, camera_intr)
    triangles2d = utils.projectND(triangles3d, camera_intr)

    ii, jj = np.meshgrid(np.arange(img_h), np.arange(img_w), indexing='ij', copy=False)
    pixel_pos = np.stack((jj,ii), axis=-1)

    mv = vertices2d.sum(axis=0)/3
    print('mv',mv)

    image = lies_in_triangles(pixel_pos, triangles2d)
    return image

if __name__ == "__main__":
    # scene_file = '3dmodels/guitar/models/model_normalized.obj'
    # scene_file = '3dmodels/rectangle.obj'
    scene_file = '3dmodels/tetrahedron.obj'
    # scene_file = '3dmodels/triangle.obj'
    scene = trimesh.load(scene_file)
    scene.show()

    vertices = scene.vertices
    faces = scene.faces
    img_w = 100
    img_h = 100
    print(f'{vertices.shape[0]} vertices, {faces.shape[0]} faces')

    theta = (np.random.rand(3)-0.5)*2*math.pi/6
    trans = (np.random.rand(3)-0.5)*2*0.5
    for i in range(1000):
        print(f'iter {i}')
        ### Semantic space: Rotation, translation
        R = utils.eulerAnglesToRotationMatrix(theta)
        t = trans + [0,0,3]
        print(f'theta: {theta*180/math.pi}')
        print(f't:     {t}')

        ### Render
        image_concrete = render_scene_concrete(vertices, faces, img_h, img_w, R, t)
        R = np.vectorize(Box)(R)
        t = np.vectorize(Box)(t)
        delta_t = Box(0,0.01)
        t = t + delta_t
        print(f'delta_t: {delta_t}')

        image = render_scene_abstract(vertices, faces, img_h, img_w, R, t)

        ### Visualize
        image_lb = np.vectorize(lambda b:b.lb)(image)
        image_ub = np.vectorize(lambda b:b.ub)(image)
        image = (image_lb + image_ub)/2

        plt.figure('abstract')
        plt.imshow(image)
        plt.colorbar()

        plt.figure('concrete')
        plt.imshow(image_concrete)
        plt.colorbar()

        plt.show()
