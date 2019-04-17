import numpy as np
import trimesh
import utils
import matplotlib.pyplot as plt
import math

def area(x1, y1, x2, y2, x3, y3):
    return np.abs((x1 * (y2 - y3) + x2 * (y3 - y1)
                + x3 * (y1 - y2)) / 2.0)

# A function to check whether point P(x, y)
# lies inside the triangle formed by
# A(x1, y1), B(x2, y2) and C(x3, y3)
def isInside(x1, y1, x2, y2, x3, y3, x, y):
    A = area(x1, y1, x2, y2, x3, y3)
    A1 = area(x, y, x2, y2, x3, y3)
    A1 = A1 + area(x1, y1, x, y, x3, y3)
    A1 = A1 + area(x1, y1, x2, y2, x, y)
    return np.isclose(A, A1, rtol=1e-3)

def lies_in_triangles(pixel, triangles2d):
    # pixel: (..., img_w, img_h)
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
    return isInside(x1, y1, x2, y2, x3, y3, x, y).any(axis=-1)

def render_scene(vertices, faces, img_h, img_w, R, t):
    # Apply transformation R, t to vertices
    # Assume camera pose is np.eye(4)

    N = vertices.shape[0]
    M = faces.shape[0]

    triangles3d = vertices[faces.reshape(-1), :].reshape(M,3,3)
    camera_intr = np.array([[img_w  , 0 , img_w/2],
                            [0  , img_h , img_h/2],
                            [0  , 0 , 1]])

    transform = np.eye(4)
    transform[:3,:3] = R
    transform[:3,3] = t
    triangles3d = utils.transformND(triangles3d, transform)
    triangles2d = utils.projectND(triangles3d, camera_intr)
    triangles2d = triangles2d.astype(np.float32)

    # Rester 2d triangles
    image = np.zeros((img_h, img_w))

    # Plot vertices
    # plt.imshow(image)
    # plt.scatter(triangles2d.reshape(-1,2)[:,0], triangles2d.reshape(-1,2)[:,1])
    # plt.show()

    # for i in range(img_h):
    #     print(i)
    #     for j in range(img_w):
    #         pixel_pos = (j,i)
    #         if lies_in_triangles(pixel_pos, triangles2d):
    #             image[i,j] = 1
    ii, jj = np.meshgrid(np.arange(img_h), np.arange(img_w), indexing='ij', copy=False)
    pixel_pos = np.stack((jj,ii), axis=-1)
    image[...] = lies_in_triangles(pixel_pos, triangles2d)
    return image

if __name__ == "__main__":
    # scene_file = '3dmodels/guitar/models/model_normalized.obj'
    # scene_file = '3dmodels/rectangle.obj'
    scene_file = '3dmodels/triangle.obj'
    scene = trimesh.load(scene_file)
    # scene.show()

    vertices = scene.vertices
    faces = scene.faces
    img_w = 100
    img_h = 100
    print(f'{vertices.shape[0]} vertices, {faces.shape[0]} faces')

    for i in range(1000):
        print(f'iter {i}')
        ### Semantic space: Rotation, translation
        theta = (np.random.rand(3)-0.5)*2*math.pi/6
        trans = (np.random.rand(3)-0.5)*2*0.5
        R = utils.eulerAnglesToRotationMatrix(theta)
        t = trans + [0,0,3]
        print(f'theta: {theta*180/math.pi}')
        print(f't:     {t}')
        image = render_scene(vertices, faces, img_h, img_w, R, t)

        np.save(f'images/triangle/{i}.npy', {
            'scene_file':scene_file,
            'theta':theta,
            't':t,
            'image':image
        })
        # image = np.where(image==1, 0.5, 1)
        # plt.imshow(np.repeat(image[:,:,None], 3, axis=2))
        # plt.show()

