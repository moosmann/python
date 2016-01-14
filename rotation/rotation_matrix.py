import numpy as np
from math import cos, sin, acos, atan2, sqrt


def rotation(theta, out=np.zeros((3, 3))):
    cx, cy, cz = np.cos(theta)
    sx, sy, sz = np.sin(theta)
    out.flat = (cx * cz - sx * cy * sz, cx * sz + sx * cy * cz, sx * sy,
                -sx * cz - cx * cy * sz, -sx * sz + cx * cy * cz,
                cx * sy, sy * sz, -sy * cz, cy)
    return out

theta = np.pi / 2 * np.array((1, 0, 0))

rm = rotation(theta)

# print(rm)

np.set_printoptions(precision=4, suppress=True)


def f(*vec):
    vec = np.array(vec)
    vecr = np.dot(rm, vec)
    print(vecr)

# print('\n rotation of cartesian basis vectors')
# f(1,0,0)
# f(0,1,0)
# f(0,0,1)

angle = 0 * np.pi/2
axis = np.array([0, 1, 0])
v = np.matrix([[1, 0, 0]]).transpose()

# matrix for rotation about the z-axis (z') in the reoriented
# coordinate system S'
rot_mat = np.matrix([
    [cos(angle), -sin(angle), 0],
    [sin(angle), cos(angle), 0],
    [0, 0, 1]])

axis /= sqrt(np.sum(axis ** 2))  # unit vector
za = acos(axis[2])
# angle between x-axis lab and the projection of axis onto
# azimutal lab plane
theta = atan2( axis[1], axis[0])

# change of basis: cartesian basis of S' to the lab system S
# determined by axis property `axis`:
# (e_x',e_y',e_z') = (e_theta, e_theta', axis)
b = np.matrix([
    [cos(za) * cos(theta), -sin(theta), axis[0]],
    [cos(za) * sin(theta), cos(theta), axis[1]],
    [sin(za), 0, axis[2]]])

r = rot_mat * b

print(b)
print('\n v   :', v.transpose())
print('\n vrot:', v.transpose())