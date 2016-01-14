axis = self.axis

# matrix for rotation about the z-axis (z') in the reoriented
# coordinate system S'
rot_mat = np.matrix([
    [cos(angle), sin(angle), 0],
    [sin(angle), cos(angle), 0],
    [0, 0, 1]])

if not (axis[0] == 0 and axis[1] == 0):
    axis /= sqrt(np.sum(axis ** 2))  # unit vector
    za = acos(axis[2])
    # angle between x-axis lab and the projection of axis onto
    # azimutal lab plane
    theta = atan2(axis[1], axis[0])

    # change of basis: cartesian basis of S' to the lab system S
    # determined by axis property `axis`:
    # (e_x',e_y',e_z') = (e_theta, e_theta', axis)
    b = np.matrix([
        [cos(za) * cos(theta), -sin(theta), axis[0]],
        [cos(za) * sin(theta), cos(theta), axis[1]],
        [sin(za), 0, axis[2]]])

return rot_mat * b