# -*- coding: utf-8 -*-
"""
Created on Sa Jul 18 2015

@author: jmoosmann
"""

from __future__ import print_function, division, absolute_import
from future import standard_library

from builtins import super
from operator import add

# External
import numpy as np
# import matplotlib.pyplot as plt
# from scipy import ndimage
from pyastra import ODLProjector
from pyastra import Geometry
from utils import partial, DisplayIntermediates

# ODL
from odl import Rn, Operator, ProductSpace
# from RL.utility.testutils import Timer

import matplotlib

matplotlib.use("qt4agg")
# matplotlib.use("gtkagg")
standard_library.install_aliases()

# pylint: disable=invalid-name


class ODLChambollePock(object):
    """See docstring of class ChambollePock."""

    def __init__(self, geometry=Geometry(),
                 projections_vector=Rn(Geometry().proj_size).zero()):
        self.geom = geometry
        self.proj = projections_vector
        self.recon_space = Rn(geometry.vol_size)
        self.adj_scal_fac = 1
        self.forward_proj_scal = 1

    def adjoint_scaling_factor(self):
        """Compute scaling factor of adjoint projector. Consider A x = y,
        the adjoint A* of A is defined as:

             <A x, y>_D = <x, A* y>_I

         Assume A* = s B with B being the ASTRA backprojector, then:

             s = <A x, A x> / <B A x, x>

        Returns
        -------
        :rtype: float
        :returns: s
        """

        vol_rn = Rn(self.geom.vol_size)
        proj_rn = Rn(self.geom.proj_size)

        vol_rn_ones = vol_rn.element(1)
        proj_rn_ones = proj_rn.element(1)

        projector = ODLProjector(self.geom, vol_rn, proj_rn)

        proj = projector.forward(vol_rn_ones)
        vol = projector.backward(proj_rn_ones)

        # print vol.data.min(), vol.data.max()
        # print proj.data.min(), proj.data.max()

        self.adj_scal_fac = proj.inner(proj_rn_ones) / vol_rn_ones.inner(vol)
        # self.adj_scal_fac = proj.norm()**2 / vol_rn.inner(vol, vol_rn_ones)
        # return proj.norm()**2 / vol_rn._inner(vol, vol_rn_ones)

        projector.clear_astra_memory()

    def matrix_norm(self, iterations, vol_init=1.0,
                    tv_norm=False, return_volume=False,
                    intermediate_results=False):
        """The matrix norm || K ||_2  of 'K' defined here as largest
        singular value of 'K'. Employs the generic power method to obtain a
        scalar 's' which tends to || K ||_2 as the iterations N increase.

        To be implemented: optionally return volume 'x', such that it can be
        re-used as initializer to continue the iteration.

        Parameters
        ----------
        :type iterations: int
        :param iterations: Number of iterations of the generic power method.
        :type vol_init: float | ndarray (default 1.0)
        :param vol_init: in I, initial image to start with.
        :type intermediate_results: bool
        :param intermediate_results: Returns list of intermediate results
        instead of scalar.
        :type return_volume: bool
        :param return_volume: Return volume in order to resume iteration via
        passing it over as initial volume.

        Returns
        -------
        :rtype: float | numpy.ndarray, numpay.array (optional)
        :returns: s, vol
         s: Scalar of final iteration or numpy.ndarray containing all
         results during iteration.
         vol: Volume vector
        """

        geom = self.geom
        vol = self.recon_space.element(vol_init)
        proj = Rn(geom.proj_size).zero()
        projector = ODLProjector(geom, vol.space, proj.space)
        # print 'projector scaling factor', projector.scal_fac
        tmp = None

        if intermediate_results:
            s = np.zeros(iterations)
        else:
            s = 0

        # Power method loop
        for n in range(iterations):

            # step 4: x_{n+1} <- K^T K x_n
            if tv_norm:
                # K = (A, grad) instead of K = A
                # Compute: - div grad x_n
                # use sum over generator expression
                tmp = -reduce(add,
                              (partial(
                                  partial(vol.data.reshape(geom.vol_shape),
                                          dim, geom.voxel_size[dim]),
                                  dim, geom.voxel_size[dim]) for dim in
                               range(geom.vol_ndim)))

            # x_n <- A^T (A x_n)
            vol = projector.backward(projector.forward(vol))
            vol *= self.adj_scal_fac

            if tv_norm:
                # x_n <- x_n - div grad x_n
                # print 'n: {2}. vol: min = {0}, max = {1}'.format(
                #     vol.data.min(), vol.data.max(), n)
                # print 'n: {2}. tv: min = {0}, max = {1}'.format(tmp.min(),
                #                                            tmp.max(), n)
                vol.data[:] += tmp.ravel()

            # step 5:
            # x_n <- x_n/||x_n||_2
            vol /= vol.norm()

            # step 6:
            # s_n <-|| K x ||_2
            if intermediate_results:
                # proj <- A^T x_n
                proj = projector.forward(vol)
                s[n] = proj.norm()
                if tv_norm:
                    s[n] = np.sqrt(s[n] ** 2 +
                                   reduce(add,
                                          (np.linalg.norm(
                                              partial(vol.data.reshape(
                                                  geom.vol_shape), dim,
                                                  geom.voxel_size[dim])) ** 2
                                           for dim in range(geom.vol_ndim))))

        # step 6: || K x ||_2
        if not intermediate_results:
            proj = projector.forward(vol)
            s = proj.norm()
            if tv_norm:
                s = np.sqrt(s ** 2 + reduce(add,
                                            (np.linalg.norm(partial(
                                                vol.data.reshape(
                                                    geom.vol_shape), dim,
                                                geom.voxel_size[dim])) ** 2
                                             for dim in range(geom.vol_ndim))))

        # Clear ASTRA memory
        projector.clear_astra_memory()

        # Returns
        if not return_volume:
            return s
        else:
            return s, vol.data

    def least_squares(self, iterations=1, L=None, tau=None, sigma=None,
                      theta=None, non_negativiy_constraint=False,
                      tv_norm=False,
                      verbose=True):
        """Least-squares problem with optional TV-regularisation and/or
        non-negativity constraint.

        Parameters
        ----------
        :type iterations: int (default 1)
        :param iterations: Number of iterations the optimization should
        run for.
        :type L: float (defaul: None)
        :param L: Matrix norm of forward projector. If 'None' matrix_norm is
        called with 20 iterations.
        :type tau: float (default 1/L)
        :param tau:
        :type sigma: float (default 1/L)
        :param sigma:
        :type theta: float (default 1)
        :param theta:
        :type non_negativiy_constraint: bool (default False)
        :param non_negativiy_constraint: Add non-negativity constraint to
        optimization problem (via indicator function).
        :type tv_norm: bool | float (default False)
        :param tv_norm: Unless False, coincides with the numerical value of
        the parameter lambda for TV-Regularisation.
        :type verbose: bool (default False)
        :param verbose: Show intermediate reconstructions and
        convergence measures during iteration.

        Returns
        -------
        :rtype: odl.Vector, odl.Vector, numpy.ndarray, numpy.ndarray
        :returns: u, p, cpd, l2_du
         u: vector of reconstructed volume
         p: vector of dual projection variable
         cpd: condition primal-dual gap (convergence measure)
         l2_du: l2-norm of constraint-induced convergence measure
        """

        # step 1:
        if L is None:
            L = self.matrix_norm(20)
        if tau is None:
            tau = 1 / L
        if sigma is None:
            sigma = 1 / L
        if theta is None:
            theta = 1

        # print 'tau:', tau
        # print 'sigma:', sigma
        # print 'theta:', theta

        geom = self.geom
        g = self.proj  # domain: D

        # l2-norm of (volume update / tau)
        l2_du = np.zeros(iterations)
        # conditional primal-dual gap
        cpd = np.zeros(iterations)

        # step 2: initialize u and p with zeros
        u = self.recon_space.zero()  # domain: I
        p = g.space.zero()  # domain: D
        # q: spatial vector = list of ndarrays in I (not Rn vectors)
        if tv_norm:
            ndim = geom.vol_ndim
            # domain of q: V = [I, I, ...]
            q = [np.zeros(geom.vol_shape, dtype=u.data.dtype) for _ in range(
                ndim)]

        # step 3: ub <- u
        ub = u.copy()  # domain: I

        # initialize projector
        A = ODLProjector(geom, u.space, p.space)

        # visual output instance
        disp = DisplayIntermediates(verbose=verbose, vol=u.data.reshape(
            geom.vol_shape), cpd=cpd, l2_du=l2_du)

        # step 4: repeat
        for n in range(iterations):

            # step 5: p_{n+1} <- (p_n + sigma(A^T ub_n - g)) / (1 + sigma)
            if n >= 0:
                # with(Timer('proj:')):
                #     # p_tmp <- A ub
                #     p_tmp = A.forward(ub)
                #     # p_tmp <- p_tmp - g
                #     p_tmp -= g
                #     # p <- p + sigma * p_tmp
                #     p += sigma * p_tmp
                # p_n <- p_n + sigma(A ub -g )
                tmp = A.forward(ub)
                # print 'p:', p.data.shape, 'Au:', tmp.data.shape, 'g:', \
                #     g.data.shape
                p += sigma * (A.forward(ub) - g)
            else:
                p -= sigma * g
            # p <- p / (1 + sigma)
            p /= 1 + sigma

            # TV step 6: q_{n+1} <- lambda(q_n + sigma grad ub_n) /
            # max(lambda 1_I, |q_n + sigma grad ub_n|)
            if tv_norm:

                for dim in range(ndim):
                    # q_n <- q_n + sigma * grad ub_n
                    q[dim] += sigma * partial(ub.data.reshape(
                        self.geom.vol_shape), dim, geom.voxel_size[dim])

                # |q_n|: isotropic TV
                # use div_q to save memory, q = [qi] where qi are ndarrays
                div_q = np.sqrt(reduce(add, (qi ** 2 for qi in q)))

                # max(lambda 1_I, |q_n + sigma diff ub_n|)
                # print 'q_mag:', div_q.min(), div_q.max()
                div_q[div_q < tv_norm] = tv_norm

                # q_n <- lambda * q_n / |q_n|
                for dim in range(ndim):
                    q[dim] /= div_q
                    q[dim] *= tv_norm

                # div q_{n+1}
                div_q = reduce(add, (partial(qi, dim, geom.voxel_size[dim])
                                     for (dim, qi) in enumerate(q)))
                div_q *= tau

            # step 6: u_{n+1} <- u_{n} - tau * A^T p_{n+1}
            # TV step 7: u_{n+1} <- u_{n} - tau * A^T p_{n+1} + div q_{n+1}
            # ub_tmp <- A^T p
            ub_tmp = A.backward(p)
            ub_tmp *= tau
            ub_tmp *= self.adj_scal_fac
            # l2-norm per voxel of ub_tmp = A^T p
            l2_du[n:] = ub_tmp.norm()  # / u.data.size
            if tv_norm:
                l2_du[n:] += np.linalg.norm(div_q.ravel())  # / u.data.size
            # store current u_n temporarily in ub_n
            ub = -u.copy()
            # u <- u - tau ub_tmp
            u -= ub_tmp
            # TV: u <- u + tau div q
            if tv_norm:
                print('{0}: u - A^T p: min = {1}, max = {2}'.format(
                    n, u.data.min(), u.data.max()))
                print('{0}: div q: min = {1}, max = {2}'.format(
                    n, div_q.min(), div_q.max()))
                u.data[:] += div_q.ravel()

            # Positivity constraint
            if non_negativiy_constraint:
                u.data[u.data < 0] = 0
                # print '\nu:', u.data.min(), u.data.max()

            # conditional primal-dual gap for current u and p
            # 1/2||A u - g||_2^2 + 1/2||p||_2^2 + <p,g>_D
            # p_tmp <- A u
            # p_tmp = A.forward(u)
            # p_tmp -= g
            # cpd[n:] = (0.5 * p_tmp.norm() ** 2 +
            cpd[n:] = (0.5 * p.space.norm(A.forward(u) - g) ** 2 +
                       0.5 * p.norm() ** 2 +
                       p.inner(g))  # / p.data.size
            if tv_norm:
                cpd[n:] += tv_norm * np.linalg.norm(
                    reduce(add, (partial(u.data.reshape(geom.vol_shape),
                                         dim, geom.voxel_size[dim]) for dim
                                 in range(geom.vol_ndim))
                           ).ravel(), ord=1)  # / u.data.size

            # step 7 / TV step 8: ub_{n+1} <- u_{n+1} + theta(u_{n+1} - u_n)
            # ub <- ub + u_{n+1}, remember ub = -u_n
            ub += u
            # ub <- theta * ub
            ub *= theta
            # ub <- ub + u_{n+1}
            ub += u

            # visual output
            disp.update()

        A.clear_astra_memory()

        # Should avoid window freezing
        disp.show()

        return u, p, cpd, l2_du
