# -*- coding: utf-8 -*-
"""
Created on Sa Jul 18 2015

@author: jmoosmann
"""

from __future__ import division
import numpy as np
from pyastra import Projector
from operator import add
from scipy.optimize import newton as scipy_newton
from utils import partial, DisplayIntermediates
# from RL.utility.testutils import Timer
import matplotlib
matplotlib.use("qt4agg")
# matplotlib.use("gtkagg")


# pylint: disable=invalid-name

class ChanGolubMullet(object):
    """Primal-dual method for total variation (TV) based image reconstruction
    for the problem K u = z according Chan, Golub, and Mulet.

    Consider the following unconstrained optimization problem employing
    Tikhonov regularization with the action functional conisiting of a data
    fidelity term and the TV-norm as regularization functional

        S[u(x)] = /int_omega dx f(x, u(x), grad u(x))
                = /int_omega dx ( a|grad u| + 1/2 (K u -z)
                = a || |grad u| ||_L1 + 1/2|| K u  - z ||_L2^2 ,

    where u is an image (volume) defined the open and bounded domain
    :math:'omega in R^n, grad u denotes the n-dimensional gradient,
    |grad u| = sqrt{u_{ x}^2}, K a linear operator, and ||.||_Lp the p-norm.

    The update equations of the primal-dual variables (u,w) read

     u_{n+1} = u_{n} + s_p * du_{n} ,
     w_{n+1} = w_{n} + s_d * dw_{n} ,

    where s_p and s_d denote the step length for u and w, respectively,
    which are chosen to be unity or computed via line search on u and w,
    respectively.

    Parameters
    ----------

    projections : numpy.ndarray
        Numpy array holding the projection data.

    projector : instance of class 'Projector'
        Projector instance representing the operator K and its adjoint K as
    defined in the class Projector in pyastra.

    """

    def __init__(self, projections=np.array([]), projector=Projector()):
        self.z = projections
        self.K = projector
        self.a = 1
        self.u = np.zeros(projector.num_voxel)
        self.w = [self.u for _ in range(self.u.ndim)]
        self.b = 1

    @property
    def g(self):
        """Returns the left-hand side of the first equation of the system of
        non-linear PDE for the primal/dual variables (w,u):

        g(w,u) = - a div w + K* (Ku - z) ,

        where 'u' is the real image (volume), 'z' the observed image (
        projections), 'w' the variable dual to 'u', 'a' a coefficient
        balancing the trade-off between data fidelity '|| K u - z||^2_L2'
        and regularization 'TV(u) = || |grad u| ||_L1'.

         Returns
         -------
         numpy.array
             Array of the same dimension as the reconstruction volume 'u'.
         """

        ushape = self.u.shape
        assert self.K.num_voxel == ushape, "Volume data dimension mismatch"

        # K u
        self.K.set_volume_data(self.u)
        self.K.forward()

        assert self.z.shape == self.K.projection_shape, \
            "Projection data mismatch"

        # K* (K u - z)
        self.K.set_projection_data(self.K.projection_data - self.z)
        self.K.backward()

        # Create slice object in order to fix dimension mismatch after
        # taking derivative s = [slice(0, nn - 1) for nn in ushape]

        # In order to compute the divergence of w, a generator comprehension
        #  is used to create an iterable object containing the 1D
        # derivatives of 'w'. return - self.a * reduce(add, (np.diff(wn, 1,
        # n)[s] for (n, wn) in enumerate(self.w))) + self.K.volume_data[s]
        return - self.a * reduce(add, (partial(wn, n) for (n, wn) in
                                       enumerate(self.w))) + self.K.volume_data

    @property
    def f(self):
        """Returns the left-hand side of the second equation of the system of
        non-linear PDE for the primal/dual variables (w,u):
        """
        # gradu = (np.diff((self.u, 1, n - 1)) for n in range(self.u.ndim))
        gradu = np.gradient(self.u)
        agradu = np.sqrt(sum(u ** 2 for u in gradu) + self.b)
        # agradu += self.b
        # agradu = np.sqrt(agradu)
        return (w * agradu - u for (u, w) in zip(self.w, gradu))

    def func_du(self, du):
        """Resulting functional for update 'du' (after eliminating 'dw' from
        the (u,w)-system and solving for 'du'). Instead of the standard
        finite difference discretization of the linear opearator '\bar{C}'
        involving the primal/ dual variables (u,w), the symmetrized matrix
        '\tilde{C} = 1/2(\bar{C}+\bar{C)^T)' is used.

        Parameters
        ----------
        du : numpy.ndarray
            Array containing the update 'du' which is of the same shape as 'u'.

        Returns
        -------
        Csym du - g(u,w) : numpy.ndarray
        """

        # K du
        self.K.set_volume_data(du)
        self.K.forward()

        # K* K du
        self.K.backward()
        grad_u = np.gradient(self.u)
        grad_du = np.gradient(du)
        abs_grad_u = np.sqrt(sum(u ** 2 for u in grad_u) + self.b)

        # gen_comp = (
        #     gdu - 0.5 / abs_grad_u * (w * (gu * gdu) + gu * (w * gdu)) for
        #     (w, gu, gdu) in zip(self.w, grad_u, grad_du))

        gen_comp = [
            gdu - 0.5 / abs_grad_u * (w * (gu * gdu) + gu * (w * gdu)) for
            (w, gu, gdu) in zip(self.w, grad_u, grad_du)]

        return - self.a * abs_grad_u * gen_comp + self.K.volume_data

    def newton(self, du0):
        """Apply approximate Newton method to solve the equation for the update 'du'

        :param du0:
        :return:
        """
        return scipy_newton(self.func_du, du0)


class ChambollePock(object):
    """Chambolle-Pock (PC) algorithm which is a primal-dual method to solve
    an optimization problem simultaneously with its dual, see Chambolle and
    Pock, A first order primal-dual algorithm for convex problems with
    applications to imaging, J. Math. Imag. Vis. 40, 1-26 (2011). A robust,
    non-heuristic convergence check is provided using the duality gap to
    assess algorithmic convergence.

    The implementation of PC follows the paper of Sidky, Jorgensen, and Pan,
    Convex optimization problem prototyping for image reconstruction in
    computed tomography with the Chambolle–Pock algorithm, Phys. Med. Biol.
    57, 3065-3091 (2012). Equation numbers in this class' doscstrings are
    identical to the ones in the aforementioned paper.


    Consider the general form of a primal minimization:

        min_x { F(K x) + G(x) },    (1)

    and the dual maximization:

        max_y { -F^*(y) - G^*(-K^T y) },    (2)

    where x,y are finite-dimensional vectors in spaces X,Y; K is a linear
    transform K:X->Y; G,F are convex, possibly non-smooth functions
    F:Y->R^+, and G:X->R^+; ^* in the dual maximization problem refers to
    convex conjugation defined below; ^T denotes matrix transposition. Note
    K needs not to be square, with X,Y having different dimensions.

    The convex conjugate of a convex function H of a vector z in Z is
    computed by the Legendre transform (Rockafellar et al. 1970):

        H^*(z) = max_z' {<z,z'>_Z - H(z') }.     (3)

    The original function is recovered by another conjugation:

        H^(z') = max_z' {<z,z'>_Z - H(z') }.    (4)

    <.,.>_Z denotes the inner product in the vector space Z.

    Formally, the primal and dual problems are connected in a generic saddle
    point optimization problem:

        min_x max_y {  <K x,y>_Y + G(x) - F^*(y) }.    (5)

    The primal minimization (1) is derived by performing the maximization
    over y in equation (5) using equation (4) with K x associated with y'.
    The dual maximization (2) is derived by performing the minimization over
    x in equation (5) using equation (3) with and the identity <K x,y> = <x,
    K^T y>.

    The duality gap is defined as the difference between the primal and the
    dual objective for estimates of x and y of the primal minimization and
    dual maximization. For intermediate estimates the primal objective is
    greater than the dual objective, and convergence is achieved when the
    gap closes.


    Consider the linear, discrete-to-discrete system model:

        A u = g,    (8)

    with u in I and g in D.

    In the following, three vector spaces are employed:

        I, 2D or 3D space of discrete images;

        D, space of sinograms (projection data);

        V = I^d, space of spatial-vector-valued image array with d = 2 or 3
        for 2D or 3D spaces I, respectively. E.g. the spatial gradient for
        an image u used to form the total variation (TV) semi-norm.


    Parameters
    ----------

    projections : numpy.ndarray
        Numpy array holding the projection data.

    projector : instance of class 'Projector'
        Projector instance representing the operator K and its adjoint K as
        defined in the class Projector in pyastra.

    """

    def __init__(self, projections=np.array([]), projector=Projector()):
        self.y = projections.astype('float32', copy=False)
        self.K = projector

    def matrix_norm(self, iterations, vol_init=1.0,
                    intermediate_results=False, continue_iteration=False):
        """The matrix norm || K ||_2  of 'K' defined here as largest
        singular value of 'K'. Employs the generic power method to obtain a
        scalar 's' which tends to || K ||_2 as the iterations N increase.

        To be implemented: optionally return volume 'x', such that it can be
        re-used as initializer to continue the iteration.

        Parameters
        ----------
        :type iterations: int
        :param iterations: Number of iterations the generic power method is
        doing.
        :type vol_init: float | ndarray (default 1.0)
        :param vol_init: in I, initial image to start with.
        :type intermediate_results: bool (default False)
        :param intermediate_results: Returns list of intermediate results
        instead of scalar.
        :type continue_iteration: bool, (default False)
        :param continue_iteration: Toggle if previous calculation should be
        returned.

        Returns
        -------
        :rtype s float | numpy.ndarray
        :returns s: Scalar of final iteration or numpy.ndarray containing all
        results during iteration.
        """

        if intermediate_results:
            s = np.zeros(iterations)
        else:
            s = 0

        if continue_iteration is False:
            if np.isscalar(vol_init):
                vol_init = vol_init * np.ones(self.K.num_voxel)

            # store initial volume in ASTRA memory
            self.K.set_volume_data(vol_init)

        for n in range(iterations):
            # step 4: K* (K x_n)

            self.K.forward()
            self.K.backward()

            # step 5: x/||x||_2
            vol_init = self.K.volume_data
            vol_init /= np.linalg.norm(np.ravel(vol_init))
            self.K.set_volume_data(vol_init)

            # step 6:
            if intermediate_results:
                self.K.forward()
                s[n] = np.linalg.norm(np.ravel(self.K.projection_data))

        # step 6:
        if not intermediate_results:
            self.K.forward()
            s = np.linalg.norm(np.ravel(self.K.projection_data))

        # print "intermediate results: ", s
        # print "update:", s[1:]-s[:-1]

        # Return approximate power: scalar or list including intermediate
        # results
        return s

    def least_squares(self, num_iterations=1, L=None, tau=None, sigma=None,
                      theta=None, non_negativiy_constraint=False,
                      verbose=True):
        """Least-squares problem, unconstrained or with
        non-negativity constraint.

        Parameters
        ----------
        :type num_iterations: int (default 1)
        :param num_iterations: Number of iterations the optimization should
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
        :type verbose: bool (default False)
        :param verbose: Show intermediate reconstructions and
        convergence measures during iteration.

        Returns
        -------

        Consider the unconstrained least-squares problem, i.e. a
        quadratic error function, only. The primal problem states:

            min_u 1/2 || A u - g ||_2^2.    (11)

        Association with the primal problem (1), see class docstring:

            F(y) = 1/2  || y - g ||_2^2.    (12)
            G(x) = 0,    (13)
            x = u, y = A u,    (14)
            K = A.    (15)

        From equation (3), the convex conjugates of F and G are obtained:

            F^*(p) = 1/2 || p ||_2^2 + <p,g>_D,    (16)
            G^*(q) = max_x <q,x>_I = delta_{O_I}(q),    (17)

        where p in D dual to g, q in I dual to x.

        The optimization problem dual to equation (11) reads:

            max_p { -1/2||p||_2^2 - <p,g>_D - delta_{0_I}(-A^T p) }.    (19)

        The proximal mappings for y in Dreads and x in I:

            prox_sigma[F^*](y) = arg min_{y'} { 1/2||y'||_2^2 + <y',g>_D
                                  + 1/(2sigma)||y-y'||_2^2 }     (22)

                                = (y = sigma g) / (1 + sigma) ,

            prox_\tau[G](x) = x .    (23)

        Conditional primal-dual gap, i.e. the difference between the primal
        and dual objectives ignoring the indicator function) for estimates
        u' and p':

            cPD(u',p') = 1/2||A u' -g||_2^2 + 1/2||p'||_2^^2 + <p',g>_D .  (21)

        cPD needs not to be positive, but should tend to zero. Also monitor
        A^T p' which should tend to 0_I.
        """

        # step 0:
        g = self.y
        # l2-norm of A^T p' with intermediate result p' = p
        l2_atp = np.zeros(num_iterations)
        # conditional primal-dual gap
        cpd = np.zeros(num_iterations)

        # step 1:
        if L is None:
            L = self.matrix_norm(20)
        if tau is None:
            tau = 1 / L
        if sigma is None:
            sigma = 1 / L
        if theta is None:
            theta = 1

        # step 2:
        u = np.zeros(self.K.num_voxel, dtype=np.float32)
        p = np.zeros((self.K.det_col_count,
                      len(self.K.angles), self.K.det_row_count),
                     dtype=np.float32)

        # step 3:
        ub = np.zeros_like(u)

        u_size = u.size
        p_size = p.size

        # visual output
        disp = DisplayIntermediates(verbose=verbose, vol=u, cpd=cpd,
                                    l2_du=l2_atp)

        # step 4: repeat
        for n in range(num_iterations):
            # step 5: p_{n+1}
            if n >= 0:
                self.K.set_volume_data(ub)
                self.K.forward()
                print 'p:', p.shape, 'g:', g.shape, 'proj:', \
                    self.K.projection_data.shape
                p += sigma * (self.K.projection_data - g)
            else:
                p += sigma * (- g)
            p /= 1 + sigma

            # step 6:
            # A^T pnnn
            self.K.set_projection_data(p)
            self.K.backward()
            # l2-norm of A^T p
            l2_atp[n:] = np.linalg.norm(np.ravel(self.K.volume_data)) / u_size
            # Use 'ub' as temporary memory for 'u_n'
            ub = u.copy()
            # u_{n+1} = u_{n} - tau * A^T p_{n+1}
            u -= tau * self.K.volume_data
            if non_negativiy_constraint:
                u[u < 0] = 0

            # conditional primal-dual gap:
            # 1/2||A u-g||_2^2 + 1/2||p||_2^2 + <p,g>_D
            self.K.set_volume_data(u)
            self.K.forward()
            cpd[n:] = (0.5 *
                       np.linalg.norm(
                           np.ravel(self.K.projection_data - g)) ** 2 +
                       0.5 * np.linalg.norm(np.ravel(p)) ** 2 +
                       np.sum(np.ravel(p * g))) / p_size

            # step 7:
            ub = u + theta * (u - ub)

            # visual output
            disp.update()

        self.K.clear()
        disp.show()

        return u, p, cpd, l2_atp
