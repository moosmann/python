"""
RL implementation of wavelet transform
"""

from odl import Rn
from odl import Operator
import libwaveletspy as wl


class WaveletDecompose(Operator):

    def __init__(self, n):
        self._domain = Rn(n)
        self._range = Rn(n)

    def _apply(self, rhs, out):
        """ Wavelet transform: Decompose input vector into wavelets.

        Parameters
        ----------

        rhs : RnVector
              RnVector of dimension n to be transformed.

        out : RnVector
              RnVector of dimension n the result should be written to.

        Returns
        -------

        RnVector of dimension n of wavelet transformed input RnVector of dimension n.


        Examples
        --------

        >>> n = 10
        >>> rn = Rn(n)
        >>> data = rn.element(range(n))
        >>> decompose = WaveletDecompose(n)
        >>> decomposedData = decompose(data)
        >>> print(decomposedData)
        [  2.03223267e-01  -8.87556695e-12  -1.47952761e-11  -1.48388367e-01
           7.03223267e-01   5.09930045e-01  -7.44210796e-01  -3.41010069e-01
           1.78491552e+01   1.38856477e+02]

        """

        tmp = rhs.copy()
        wl.wavelet_transform1D(tmp.data.ctypes.data, self.domain.n, out.data.ctypes.data)

    def _applyAdjoint(self, rhs, out):
        """Apply the adjoint of the operator.

        """

    @property
    def domain(self):
        return self._domain

    @property
    def range(self):
        return self._range


class WaveletReconstruct(Operator):

    def __init__(self, n):
        self._domain = Rn(n)
        self._range = Rn(n)

    def _apply(self, rhs, out):
        """ Inverse wavelet transform: Reconstruct output vector from vector of wavelet coefficients.

        Parameters
        ----------

        rhs : RnVector
              RnVector of dimension n to be transformed.

        out : RnVector
              RnVector of dimension n the result should be written to.

        Returns
        -------

        RnVector of dimension n of inverse wavelet transformed input RnVector of dimension n.


        Examples
        --------

        >>> n = 10
        >>> rn = Rn(n)
        >>> data = rn.element(range(1, n + 1, 1))
        >>> decompose = WaveletDecompose(n)
        >>> decomposedData = decompose(data)
        >>> reconstruct = WaveletReconstruct(n)
        >>> result = reconstruct(decomposedData)
        >>> print(result)
        [  1.   2.   3.   4.   5.   6.   7.   8.   9.  10.]

        """

        tmp = rhs.copy()
        wl.invwavelet_transform1D(tmp.data.ctypes.data, self.range.n, out.data.ctypes.data)

    def _applyAdjointImpl(self, rhs, out):
        """Apply the adjoint of the operator.

        """

    @property
    def domain(self):
        return self._domain

    @property
    def range(self):
        return self._range
