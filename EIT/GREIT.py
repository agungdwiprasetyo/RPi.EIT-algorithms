from __future__ import division, absolute_import, print_function

import numpy as np
import scipy.linalg as la
from matplotlib.path import Path

from .EITbase import EITBase


class GREIT(EITBase):
    """ the GREIT algorithm """

    def setup(self, method='dist', w=None, p=0.20, lamb=1e-2, n=32, s=20., ratio=0.1):
        """
        set up for GREIT

        Parameters
        ----------
        method: str, optional, 'set' or 'dist'
        w: NDArray, optional, weight on each element
        p: float, optional, noise covariance
        lamb: float, regularization parameters
        n: int, optional, grid size
        s: float, optional, control the blur
        ratio : float, optional, desired ratio

        References
        ----------
        .. [1] Bartlomiej Grychtol, Beat Muller, Andy Adler
               "3D EIT image reconstruction with GREIT"
        .. [2] Adler, Andy, et al. "GREIT: a unified approach to
               2D linear EIT reconstruction of lung images."
               Physiological measurement 30.6 (2009): S35.
        """
        # parameters for GREIT projection
        if w is None:
            w = np.ones_like(self.alpha)
        self.params = {
            'w': w,
            'p': p,
            'lamb': lamb,
            'n': n,
            's': s,
            'ratio': ratio
        }
        # action (currently only support 'dist')
        if method is 'dist':
            self.H = self._build_dist()
        else:
            raise ValueError('method ' + method + ' not supported yet')

    def solve(self, v1, v0, normalize=False):
        """ solving and interpolating (psf convolve) on grids. """
        if normalize:
            dv = -(v1 - v0)/v0
        else:
            dv = v1 - v0
        return -np.dot(self.H, dv)

    def solveGramSchmidt(self, v1, v0):
        a = np.dot(v1, v0) / np.dot(v0, v0)
        dv = (v1 - a * v0)  # (v1 - a*v0)/np.sign(v0)
        ds = -np.dot(self.H, dv)
        return np.real(ds)

    def map(self, v):
        """ return H*v """
        return -np.dot(self.H, v)

    def _build_dist(self):
        """ generate R using distribution method. """
        lamb = self.params['lamb']
        p = self.params['p']

        jac = self.jacobian
        # build D on grids
        xg, yg, mask = self._mask_grid()
        r_max = self._r_max(xg, yg, mask)
        d_mat = self._psf_grid(xg, yg, r_max=r_max)
        # E[yy^T]
        j_j_w = np.dot(jac, jac.transpose())
        r_mat = np.diag(np.diag(j_j_w) ** p)
        jac_inv = la.inv(j_j_w + lamb*r_mat)
        # RM = E[xx^T] / E[yy^T]
        h_mat = np.dot(np.dot(d_mat, jac.transpose()), jac_inv)
        return h_mat

    def _build_grid(self):
        """ building grid from mesh in 2D. grid size should be small """
        n = self.params['n']

        x_min, x_max = min(self.node[:, 0]), max(self.node[:, 0])
        y_min, y_max = min(self.node[:, 1]), max(self.node[:, 1])
        xv = np.linspace(x_min, x_max, num=n, endpoint=True)
        yv = np.linspace(y_min, y_max, num=n, endpoint=True)
        xg, yg = np.meshgrid(xv, yv, sparse=False, indexing='xy')

        return xg, yg

    def _build_mask(self, xg, yg):
        """ build boolean matrix mark interior points """

        # 1. create mask based on meshes
        points = np.vstack((xg.flatten(), yg.flatten())).T

        # 2. extract edge points using el_pos
        edge_points = self.node[np.arange(16)]
        path = Path(edge_points, closed=False)
        mask = path.contains_points(points)

        return mask

    def _mask_grid(self):
        """
        generate xy grids and mask
        """
        xg, yg = self._build_grid()
        mask = self._build_mask(xg, yg)
        mask = mask.reshape(xg.shape)

        return xg, yg, mask

    def _r_max(self, xg, yg, mask):
        """
        calculate max radius using mask and xy

        Notes
        -----
        mask is a 1d boolean array, xg and yg are 2D meshgrid,
        so mask should be reshaped before indexing (numpy 1.10)
        """
        xg_mask, yg_mask = xg[mask], yg[mask]
        r = self._distance2d(xg_mask, yg_mask)
        return np.max(r)

    def _psf_grid(self, xg, yg, r_max=1.):
        """
        point spread function (psf) mapping (convolve)
        values of elements on grid.
        """
        ratio = self.params['ratio']
        s = self.params['s']

        ng = xg.size
        ne = self.element.shape[0]
        d_mat = np.zeros((ng, ne))
        r_mat = r_max * ratio
        # loop over all elements
        for i in range(ne):
            ei = self.element[i, :]
            xy = np.mean(self.node[ei], axis=0)
            # there may be bias between grid and center of a element
            r = self._distance2d(xg, yg, center=xy)
            f = 1./(1.+np.exp(s*np.abs(r))/np.exp(s*r_mat))
            d_mat[:, i] = f
        return d_mat

    def mask_value(self, ds, mask_value=0):
        """ (plot only) mask values on nodes outside 2D mesh. """
        xg, yg, mask = self._mask_grid()
        ds = ds.reshape(xg.shape)
        ds[mask == 0] = mask_value
        return xg, yg, ds

    @staticmethod
    def _distance2d(x, y, center=None):
        """ Calculate radius given center. This function can be OPTIMIZED """
        if center is None:
            xc, yc = np.mean(x), np.mean(y)
        else:
            xc, yc = center[0], center[1]
        r = np.sqrt((x-xc)**2 + (y-yc)**2).ravel()
        return r

    @staticmethod
    def build_set(x, y):
        """ generate R from a set of training sets (deprecate). """
        # E_w[yy^T]
        y_y_t = la.inv(np.dot(y, y.transpose()))
        h_matrix = np.dot(np.dot(x, y), y_y_t)
        return h_matrix
