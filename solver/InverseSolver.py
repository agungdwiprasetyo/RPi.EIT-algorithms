from EIT.BackProjection import BackProjection
from EIT.Jacobian import Jacobian
from EIT.GREIT import GREIT

import numpy as np
import matplotlib.pyplot as plt

class InverseSolver(object):
    def __init__(self, mesh, forward):
        self.forward = forward
        self.ref = forward.v
        self.mesh = mesh
        node = mesh['node']
        self.element = mesh['element']
        self.nodeX = node[:,0]
        self.nodeY = node[:,1]

    def solve(self, data, algor="BP"):
        self.algor = algor
        if algor=="BP":
            inverse = BackProjection(self.mesh, self.forward)
            inverse.setup(weight="simple")
            self.result = inverse.solveGramSchmidt(data, self.ref)

        elif algor=="JAC":
            inverse = Jacobian(self.mesh, self.forward)
            inverse.setup(p=0.40, lamb=1e-4, method='kotre')
            self.result = inverse.solveGramSchmidt(data, self.ref)

        elif algor=="GREIT":
            inverse = GREIT(self.mesh, self.forward)
            inverse.setup(p=0.50, lamb=1e-4)
            ds = inverse.solveGramSchmidt(data, self.ref)
            x, y, ds = inverse.mask_value(ds, mask_value=np.NAN)
            self.result = np.real(ds)
        return self.result

    def plot(self, size, colorbar, showPlot):
        fig = plt.figure()
        ax1 = fig.add_subplot(111)

        if self.algor=="GREIT":
            im = ax1.imshow(self.result, interpolation='none')
            ax1.axis(size)
            fig.set_size_inches(6, 6)
            plt.axis('off')
            ax1.axis('equal')

        else:
            im = ax1.tripcolor(self.nodeX, self.nodeY, self.element, self.result)
            ax1.axis('equal')
            ax1.axis(size)
            fig.set_size_inches(6, 6)
            plt.axis('off')

        if colorbar:
            fig.colorbar(im)

        if showPlot:
            plt.show()

        return fig