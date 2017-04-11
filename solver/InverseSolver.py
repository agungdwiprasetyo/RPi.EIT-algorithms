from EIT.BackProjection import BackProjection
from EIT.Jacobian import Jacobian
from EIT.GREIT import GREIT

import matplotlib.pyplot as plt

class InverseSolver(object):
    def __init__(self, mesh, forward):
        self.forward = forward
        self.ref = forward.v
        self.mesh = mesh
        nodeXY = mesh['node']
        self.element = mesh['element']
        self.nodeX = nodeXY[:, 0]
        self.nodeY = nodeXY[:, 1]

    def solve(self, algor, data):
        self.algor = algor
        if algor=="BP":
            inverse = BackProjection(self.mesh, self.forward)
            self.result = inverse.solveGramSchmidt(data, self.ref)

        elif algor=="JAC":
            inverse = Jacobian(self.mesh, self.forward)
            self.result = inverse.solve(data, self.ref)

        elif algor=="GREIT":
            inverse = GREIT(self.mesh, self.forward)
            self.result = inverse.solve(data, self.ref)

    def plot(self, size, colorbar):
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

        # plt.show()
        return fig