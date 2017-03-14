import numpy as np

# cdef class BackProjection(EITBase):
#     cdef void setup(self, string weight='none'):
#         self.params = {'weight': weight}
#         if weight=='simple':
#             weight = self.simpleWeight(self.B.shape[0])
#             self.H = weight*self.B

#     def solve(self, double[:] v1, double[:] v0, bool normalize):
#         self._solve(v1, v0, normalize):

#     cdef double[:] _solve(self, double[:] v1, double[:] v2, bool normalize):
#         cdef int i,j
#         cdef int colA = self.H.shape[1]
#         cdef int rowA = self.H.shape[0]
#         cdef int N = v0.shape[0]
#         cdef int[:,;] y
#         cdef double[:] vn
#         cdef double[:,:] hasil

#         for i in range(N):
#             for i in range(N):
#             y[i][j] = 1
#         if normalize:
#             vn = -(v1-v0)/y
#         else:
#             vn = (v1-v0)

#         cdef int colB = vn.shape[1]
#         cdef int rowB = vn.shape[0]

#         for i in range(colA):
#             for j in range(rowA):

cdef class BackProjection(object):
    cdef double[:] node
    cdef double[:] element
    cdef double[:] alpha
    cdef double[:] jacobian
    cdef double[:] B
    cdef double[:] H
    cdef str[:] params

    def __init__(self, mesh, forward):
        self.node = mesh['node']
        self.element = mesh['element']
        self.alpha = mesh['alpha']
        self.jacobian = forward.jac
        self.B = forward.b_matrix
        self.H = self.B

        self.setup()

    def setup(self, weight='none'):
        # self.params = ["weight": weight]

        if weight is 'simple':
            weight = self.simpleWeight(self.B.shape[0])
            self.H = weight * self.B

    def solve(self, v1, v0, normalize=True):
        if normalize:
            vn = -(v1 - v0)/np.sign(v0)
        else:
            vn = (v1 - v0)

        hasil = np.dot(self.H.transpose(), vn)
        return np.real(hasil)

    def solveGramSchmidt(self, v1, v0):
        a = np.dot(v1, v0)/np.dot(v0, v0)
        vn = -(v1 - a*v0)/np.sign(v0)
        hasil = np.dot(self.H.transpose(), vn)
        return np.real(hasil)

    def simpleWeight(self, numVoltages):
        d = np.sqrt(np.sum(self.nodeXY**2, axis=1))
        r = np.max(d)
        w = (1.01*r - d)/(1.01*r)

        weights = np.dot(np.ones((numVoltages, 1)), w.reshape(1,-1))
        return weights
