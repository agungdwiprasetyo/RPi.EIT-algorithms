from __future__ import division, absolute_import, print_function

cdef class EITBase(object):
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
