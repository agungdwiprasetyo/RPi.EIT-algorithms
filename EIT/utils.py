from __future__ import division, absolute_import, print_function
import numpy as np

def EIT_scanLines(numEl=16, dist=1):
	exPos = np.eye(numEl)
	exNeg = -1*np.roll(exPos, dist, axis=1)
	ex = exPos+exNeg
	return ex

if __name__ == '__main__':
	print(EIT_scanLines(dist=3))