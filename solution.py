import numpy as np
import sklearn
from sklearn.metrics.pairwise import polynomial_kernel

DEGREE = 2
COEF0  = 0.3

def delays(w):
    alpha = w[:-1]
    last = w[-1]

    s = alpha.copy()
    s[-1] += last

    d = alpha.copy()
    d[-1] -= last

    a = np.maximum(s, 0.0)
    b = np.maximum(-s, 0.0)
    c = np.maximum(d, 0.0)
    e = np.maximum(-d, 0.0)

    return a, b, c, e

################################
# Non Editable Region Starting #
################################
def my_kernel( X1, Z1, X2, Z2 ):
################################
#  Non Editable Region Ending  #
################################

	X1 = np.asarray(X1).reshape(-1)
	X2 = np.asarray(X2).reshape(-1)
	Z1 = np.asarray(Z1)
	Z2 = np.asarray(Z2)

	if Z1.ndim == 1:
		Z1 = Z1.reshape(1, -1)
	if Z2.ndim == 1:
		Z2 = Z2.reshape(1, -1)

	poly = polynomial_kernel(Z1, Z2, degree=DEGREE, coef0=COEF0)

	x_outer = np.outer(X1, X2)  
	G = x_outer * poly + 1.0
	
	return G


################################
# Non Editable Region Starting #
################################
def my_decode( w ):
################################
#  Non Editable Region Ending  #
################################

	Z = w.reshape((33, 33))

	U, S, VT = np.linalg.svd(Z, full_matrices=False)

	sigma = S[0]
	root_sigma = np.sqrt(sigma)
	u = U[:, 0] * root_sigma
	v = VT[0, :] * root_sigma

	a, b, c, d = delays(u)
	p, q, r, s = delays(v)
	
	return a, b, c, d, p, q, r, s

