import numpy as np
import unittest

x_vec = np.array([1., 2., 3., 4.])
x_vec = x_vec.reshape(x_vec.shape + (1,))

def variance(alpha, beta, new_x, x_vec, D):
    assert isinstance(alpha, float), "Alpha has to be a float."
    assert isinstance(beta, float),  "Beta has to be a float."
    assert isinstance(new_x, float), "New_x has to be a float."
    assert x_vec.shape[1] == 1,      "x_vec has to be a column vector."
    assert isinstance(D, int),       "Dimensions has to be an integer."

    S = matrix_S(alpha,beta,x_vec, D)
    return (1/beta)+(phi(new_x, D).T.dot(S).dot(phi(new_x, D)))[0][0]

def phi(x, D):
    """
    x - is the real valued constant.
    D - is an integer dimension.
    """
    assert isinstance(x, float), "x should be a float."
    assert isinstance(D, int), "Dimension D, should be an integer."

    dimension = D + 1  # We are including a constant here.
    returnVector = [0] * dimension
    for i in range(dimension): returnVector[i] += x ** i

    returnVector = np.array(returnVector)
    returnVector = returnVector.reshape(returnVector.shape + (1,))

    return returnVector

def mean(alpha, beta, new_x, target_vec, x_vec, D):
    assert isinstance(alpha, float), "Alpha should be a float."
    assert isinstance(beta, float),  "Beta should be a float."
    assert isinstance(new_x, float), "new_x should be a float."
    assert target_vec.shape[1] == 1, "target_vec should be a column vector."
    assert x_vec.shape[1] == 1,      "x_vec should be a column vector."
    assert isinstance(D, int),       "Dimension should be an integer."
    assert len(target_vec)==len(x_vec), "x_vec and target_vec must be of the same dimensionality."

    sum_vec_x = 0
    for i in range(len(x_vec)): sum_vec_x += phi(x_vec[i][0], D) * target_vec[i][0]

    S = matrix_S(alpha, beta, x_vec, D)

    return beta * ((phi(new_x, D).T.dot(S)).dot(sum_vec_x))[0][0]

def matrix_S(alpha, beta, x_vec, D):

    assert isinstance(alpha, float), "Alpha is not a float."
    assert isinstance(beta, float),  "Beta is not a float."
    assert isinstance(D, int),       "Dimension is not an int."
    assert x_vec.shape[1] == 1, "x_vec must be a column vector."

    first_expr  = alpha*np.eye(D+1)
    second_expr = 0
    for i in range(len(x_vec)): second_expr += phi(x_vec[i][0], D).dot(phi(x_vec[i][0],D).T)
    return np.linalg.inv(first_expr+beta*second_expr)

class TestPredictiveDistribution(unittest.TestCase):

    # ---------- Phi method testing -------------
    def test_phi_validArg(self):
        new_x = 0.2
        phi_x = phi(new_x,1)
        self.assertEqual((2,1),phi_x.shape) # Making sure we have a 2x1 vector
        for i in range(phi_x.shape[0]):
            self.assertEqual(new_x**i,phi_x[i]) # Checking the method returns the correct values for the vector

    def test_phi_invalidD(self):
        with self.assertRaises(AssertionError):
            phi(2.,3.5)

    def test_phi_invalidX(self):
        with self.assertRaises(AssertionError):
            phi(2,10)

    # ----------- Matrix S testing ------------
    def test_matrixS_validArgs(self):
        alpha = 2.
        beta  = 2.
        D = 3
        S = matrix_S(alpha,beta,x_vec,D)
        S = np.linalg.inv(S)

        S_expected = 2.*np.eye(D+1)+2.*np.array([[4., 10., 30., 100.],
                                 [10., 30., 100., 354.],
                                 [30., 100., 354., 1300.],
                                 [100., 354., 1300., 4890.]])

        self.assertEqual(S_expected.shape,S.shape)

        S_D = S.shape
        for row in range(S_D[0]):
            for column in range(S_D[1]):
                self.assertAlmostEqual(S_expected[row][column],
                                 S[row][column],delta=0.0001)


    def test_matrixS_invalidAlpha(self):
        with self.assertRaises(AssertionError):
            matrix_S(0,1.,np.array([1,2,3]),2)

    def test_matrixS_invalidBeta(self):
        with self.assertRaises(AssertionError):
            matrix_S(0.,1,np.array([1,2,3]),3)

    def test_matrixS_invalidDim(self):
        with self.assertRaises(AssertionError):
            matrix_S(0.,1.,x_vec,2.3)

    def test_matrixS_invalidVec(self):
        x_vec = np.array([1.,2.,3.])
        with self.assertRaises(IndexError):
            matrix_S(0.,1.,x_vec,3)

    # --------- Variance tests ----------------
    def test_variance_validArgs(self):
        alpha = 0.
        beta = 1.
        D = 3
        S = matrix_S(alpha, beta, x_vec, D)
        S = np.linalg.inv(S)

        var = variance(alpha,beta,2.,x_vec,D)
        self.assertAlmostEqual(2.,var,delta=0.00001)

    def test_variance_invalidAlpha(self):
        with self.assertRaises(AssertionError):
            variance(1,2.,.25,x_vec,3)

    def test_variance_invalidBeta(self):
        with self.assertRaises(AssertionError):
            variance(1.,2,.25,x_vec,3)

    def test_variance_invalidNewx(self):
        with self.assertRaises(AssertionError):
            variance(1.,2.,3,x_vec,3)

    def test_variance_invalidXvec(self):
        with self.assertRaises(IndexError):
            variance(1.,2.,0.25,np.array([1.,2.,3.,4.]),3)

    def test_variance_invalidDim(self):
        with self.assertRaises(AssertionError):
            variance(1.,2.,0.25,x_vec,3.3)

    # ----------------- Mean tests ------------------
    def test_means_validArgs(self):
        target_vec = np.array([5.,4.,3.,2.]).reshape((4,)+(1,))
        # sum_vec_x should be [14,30,80,246].T
        new_x = 1.0
        # left matrix product should be [4.0,-4.3,1.5,-0.166667].T
        # result should be thus: 5.01158
        mn = mean(0.,1.,new_x,target_vec,x_vec,3)
        self.assertAlmostEqual(5.0115800,mn,delta=0.1)

    #TODO: invalid means tests
