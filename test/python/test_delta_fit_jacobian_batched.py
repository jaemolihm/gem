'''
jacobian_LR and jacobian_LcD (one eigh of H, parameter directions as dF_spectral stacks) against the per-direction
reference below (one dF_spectral, i.e. one eigh, per packed parameter), on random complex and real parameters and
on an SU(2) case X = A (x) 1_2 whose H has exactly degenerate eigenvalue pairs (the f' limit of dF_spectral).
The degenerate case is also checked against a central finite difference of the residual, and a chunked Jacobian
(JACOBIAN_CHUNK smaller than the number of parameters) against one chunk.
'''
import unittest
import numpy as np
from gem import delta_fit
from gem.delta_fit import (
    pack_params,
    unpack_params,
    residual_LR,
    residual_LcD,
    jacobian_LR,
    jacobian_LcD,
    build_H,
    F_of_H,
    dF_spectral,
)


# ---------- Per-direction reference Jacobians ----------
def dH_dLambda(dLambda, n):
    Z = np.zeros((n, n), dtype=complex)
    return np.block([[dLambda, Z], [Z, Z]])

def dH_dR(dR, D):
    n, p = D.shape
    Z_nn = np.zeros((n, n), dtype=complex)
    term12 = dR @ D.T
    return np.block([[Z_nn, term12], [term12.conj().T, Z_nn]])

def dH_dLambdac(dLambdac, n):
    Z = np.zeros((n, n), dtype=complex)
    return np.block([[Z, Z], [Z, -dLambdac]])

def dH_dD(dD, R):
    n, p = R.shape
    Z_nn = np.zeros((n, n), dtype=complex)
    term12 = R @ dD.T
    return np.block([[Z_nn, term12], [term12.conj().T, Z_nn]])

def jacobian_LR_reference(x, beta, Lambda_c, D, F22_target, RTF12_target):
    n, p = D.shape
    Lambda, R = unpack_params(x, n, p)
    H = build_H(Lambda, Lambda_c, D, R)
    F = F_of_H(H, beta).T
    F12 = F[:n, n:]
    iu = np.triu_indices(n)
    iu_strict = np.triu_indices(n, k=1)
    Jcols = []
    for k in range(len(x)):
        dx = np.zeros_like(x)
        dx[k] = 1.0
        dLambda, dR = unpack_params(dx, n, p)
        dH = dH_dLambda(dLambda, n) + dH_dR(dR, D)
        dF = dF_spectral(H, dH, beta).T
        dF22 = dF[n:, n:]
        dRTF12 = R.T @ dF[:n, n:] + dR.T @ F12
        Jcols.append(np.concatenate([dF22.real[iu], dF22.imag[iu_strict],
                                     dRTF12.real.ravel(), dRTF12.imag.ravel()]))
    return np.column_stack(Jcols)

def jacobian_LcD_reference(x, beta, Lambda, R, F11_target, F12D_target):
    n, p = R.shape
    Lambda_c, D = unpack_params(x, n, p)
    H = build_H(Lambda, Lambda_c, D, R)
    F = F_of_H(H, beta).T
    F12 = F[:n, n:]
    iu = np.triu_indices(n)
    iu_strict = np.triu_indices(n, k=1)
    Jcols = []
    for k in range(len(x)):
        dx = np.zeros_like(x)
        dx[k] = 1.0
        dLambda_c, dD = unpack_params(dx, n, p)
        dH = dH_dLambdac(dLambda_c, n) + dH_dD(dD, R)
        dF = dF_spectral(H, dH, beta).T
        dF11 = dF[:n, :n]
        dF12D = dF[:n, n:] @ D + F12 @ dD
        Jcols.append(np.concatenate([dF11.real[iu], dF11.imag[iu_strict],
                                     dF12D.real.ravel(), dF12D.imag.ravel()]))
    return np.column_stack(Jcols)


# ---------- Inputs ----------
def random_hermitian(rng, n, real):
    A = rng.standard_normal((n, n)) + (0 if real else 1j * rng.standard_normal((n, n)))
    return 0.5 * (A + A.conj().T)

def random_rect(rng, n, p, real):
    return rng.standard_normal((n, p)) + (0 if real else 1j * rng.standard_normal((n, p)))

def inputs(seed, n, p, real=False, su2=False, beta=10.0):
    '''(Lambda, R, Lambda_c, D) and the targets of residual_LR / residual_LcD; su2: each X = A (x) 1_2.'''
    rng = np.random.default_rng(seed)
    nn, pp = (n // 2, p // 2) if su2 else (n, p)
    L, Lc = random_hermitian(rng, nn, real), random_hermitian(rng, nn, real)
    R, D = random_rect(rng, nn, pp, real), random_rect(rng, nn, pp, real)
    if su2:
        L, Lc, R, D = (np.kron(X, np.eye(2)) for X in (L, Lc, R, D))
    F = F_of_H(build_H(L, Lc, D, R), beta).T
    F11, F12, F22 = F[:n, :n], F[:n, n:], F[n:, n:]
    return dict(L=L, R=R, Lc=Lc, D=D, beta=beta, F11=F11 + 0.01, F12D=F12 @ D, F22=F22 - 0.01, RTF12=R.T @ F12)


class TestJacobianBatched(unittest.TestCase):

    def check(self, d):
        x_LR, x_LcD = pack_params(d["L"], d["R"]), pack_params(d["Lc"], d["D"])
        J = jacobian_LR(x_LR, d["beta"], d["Lc"], d["D"], d["F22"], d["RTF12"])
        J0 = jacobian_LR_reference(x_LR, d["beta"], d["Lc"], d["D"], d["F22"], d["RTF12"])
        self.assertEqual(J.shape, J0.shape)
        self.assertLessEqual(np.abs(J - J0).max(), 1e-13 * np.abs(J0).max())
        J = jacobian_LcD(x_LcD, d["beta"], d["L"], d["R"], d["F11"], d["F12D"])
        J0 = jacobian_LcD_reference(x_LcD, d["beta"], d["L"], d["R"], d["F11"], d["F12D"])
        self.assertEqual(J.shape, J0.shape)
        self.assertLessEqual(np.abs(J - J0).max(), 1e-13 * np.abs(J0).max())

    def test_random_complex(self):
        self.check(inputs(1, 4, 4))
        self.check(inputs(2, 6, 2))

    def test_random_real(self):
        self.check(inputs(3, 4, 4, real=True))
        self.check(inputs(4, 6, 2, real=True))

    def test_degenerate_su2(self):
        for real in (False, True):
            d = inputs(5, 6, 2, real=real, su2=True)
            eps = np.linalg.eigvalsh(build_H(d["L"], d["Lc"], d["D"], d["R"]))
            self.assertLess(np.diff(eps).min(), 1e-10)  # the f' limit of dF_spectral is used
            self.check(d)

    def test_chunks(self):
        # chunks of 7 directions (N = 60: 8 full chunks and a partial one) equal one chunk, bitwise
        d = inputs(7, 6, 2)
        x_LR, x_LcD = pack_params(d["L"], d["R"]), pack_params(d["Lc"], d["D"])
        calls = (lambda: jacobian_LR(x_LR, d["beta"], d["Lc"], d["D"], d["F22"], d["RTF12"]),
                 lambda: jacobian_LcD(x_LcD, d["beta"], d["L"], d["R"], d["F11"], d["F12D"]))
        chunk = delta_fit.JACOBIAN_CHUNK
        try:
            delta_fit.JACOBIAN_CHUNK = len(x_LR)
            J_one = [jac() for jac in calls]
            delta_fit.JACOBIAN_CHUNK = 7
            J_chunked = [jac() for jac in calls]
        finally:
            delta_fit.JACOBIAN_CHUNK = chunk
        for J0, J in zip(J_one, J_chunked):
            self.assertTrue(np.array_equal(J0, J))

    def test_degenerate_su2_finite_difference(self):
        d = inputs(6, 6, 2, su2=True)
        h = 1e-6
        for x, jac, res in (
                (pack_params(d["L"], d["R"]),
                 lambda x: jacobian_LR(x, d["beta"], d["Lc"], d["D"], d["F22"], d["RTF12"]),
                 lambda x: residual_LR(x, d["beta"], d["Lc"], d["D"], d["F22"], d["RTF12"])),
                (pack_params(d["Lc"], d["D"]),
                 lambda x: jacobian_LcD(x, d["beta"], d["L"], d["R"], d["F11"], d["F12D"]),
                 lambda x: residual_LcD(x, d["beta"], d["L"], d["R"], d["F11"], d["F12D"]))):
            E = np.eye(len(x))
            J_fd = np.column_stack([(res(x + h * E[k]) - res(x - h * E[k])) / (2 * h) for k in range(len(x))])
            J = jac(x)
            self.assertLess(np.abs(J - J_fd).max(), 1e-7 * np.abs(J).max())


if __name__ == '__main__':
    unittest.main()
