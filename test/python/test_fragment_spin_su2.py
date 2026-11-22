'''
Fragment.update_hybridization / update_self_energy with spin_su2=True: on SU(2) inputs (X = A (x) 1_2 for the
parameters and the targets) the fit of the spin-up block alone equals the full fit to 1e-10, at T > 0 (least
squares, on half-size blocks) and T = 0 (closed form); a non-SU(2) input raises ValueError.
'''
import unittest
import numpy as np
import gem.fragment as fragment_module
from gem.fragment import Fragment
from gem.solvers.simple_ed import SimpleED
from gem.utilities import U_matrix_kanamori
from gem.delta_fit import build_H, F_of_H


def su2(rng, n, p, hermitian=False):
    '''A (x) 1_2 with A random complex (n/2, p/2), Hermitian if asked.'''
    A = rng.standard_normal((n // 2, p // 2)) + 1j * rng.standard_normal((n // 2, p // 2))
    if hermitian:
        A = 0.5 * (A + A.conj().T)
    return np.kron(A, np.eye(2))

def occupation(rng, n):
    '''A (x) 1_2 Hermitian with eigenvalues in (0.1, 0.9), a density matrix target.'''
    Q, _ = np.linalg.qr(rng.standard_normal((n // 2, n // 2)) + 1j * rng.standard_normal((n // 2, n // 2)))
    A = (Q * rng.uniform(0.1, 0.9, n // 2)) @ Q.conj().T
    return np.kron(A, np.eye(2))

def make_fragment(seed, T=0.0):
    '''Two orbitals (nimp = 4 spin-orbitals), B = 2, SU(2) parameters and targets. At T > 0 the targets are those
    of nearby SU(2) parameters (Lambda*, R*, Lambda_c*, D*), so the fits converge to a well defined minimum:
    Delta_qp = F11, ERD^T = F12 D* of H(Lambda, Lambda_c*, D*, R), and the denMat blocks <f^dag f> = F22,
    <c^dag f> = R*^T F12 of H(Lambda*, Lambda_c, D, R*), with F = f(H)^T as in delta_fit.'''
    norb, B = 2, 2
    nimp, nbath = 2 * norb, 2 * norb * B
    solver = SimpleED(nimp + nbath, use_Ntot=True, use_Sz=True, dtype=np.complex128)
    eloc = np.zeros((nimp, nimp), dtype=complex)
    frag = Fragment(nimp, nbath, eloc, U_matrix_kanamori(norb, 2.0, 0.5), solver)
    rng = np.random.default_rng(seed)
    frag.Lambda = frag.Lambda + 0.2 * su2(rng, nbath, nbath, hermitian=True)
    frag.Lambda_c = frag.Lambda_c + 0.2 * su2(rng, nbath, nbath, hermitian=True)
    frag.R = frag.R + 0.1 * su2(rng, nbath, nimp)
    frag.D = frag.D + 0.1 * su2(rng, nbath, nimp)
    frag.Delta_qp = occupation(rng, nbath)
    frag.ERD = 0.3 * su2(rng, nimp, nbath)
    frag.denMat = np.zeros((nimp + nbath, nimp + nbath), dtype=complex)
    frag.denMat[nimp:, nimp:] = occupation(rng, nbath)
    frag.denMat[:nimp, nimp:] = 0.3 * su2(rng, nimp, nbath)
    frag.denMat[nimp:, :nimp] = frag.denMat[:nimp, nimp:].conj().T
    frag.denMat[:nimp, :nimp] = occupation(rng, nimp)
    if T > 0:
        n, beta = nbath, 1 / T
        Lambda_t = frag.Lambda + 0.05 * su2(rng, nbath, nbath, hermitian=True)
        R_t = frag.R + 0.05 * su2(rng, nbath, nimp)
        Lambda_c_t = frag.Lambda_c + 0.05 * su2(rng, nbath, nbath, hermitian=True)
        D_t = frag.D + 0.05 * su2(rng, nbath, nimp)
        F = F_of_H(build_H(frag.Lambda, Lambda_c_t, D_t, frag.R), beta).T
        frag.Delta_qp, frag.ERD = F[:n, :n], (F[:n, n:] @ D_t).T
        F = F_of_H(build_H(Lambda_t, frag.Lambda_c, frag.D, R_t), beta).T
        frag.denMat[nimp:, nimp:] = F[n:, n:]
        frag.denMat[:nimp, nimp:] = R_t.T @ F[:n, n:]
        frag.denMat[nimp:, :nimp] = frag.denMat[:nimp, nimp:].conj().T
    return frag

def copy_state(frag):
    return {k: np.array(getattr(frag, k)) for k in ("Lambda", "R", "Lambda_c", "D")}

def set_state(frag, state):
    for k, v in state.items():
        setattr(frag, k, v.copy())


class TestSpinSU2Fit(unittest.TestCase):

    def compare(self, method, T):
        frag = make_fragment(7, T)
        state = copy_state(frag)
        full = getattr(frag, method)(T=T)
        full = [np.array(X) for X in full]
        set_state(frag, state)
        shapes = []
        wrapped = {name: getattr(fragment_module, name) for name in
                   ("update_self_energy_thermal_penalty", "update_hybridization_thermal_penalty")}
        def record(f):
            def g(*args, **kwargs):
                shapes.append(args[0].shape)
                return f(*args, **kwargs)
            return g
        try:
            for name, f in wrapped.items():
                setattr(fragment_module, name, record(f))
            half = getattr(frag, method)(T=T, spin_su2=True)
        finally:
            for name, f in wrapped.items():
                setattr(fragment_module, name, f)
        if T > 0:  # one fit, on the spin-up block
            self.assertEqual(shapes, [(frag.nbath // 2, frag.nbath // 2)])
        names = ("R", "Lambda") if method == "update_self_energy" else ("D", "Lambda_c")
        for name, X, Y in zip(names, full, half):
            self.assertGreater(np.abs(X - state[name]).max(), 1e-2)  # the fit moved
            self.assertLess(np.abs(X - Y).max(), 1e-10 * np.abs(X).max())
            np.testing.assert_array_equal(Y, np.kron(Y[::2, ::2], np.eye(2)))

    def test_self_energy_T(self):
        self.compare("update_self_energy", 0.05)

    def test_hybridization_T(self):
        self.compare("update_hybridization", 0.05)

    def test_self_energy_T0(self):
        self.compare("update_self_energy", 0.0)

    def test_hybridization_T0(self):
        self.compare("update_hybridization", 0.0)

    def test_not_su2_raises(self):
        splitting = np.array([[1, 0], [0, -1]])  # X_uu != X_dd of one orbital
        spin_flip = np.array([[0, 1], [0, 0]])  # X_ud != 0 of one orbital
        for method, attr, block, delta in (
                ("update_self_energy", "denMat", (slice(4, 6), slice(4, 6)), splitting),
                ("update_self_energy", "R", (slice(0, 2), slice(0, 2)), splitting),
                ("update_self_energy", "R", (slice(0, 2), slice(0, 2)), spin_flip),
                ("update_hybridization", "Delta_qp", (slice(0, 2), slice(0, 2)), splitting),
                ("update_hybridization", "ERD", (slice(0, 2), slice(0, 2)), splitting),
                ("update_hybridization", "ERD", (slice(0, 2), slice(0, 2)), spin_flip)):
            frag = make_fragment(8, 0.05)
            X = np.array(getattr(frag, attr))
            X[block] += 1e-8 * delta
            setattr(frag, attr, X)
            with self.assertRaises(ValueError):
                getattr(frag, method)(T=0.05, spin_su2=True)
            getattr(frag, method)(T=0.05)  # the full fit takes it


if __name__ == '__main__':
    unittest.main()
