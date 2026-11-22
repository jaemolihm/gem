'''
Fragment.impose_time_reversal_symmetry: R, Lambda, D, Lambda_c -> real parts, dtype and shape kept, and the
returned ratio max|Im X| / max|X| of what was dropped. After it, a SimpleED requested in float64 solves in float64.
'''
import unittest
import numpy as np
from gem.fragment import Fragment
from gem.solvers.simple_ed import SimpleED
from gem.utilities import U_matrix_kanamori


def _fragment(dtype):
    nimp, B = 2, 1
    solver = SimpleED(nimp * (B + 1), use_Ntot=True, use_Sz=True, dtype=dtype)
    eloc = np.diag([-0.5, -0.5]).astype(np.complex128)
    return Fragment(nimp, nimp * B, eloc, U_matrix_kanamori(1, 2.0, 0.0), solver)


class TestTimeReversal(unittest.TestCase):

    def test_real_parts_and_ratio(self):
        frag = _fragment(np.complex128)
        rng = np.random.default_rng(1)
        names = ("R", "Lambda", "D", "Lambda_c")
        for name in names:  # 2x2 matrices (one orbital, B = 1) with a real shift and an imaginary part 1e-3
            X = np.asarray(getattr(frag, name), dtype=np.complex128)
            setattr(frag, name, X + 0.3 * rng.standard_normal() * np.eye(2) + 1e-3j * np.ones_like(X))
        frag.impose_spin_SU2_symmetry()
        before = {name: np.array(getattr(frag, name)) for name in names}
        ratio = frag.impose_time_reversal_symmetry()
        expected = max(np.abs(X.imag).max() / np.abs(X).max() for X in before.values())
        self.assertAlmostEqual(ratio, expected, places=14)
        for name, X in before.items():
            Y = getattr(frag, name)
            self.assertEqual(Y.dtype, X.dtype)
            self.assertEqual(Y.shape, X.shape)
            np.testing.assert_array_equal(Y.real, X.real)
            self.assertEqual(np.abs(Y.imag).max(), 0.0)
        self.assertEqual(frag.impose_time_reversal_symmetry(), 0.0)

    def test_float64_ed(self):
        frag = _fragment(np.float64)
        frag.D = frag.D + 1e-9j
        frag.impose_spin_SU2_symmetry()
        frag.impose_time_reversal_symmetry()
        frag.solve_impurity(mu=0.0, T=0.1)
        self.assertEqual(frag.solver.data_type, np.dtype(np.float64))


if __name__ == '__main__':
    unittest.main()
