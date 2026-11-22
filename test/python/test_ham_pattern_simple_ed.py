'''
SimpleED.build_Hemb forms each sector's H on a fixed sparse pattern, its
one-body data as P_s @ h1e.ravel(). It must give the matrix of the direct
construction, sum_ij h1e[i,j] c^dag_i c_j + Htwo, and follow h1e from one
build to the next while the pattern is reused.
'''
import unittest
import numpy as np

from gem.solvers.simple_ed import SimpleED


def _random_embedding(norb, nimp, rng, complex_=True):
    '''Random Hermitian h1e (all (i, j), so terms leaving a sector are present
    too) split into eloc, D, Lambda_c, and a random V2E on the impurity.'''
    h = rng.normal(size=(norb, norb))
    V2E = rng.normal(size=(nimp,) * 4)
    if complex_:
        h = h + 1j * rng.normal(size=(norb, norb))
        V2E = V2E + 1j * rng.normal(size=(nimp,) * 4)
    h = 0.5 * (h + h.conj().T)
    h[0, 1] = h[1, 0] = 1e-9   # below the 1e-8 truncation of the one-body terms
    return h[:nimp, :nimp], h[nimp:, :nimp].conj(), -h[nimp:, nimp:], V2E


class TestHamPattern(unittest.TestCase):

    norb, nimp = 8, 4

    def _direct_ham(self, ed, s, V2E):
        '''H of local sector s built term by term.'''
        return (ed._build_one_body(ed.h1e, ed.hsize_list[s], ed.denmat_op_list[s])
                + ed._build_two_body(V2E, ed.basis_list[s], ed.hsize_list[s]))

    def _check(self, dtype, complex_):
        rng = np.random.default_rng(3)
        ed = SimpleED(self.norb, use_Ntot=True, use_Sz=True, dtype=dtype,
                      solver_params=dict(use_mpi=False))
        eloc, D, Lambdac, V2E = _random_embedding(self.norb, self.nimp, rng, complex_)
        if not complex_:
            V2E = V2E.real
        ed.build_Hemb(D, eloc, Lambdac, V2E)
        big = [s for s in range(ed.nloc) if ed.hsize_list[s] > 1]
        self.assertGreater(len(big), 5)
        for s in big:
            self.assertEqual(ed.Ham_list[s].dtype, ed.data_type)
            # Htwo is stored on the pattern of H, sharing its index arrays
            self.assertTrue(np.shares_memory(ed.Htwo_list[s].indices, ed.Ham_list[s].indices))
            diff = abs(ed.Ham_list[s] - self._direct_ham(ed, s, V2E)).max()
            self.assertLess(diff, 1e-13)

        # second build, h1e changed and V2E not: the pattern is reused
        old = [ed.Ham_list[s].copy() for s in big]
        pattern = ed.ham_pattern_list[big[0]]
        eloc2 = eloc + 0.3 * np.eye(self.nimp)
        D2 = D + 0.1 * rng.normal(size=D.shape)
        ed.build_Hemb(D2, eloc2, Lambdac, V2E)
        self.assertIs(ed.ham_pattern_list[big[0]], pattern)
        for s, H_old in zip(big, old):
            self.assertGreater(abs(ed.Ham_list[s] - H_old).max(), 1e-2)
            diff = abs(ed.Ham_list[s] - self._direct_ham(ed, s, V2E)).max()
            self.assertLess(diff, 1e-13)

    def test_complex(self):
        self._check(np.complex128, complex_=True)

    def test_real(self):
        self._check(np.float64, complex_=False)

    def test_spin_penalty(self):
        rng = np.random.default_rng(5)
        ed = SimpleED(self.norb, use_Ntot=True, use_Sz=True,
                      solver_params=dict(use_mpi=False))
        eloc, D, Lambdac, V2E = _random_embedding(self.norb, self.nimp, rng)
        ed.build_Hemb(D, eloc, Lambdac, V2E, spin_pen=0.7, sz_pen=0.3)
        for s in range(ed.nloc):
            Sz = ed.Sz_list[s]
            ref = (self._direct_ham(ed, s, V2E) + 0.7 * ed.S2_list[s]
                   + 0.3 * Sz.dot(Sz))
            self.assertLess(abs(ed.Ham_list[s] - ref).max(), 1e-13)

    def test_build_two_body_invalidates_pattern(self):
        '''The public build_two_body replaces Htwo by one off the pattern; the
        next build_Hemb must rebuild Htwo and the pattern, even for the V2E
        it cached.'''
        rng = np.random.default_rng(11)
        ed = SimpleED(self.norb, solver_params=dict(use_mpi=False))   # one sector
        eloc, D, Lambdac, V2E = _random_embedding(self.norb, self.nimp, rng)
        ed.build_Hemb(D, eloc, Lambdac, V2E)
        ed.build_two_body(2.0 * V2E)
        self.assertIsNone(ed.ham_pattern_list[0])
        ed.build_Hemb(D, eloc, Lambdac, V2E)
        self.assertLess(abs(ed.Ham - self._direct_ham(ed, 0, V2E)).max(), 1e-13)
        self.assertLess(abs(ed.Hone + ed.Htwo - ed.Ham).max(), 1e-13)


if __name__ == '__main__':
    unittest.main()
