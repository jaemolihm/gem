'''
SimpleED at T>0 computes states only in the sectors whose lowest eigenvalue
lies within the Boltzmann window, with an ARPACK k grown until the window is
covered. It must keep exactly the states of a full diagonalisation of the
whole Fock space (no symmetry, one sector, always solved) truncated at the
same bw_cutoff.

The embedding is chosen so that the window holds several sectors with several
states each, some sectors are skipped, and the ARPACK sectors need k well
above its starting value of 2.
'''
import unittest
import numpy as np

from gem.solvers.simple_ed import SimpleED


def _random_embedding(norb=10, nimp=4, seed=7):
    '''Random spin-diagonal h1e (h_orb x 1_spin, so N and Sz are conserved) and
    a density-density V2E on the impurity spin-orbitals.'''
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(norb // 2,) * 2) + 1j * rng.normal(size=(norb // 2,) * 2)
    h = np.kron(0.4 * (A + A.conj().T), np.eye(2))
    V2E = np.zeros((nimp,) * 4)
    for i in range(nimp):
        for j in range(nimp):
            if i != j:
                V2E[i, i, j, j] = 2.0
    return h[:nimp, :nimp], h[nimp:, :nimp].conj(), -h[nimp:, nimp:], V2E


class TestTwoPhaseSolve(unittest.TestCase):

    T, bw_cutoff, norb = 0.3, 1e-6, 10

    def _solve(self, sym=True, **params):
        eloc, D, Lambdac, V2E = _random_embedding(self.norb)
        ed = SimpleED(self.norb, use_Ntot=sym, use_Sz=sym,
                      solver_params=dict(use_mpi=False, bw_cutoff=self.bw_cutoff, **params))
        ed.build_Hemb(D, eloc, Lambdac, V2E)
        ed.solve_Hemb(T=self.T)
        return ed, ed.calc_density_matrix().copy()

    def _assert_matches_full(self, ed, dm, ref, dm_ref):
        self.assertAlmostEqual(ed.gs_ene, ref.gs_ene, delta=1e-10)
        self.assertEqual(ed.Tstates, ref.Tstates)
        self.assertAlmostEqual(ed.Zpart, ref.Zpart, delta=1e-10)
        np.testing.assert_allclose(sorted(ed.bw_list), sorted(ref.bw_list), atol=1e-10)
        np.testing.assert_allclose(np.sort(np.concatenate(ed.evals_list)),
                                   ref.evals_list[0], atol=1e-10)
        np.testing.assert_allclose(dm, dm_ref, atol=1e-10)

    def test_matches_full_diagonalisation(self):
        ref, dm_ref = self._solve(sym=False)

        for matrix_free in (False, True):
            with self.subTest(matrix_free=matrix_free):
                ed, dm = self._solve(num_eig=6, dense_cutoff=40, tol=0.0,
                                     matrix_free=matrix_free)
                self._assert_matches_full(ed, dm, ref, dm_ref)
                # the case is non-trivial: several sectors with several states
                kept = [len(e) for e in ed.evals_list]
                self.assertGreater(sum(n > 1 for n in kept), 5)
                # [sector, dim, t1, t3, kept, needed, k]: sectors skipped, and
                # an ARPACK sector (k < dim) grown past num_eig=6 from k=2
                prof = ed._prof_sec
                self.assertGreater(sum(not x[5] for x in prof), 5)
                self.assertGreater(max(x[6] for x in prof if x[6] < x[1]), 6)

                # second solve: k starts from the states kept by the first
                ed.solve_Hemb(T=self.T)
                np.testing.assert_allclose(ed.calc_density_matrix(), dm_ref, atol=1e-10)
                self.assertAlmostEqual(ed.Zpart, ref.Zpart, delta=1e-10)


if __name__ == '__main__':
    unittest.main()
