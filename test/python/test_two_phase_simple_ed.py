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
from unittest import mock
import numpy as np
from scipy.sparse.linalg import eigsh, ArpackNoConvergence

import gem.solvers.simple_ed as simple_ed
from gem.solvers.simple_ed import SimpleED


def _random_embedding(norb=10, nimp=4, seed=7, decoupled=False):
    '''Random spin-diagonal h1e (h_orb x 1_spin, so N and Sz are conserved) and
    a density-density V2E on the impurity spin-orbitals. With
    decoupled=True the last bath orbital has no hopping and a level 10
    above the others, so its occupation is conserved inside every (N, Sz)
    sector and the lowest state of the wrong occupation lies far outside the
    window.'''
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(norb // 2,) * 2) + 1j * rng.normal(size=(norb // 2,) * 2)
    A = A + A.conj().T
    if decoupled:
        A[-1, :-1] = A[:-1, -1] = 0.0
        A[-1, -1] = 25.0
    h = np.kron(0.4 * A, np.eye(2))
    V2E = np.zeros((nimp,) * 4)
    for i in range(nimp):
        for j in range(nimp):
            if i != j:
                V2E[i, i, j, j] = 2.0
    return h[:nimp, :nimp], h[nimp:, :nimp].conj(), -h[nimp:, nimp:], V2E


class TestTwoPhaseSolve(unittest.TestCase):

    T, bw_cutoff, norb = 0.3, 1e-6, 10

    def _solve(self, sym=True, decoupled=False, **params):
        eloc, D, Lambdac, V2E = _random_embedding(self.norb, decoupled=decoupled)
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

    def test_capped_eigsh_fallback(self):
        '''The capped eigsh (maxiter set) is made to fail once per sector, as
        it does when k cuts a cluster whose splittings are ~ tol |E|, so the
        path k += 8 with the large ncv runs. The kept states must still be
        those of the full diagonalisation.'''
        ref, dm_ref = self._solve(sym=False)
        failed = set()

        def eigsh_cap_fails(A, k=6, **kw):
            if kw.get('maxiter') is not None and A.shape[0] not in failed:
                failed.add(A.shape[0])
                raise ArpackNoConvergence('forced', np.empty(0), np.empty((A.shape[0], 0)))
            return eigsh(A, k=k, **kw)

        for matrix_free in (False, True):
            with self.subTest(matrix_free=matrix_free):
                failed.clear()
                with mock.patch.object(simple_ed, 'eigsh', eigsh_cap_fails):
                    ed, dm = self._solve(num_eig=6, dense_cutoff=40,
                                         tol=0.0, matrix_free=matrix_free)
                self.assertTrue(failed)
                self._assert_matches_full(ed, dm, ref, dm_ref)

    def test_warm_start_orthogonal_to_ground_state(self):
        '''Phase 1 at T>0 starts from the last solve's vector plus a random
        admixture. With a decoupled bath orbital, its occupation n_b is
        conserved in every sector, and a Krylov space started in the n_b
        subspace without the sector's lowest state stays there exactly. Set
        the last vector to a random one in that subspace: the random part
        must still lead eigsh to the true lowest state of every sector.'''
        ref, dm_ref = self._solve(sym=False, decoupled=True)
        for matrix_free in (False, True):
            with self.subTest(matrix_free=matrix_free):
                ed, _ = self._solve(decoupled=True, num_eig=6, dense_cutoff=40,
                                    tol=0.0, matrix_free=matrix_free)
                rng = np.random.default_rng(1)
                n_set = 0
                for s, Ham_s in enumerate(ed.Ham_list):
                    if ed.hsize_list[s] < 100:   # phase 1 uses eigh there
                        continue
                    H = Ham_s.to_dense() if matrix_free else Ham_s.toarray()
                    gs = np.linalg.eigh(H)[1][:, 0]
                    for bit in range(self.norb):
                        occ = (ed.basis_list[s] >> bit) & 1
                        if np.any(H[np.ix_(occ == 0, occ == 1)]):
                            continue
                        # conserved n_b: start in the subspace without the gs
                        other = occ != occ[np.argmax(np.abs(gs))]
                        self.assertLess(np.linalg.norm(gs[other]), 1e-12)
                        ed.prev_gs_list[s] = rng.normal(size=len(occ)) * other
                        n_set += 1
                        break
                self.assertGreater(n_set, 2)
                ed.solve_Hemb(T=self.T)
                self._assert_matches_full(ed, ed.calc_density_matrix(), ref, dm_ref)

if __name__ == '__main__':
    unittest.main()
