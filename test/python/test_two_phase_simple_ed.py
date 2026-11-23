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
from gem.solvers.utilities.simple_ed_matvec import EmbeddingHamiltonian


def _random_embedding(norb=10, nimp=4, seed=7, degenerate=False, decoupled=False):
    '''Random spin-diagonal h1e (h_orb x 1_spin, so N and Sz are conserved) and
    a density-density V2E on the impurity spin-orbitals. With degenerate=True,
    h_orb commutes with the swap of orbitals 0 and 1, so the spectrum holds
    exactly degenerate clusters beyond the spin multiplets. With
    decoupled=True the last bath orbital has no hopping and a level 10
    above the others, so its occupation is conserved inside every (N, Sz)
    sector and the lowest state of the wrong occupation lies far outside the
    window.'''
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(norb // 2,) * 2) + 1j * rng.normal(size=(norb // 2,) * 2)
    A = A + A.conj().T
    if degenerate:
        P = np.eye(norb // 2)[[1, 0] + list(range(2, norb // 2))]
        A = 0.5 * (A + P @ A @ P)
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

    def _solve(self, sym=True, degenerate=False, decoupled=False, **params):
        eloc, D, Lambdac, V2E = _random_embedding(self.norb, degenerate=degenerate,
                                                  decoupled=decoupled)
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

    def test_degenerate_clusters(self):
        '''Exactly degenerate states in a sector: eigsh returns them with
        non-orthogonal vectors, the Rayleigh-Ritz step must make the density
        matrix that of the full diagonalisation.'''
        ref, dm_ref = self._solve(sym=False, degenerate=True)
        for matrix_free in (False, True):
            with self.subTest(matrix_free=matrix_free):
                ed, dm = self._solve(degenerate=True, num_eig=6, dense_cutoff=40,
                                     tol=0.0, matrix_free=matrix_free)
                self._assert_matches_full(ed, dm, ref, dm_ref)
                # exactly degenerate pairs among the kept states of eigsh sectors
                kept = [np.diff(e) for e, x in zip(ed.evals_list, ed._prof_sec)
                        if x[6] < x[1]]
                self.assertGreater(sum(np.sum(d < 1e-10) for d in kept), 2)

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

    def test_rank_deficient_eigsh(self):
        '''The capped eigsh (maxiter set) is made to return a duplicate of its
        lowest vector in place of the second one (rank-deficient V, as for a
        near-degenerate cluster cut by k), with the right eigenvalues. The
        solve must detect it from the QR of V, switch to the uncapped eigsh
        as on no convergence, and keep the states of the full
        diagonalisation.'''
        ref, dm_ref = self._solve(sym=False)
        spoiled, calls = set(), []

        def eigsh_duplicate(A, k=6, **kw):
            vals, vecs = eigsh(A, k=k, **kw)
            calls.append((A.shape[0], k, kw.get('maxiter')))
            if kw.get('maxiter') is not None:
                spoiled.add(A.shape[0])
                vecs = vecs.copy()
                vecs[:, 1] = vecs[:, 0]
            return vals, vecs

        for matrix_free in (False, True):
            with self.subTest(matrix_free=matrix_free):
                spoiled.clear(); calls.clear()
                with mock.patch.object(simple_ed, 'eigsh', eigsh_duplicate):
                    ed, dm = self._solve(num_eig=6, dense_cutoff=40,
                                         tol=0.0, matrix_free=matrix_free)
                self.assertTrue(spoiled)
                # each spoiled sector was solved again with the uncapped eigsh
                for dim in spoiled:
                    self.assertIn(None, [c[2] for c in calls if c[0] == dim and c[1] >= 2])
                self._assert_matches_full(ed, dm, ref, dm_ref)

    def test_start_k_at_gap(self):
        '''The last eigsh spectrum E_1..E_K of a sector sets the next starting
        k: the first cut E_k | E_{k+1} with k >= kept + 1 in a gap
        (E_{k+1} - E_k > GAP_START_MIN = 1e-2), else k = K. Give every other
        eigsh sector a last spectrum with a cluster (splittings 1e-6) on
        kept + 1 .. kept + 3 and a gap (1.0) after it, so the cut
        kept + 1 | kept + 2 lies inside the cluster and k must be kept + 3;
        the other sectors get the cluster alone (K = kept + 3, no gap), so
        k must be K. Each first eigsh must run at that k uncapped, and the
        kept states stay those of the full diagonalisation.'''
        ref, dm_ref = self._solve(sym=False)
        calls = []

        def eigsh_logged(A, k=6, **kw):
            calls.append((id(A), k, kw.get('maxiter')))
            return eigsh(A, k=k, **kw)

        for matrix_free in (False, True):
            with self.subTest(matrix_free=matrix_free):
                ed, _ = self._solve(num_eig=6, dense_cutoff=40, tol=0.0,
                                    matrix_free=matrix_free)
                expected = {}
                eig_secs = [s for s, x in enumerate(ed._prof_sec) if x[5] and x[6] < x[1]]
                for i, s in enumerate(eig_secs):
                    k0 = max(2, ed.prev_nkept_list[s] + 1)
                    E_low = np.arange(k0 - 1, dtype=float)
                    cluster = k0 + np.array([0.0, 1e-6, 2e-6])
                    after = cluster + 1.0 if i % 2 == 0 else []
                    ed.prev_evals_list[s] = np.concatenate([E_low, cluster, after])
                    expected[id(ed.Ham_list[s])] = k0 + 2
                self.assertGreater(len(expected), 3)
                calls.clear()
                with mock.patch.object(simple_ed, 'eigsh', eigsh_logged):
                    ed.solve_Hemb(T=self.T)
                first = {}
                for a, k, maxiter in calls:
                    if k >= 2:
                        first.setdefault(a, (k, maxiter))
                for a in expected:
                    self.assertIn(a, first)
                self.assertEqual({a: first[a] for a in expected},
                                 {a: (k, None) for a, k in expected.items()})
                self._assert_matches_full(ed, ed.calc_density_matrix(), ref, dm_ref)

    def test_matrix_free_window_too_large(self):
        '''Matrix-free, a window holding half a sector or more raises instead
        of densifying the sector.'''
        eloc, D, Lambdac, V2E = _random_embedding(self.norb)
        ed = SimpleED(self.norb, use_Ntot=True, use_Sz=True,
                      solver_params=dict(use_mpi=False, bw_cutoff=1e-300, num_eig=6,
                                         dense_cutoff=40, matrix_free=True))
        ed.build_Hemb(D, eloc, Lambdac, V2E)
        with self.assertRaisesRegex(ValueError, 'matrix_free'):
            ed.solve_Hemb(T=50.0)

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


@unittest.skipIf(simple_ed.primme is None, 'primme is not installed')
class TestTwoPhaseSolvePrimme(TestTwoPhaseSolve):
    '''The cases above with the eigsh sectors of phase 3 solved by PRIMME
    (eigensolver='primme'), at primme_tol 1e-12 so that the 1e-10 comparisons
    with the full diagonalisation resolve it; the kept vectors must be
    orthonormal. The eigsh-specific cases (capped fallback, rank-deficient
    output, gap start) do not reach eigsh here and are skipped.'''

    def _solve(self, sym=True, degenerate=False, decoupled=False, **params):
        return super()._solve(sym, degenerate, decoupled,
                              **dict(dict(eigensolver='primme', primme_tol=1e-12), **params))

    def _assert_matches_full(self, ed, dm, ref, dm_ref):
        super()._assert_matches_full(ed, dm, ref, dm_ref)
        for V in ed.evecs_list:
            np.testing.assert_allclose(V.conj().T @ V, np.eye(V.shape[1]), atol=1e-12)

    def test_capped_eigsh_fallback(self):
        self.skipTest('eigsh path only')

    def test_rank_deficient_eigsh(self):
        self.skipTest('eigsh path only')

    def test_start_k_at_gap(self):
        self.skipTest('eigsh path only')

    def test_primme_path_and_warm_start(self):
        '''Phase 3 calls PRIMME and never eigsh with k >= 2; the second
        solve starts each sector from the kept vectors of the first, plus
        PRIMME_N_RANDOM random columns.'''
        ref, dm_ref = self._solve(sym=False)
        calls, eigsh_k = [], []

        def primme_logged(A, k, **kw):
            calls.append((id(A), k, kw['v0']))
            return primme_eigsh(A, k, **kw)

        def eigsh_logged(A, k=6, **kw):
            eigsh_k.append(k)
            return eigsh(A, k=k, **kw)

        primme_eigsh = simple_ed.primme.eigsh
        for matrix_free in (False, True):
            with self.subTest(matrix_free=matrix_free):
                calls.clear(); eigsh_k.clear()
                with mock.patch.object(simple_ed.primme, 'eigsh', primme_logged), \
                     mock.patch.object(simple_ed, 'eigsh', eigsh_logged):
                    ed, dm = self._solve(num_eig=6, dense_cutoff=40, matrix_free=matrix_free)
                    self._assert_matches_full(ed, dm, ref, dm_ref)
                    self.assertTrue(calls)
                    self.assertEqual(set(eigsh_k), {1})
                    prev = {id(ed.Ham_list[s]): ed.evecs_list[s].copy()
                            for s, x in enumerate(ed._prof_sec) if x[5] and x[6] < x[1]}
                    calls.clear()
                    ed.solve_Hemb(T=self.T)
                    self._assert_matches_full(ed, ed.calc_density_matrix(), ref, dm_ref)
                first = {}
                for a, k, X0 in calls:
                    first.setdefault(a, X0)
                for a, V in prev.items():
                    X0 = first[a]
                    self.assertEqual(X0.shape[1], V.shape[1] + simple_ed.PRIMME_N_RANDOM)
                    # V lies in span(X0) up to the admixture
                    resid = np.linalg.norm(V - X0 @ (X0.conj().T @ V), axis=0)
                    self.assertLess(resid.max(), 2 * simple_ed.PRIMME_WARM_ADMIXTURE)

    def test_primme_error_falls_back_to_eigsh(self):
        '''A PrimmeError in a sector sends that sector to the eigsh branch,
        whose states must still be those of the full diagonalisation.'''
        ref, dm_ref = self._solve(sym=False)
        failed = set()
        primme_eigsh = simple_ed.primme.eigsh

        def primme_fails(A, k, **kw):
            if id(A) not in failed:
                failed.add(id(A))
                raise simple_ed.primme.PrimmeError(-3)
            return primme_eigsh(A, k, **kw)

        for matrix_free in (False, True):
            with self.subTest(matrix_free=matrix_free):
                failed.clear()
                with mock.patch.object(simple_ed.primme, 'eigsh', primme_fails), \
                     self.assertWarnsRegex(UserWarning, 'PRIMME failed'):
                    ed, dm = self._solve(num_eig=6, dense_cutoff=40, tol=0.0,
                                         matrix_free=matrix_free)
                self.assertGreater(len(failed), 2)
                self._assert_matches_full(ed, dm, ref, dm_ref)

    def test_primme_missing_runs_eigsh(self):
        '''Without primme, eigensolver='primme' warns and runs eigsh.'''
        ref, dm_ref = self._solve(sym=False)
        eigsh_k = []

        def eigsh_logged(A, k=6, **kw):
            eigsh_k.append(k)
            return eigsh(A, k=k, **kw)

        with mock.patch.object(simple_ed, 'primme', None), \
             mock.patch.object(simple_ed, 'eigsh', eigsh_logged), \
             self.assertWarnsRegex(UserWarning, 'primme is not installed'):
            ed, dm = self._solve(num_eig=6, dense_cutoff=40, tol=0.0)
        self.assertGreater(max(eigsh_k), 1)
        self._assert_matches_full(ed, dm, ref, dm_ref)

    def test_primme_warm_start_orthogonal_to_ground_state(self):
        '''As test_warm_start_orthogonal_to_ground_state, for PRIMME's warm
        start: with a conserved bath occupation n_b, set each sector's last
        kept vectors to random ones in the n_b subspace without the sector's
        lowest state, which then span (nearly) an invariant subspace. Without
        the random admixture to each vector PRIMME returns them, converged at
        once, in place of lower states (Tstates 72-74 of 80); with it, the
        states of the full diagonalisation.'''
        ref, dm_ref = self._solve(sym=False, decoupled=True)
        for matrix_free in (False, True):
            with self.subTest(matrix_free=matrix_free):
                ed, _ = self._solve(decoupled=True, num_eig=6, dense_cutoff=40,
                                    tol=0.0, matrix_free=matrix_free)
                rng = np.random.default_rng(1)
                n_set = 0
                for s, x in enumerate(ed._prof_sec):
                    if not (x[5] and x[6] < x[1]):   # PRIMME sectors only
                        continue
                    Ham_s = ed.Ham_list[s]
                    H = Ham_s.to_dense() if matrix_free else Ham_s.toarray()
                    gs = np.linalg.eigh(H)[1][:, 0]
                    for bit in range(self.norb):
                        occ = (ed.basis_list[s] >> bit) & 1
                        if np.any(H[np.ix_(occ == 0, occ == 1)]):
                            continue
                        other = occ != occ[np.argmax(np.abs(gs))]
                        self.assertLess(np.linalg.norm(gs[other]), 1e-12)
                        m = max(1, ed.prev_nkept_list[s])
                        ed.prev_kept_vecs_list[s] = np.linalg.qr(
                            rng.normal(size=(len(occ), m)) * other[:, None])[0]
                        n_set += 1
                        break
                self.assertGreater(n_set, 2)
                ed.solve_Hemb(T=self.T)
                self._assert_matches_full(ed, ed.calc_density_matrix(), ref, dm_ref)


class TestEmbeddingDiagonal(unittest.TestCase):

    def test_diagonal_matches_dense(self):
        '''EmbeddingHamiltonian.diagonal() (PRIMME's preconditioner,
        matrix-free) equals the diagonal of the dense H, in every sector.'''
        eloc, D, Lambdac, V2E = _random_embedding(10)
        ed = SimpleED(10, use_Ntot=True, use_Sz=True,
                      solver_params=dict(use_mpi=False, matrix_free=True))
        ed.build_Hemb(D, eloc, Lambdac, V2E)
        for Ham_s in ed.Ham_list:
            self.assertIsInstance(Ham_s, EmbeddingHamiltonian)
            np.testing.assert_allclose(Ham_s.diagonal(), np.diag(Ham_s.to_dense()), atol=1e-13)


if __name__ == '__main__':
    unittest.main()
