#######################################################
# Finite temperature density matrix fitting routines
# for thermal ghost Gutzwiller
# Author: Samuele Giuli
# Email:  samuele.giuli@gmail.com
#######################################################

import numpy as np
from scipy.optimize import root, least_squares, brentq, minimize

# Parameter directions per dF_spectral stack in jacobian_LR / jacobian_LcD: bounds their work arrays to
# ~6 x 16 JACOBIAN_CHUNK (2n)^2 bytes, whatever the number of parameters.
JACOBIAN_CHUNK = 256


# -------------------------
# Build H
# -------------------------
def build_H(Lambda, Lambda_c, D, R):
    H11 = Lambda
    H12 = R @ D.T
    H21 = H12.T.conj()
    H22 = -Lambda_c
    return np.block([[H11, H12],
                     [H21, H22]])

# -------------------------------------------------
# Scalar Fermi-like function (stable)
# -------------------------------------------------
def fermi(eps, beta):
    # avoids overflow using where since else is zero
    return np.where(
        beta * eps > 0,
        np.exp(-beta * eps) / (1 + np.exp(-beta * eps)),
        1 / (1 + np.exp(beta * eps))
    )

def fermi_prime(eps, beta):
    f = fermi(eps, beta)
    return -beta * f * (1 - f)

# -------------------------------------------------
# F(H) via diagonalization
# -------------------------------------------------
def F_of_H(H, beta):
    eps, U = np.linalg.eigh(H)
    f = fermi(eps, beta)
    return (U * f) @ U.conj().T

# -------------------------------------------------
# Frechet derivative dF(H)[dH]
# -------------------------------------------------
def fermi_divided_differences(H, beta, tol=1e-10):
    r"""
    Eigendecomposition H = V diag(e) V^dag, the occupations f(e) and the divided differences G of dF_spectral.

    Called by dF_spectral, and by jacobian_LR and jacobian_LcD, which pass the result to every dF_spectral call
    of their direction chunks and build F = V diag(f) V^dag from it, so a Jacobian needs one eigh of H.

    Return:
        (e, V, f, G), G_ij = (f(e_i) - f(e_j)) / (e_i - e_j) if |e_i - e_j| > tol, else (f'(e_i) + f'(e_j)) / 2.
    """
    # 1. H = V diag(e) V^dag
    eps, U = np.linalg.eigh(H)
    f = fermi(eps, beta)
    fp = fermi_prime(eps, beta)

    # 2. G_ij = (f_i - f_j) / (e_i - e_j), or (f'_i + f'_j) / 2 for |e_i - e_j| <= tol
    eps_i = eps[:, None]
    eps_j = eps[None, :]
    G = 0.5*(fp.reshape((1,-1)) + fp.reshape((-1,1)) )
    mask = np.abs(eps_i - eps_j) > tol
    G[mask] = (f[:, None] - f[None, :])[mask] / (eps_i - eps_j)[mask]
    return eps, U, f, G

def dF_spectral(H, dH, beta, tol=1e-10, spectral=None):
    r"""
    Frechet derivative of F(H) = f(H), f the Fermi function, along dH (Daleckii-Krein formula).

    With H = V diag(e) V^dag,

        dF(H)[dH] = V (G o (V^dag dH V)) V^dag,
        G_ij = (f(e_i) - f(e_j)) / (e_i - e_j)    if |e_i - e_j| > tol,
        G_ij = (f'(e_i) + f'(e_j)) / 2           otherwise (the limit e_j -> e_i),

    with o the elementwise product. dH is one (m, m) matrix or a stack (N, m, m) of directions; the stack shares
    one eigh of H. spectral = fermi_divided_differences(H, beta, tol) if given, else computed here: jacobian_LR and
    jacobian_LcD pass it, so that their chunks of unit parameter directions share one eigh.
    """
    # 1-2. H = V diag(e) V^dag and G
    if spectral is None:
        spectral = fermi_divided_differences(H, beta, tol)
    eps, U, f, G = spectral

    # 3. dH~ = V^dag dH V
    dH_tilde = U.conj().T @ dH @ U

    # 4. dF = V (G o dH~) V^dag
    dF_tilde = G * dH_tilde
    return U @ dF_tilde @ U.conj().T


# ============================================================
# Packing / unpacking parameters
# It works for both Lambda+R and Lambda_c+D
# ============================================================
def pack_params(Lambda, R):
    '''
    Routine to pack the parameters Lambda and R into a single vector for least-squares fitting.
    Its inverse is unpack_params.
    '''
    n = R.shape[0]
    p = R.shape[1]

    iu = np.triu_indices(n)
    iu_strict = np.triu_indices(n, k=1)

    Lam_re = Lambda.real[iu]
    Lam_im = Lambda.imag[iu_strict]

    return np.concatenate([
        Lam_re,
        Lam_im,
        R.real.reshape(-1),
        R.imag.reshape(-1)
    ])

def unpack_params(x, n, p):
    '''
    Routine to unpack the parameters Lambda and R from a single vector for least-squares fitting.
    Its inverse is pack_params.
    '''
    iu = np.triu_indices(n)
    iu_strict = np.triu_indices(n, k=1)

    n_re = len(iu[0])
    n_im = len(iu_strict[0])

    Lam_re = x[:n_re]
    Lam_im = x[n_re:n_re+n_im]

    offset = n_re + n_im

    R_re = x[offset:offset+n*p].reshape(n, p)
    offset += n*p
    R_im = x[offset:offset+n*p].reshape(n, p)

    # rebuild Hermitian Lambda
    Lambda = np.zeros((n, n), dtype=complex)
    Lambda.real[iu] = Lam_re.real
    Lambda.real[(iu[1], iu[0])] = Lam_re.real
    Lambda.imag[iu_strict] = Lam_im.real
    Lambda.imag[iu_strict[1],iu_strict[0]] = -Lam_im.real
    R = R_re + 1j * R_im
    return Lambda, R

def unit_perturbations(n, p):
    '''
    unpack_params(e_k, n, p) for every unit vector e_k of the packed parameters, as two stacks.

    Called by jacobian_LR and jacobian_LcD: x -> (A, B) is linear, so column k of the Jacobian is the derivative
    along (dA_k, dB_k) = unpack_params(e_k), and the stacks let all columns share one eigh (dF_spectral).

    Return:
        dA: (N, n, n) complex, Hermitian unit matrices: E_ii, E_ij + E_ji (i < j), then i (E_ij - E_ji) (i < j).
        dB: (N, n, p) complex, the unit matrices E_ab, then i E_ab. N = n^2 + 2 n p, in the order of pack_params.
    '''
    iu = np.triu_indices(n)
    iu_strict = np.triu_indices(n, k=1)
    n_re, n_im = len(iu[0]), len(iu_strict[0])
    N = n_re + n_im + 2 * n * p
    dA = np.zeros((N, n, n), dtype=complex)
    dB = np.zeros((N, n * p), dtype=complex)
    # 1. Re A_ij (i <= j): dA = E_ij + E_ji (E_ii on the diagonal)
    k = np.arange(n_re)
    dA[k, iu[0], iu[1]] = 1.0
    dA[k, iu[1], iu[0]] = 1.0
    # 2. Im A_ij (i < j): dA = i (E_ij - E_ji)
    k = n_re + np.arange(n_im)
    dA[k, iu_strict[0], iu_strict[1]] = 1j
    dA[k, iu_strict[1], iu_strict[0]] = -1j
    # 3. Re B_ab, then Im B_ab: dB = E_ab, i E_ab
    offset = n_re + n_im
    dB[offset:offset + n * p] = np.eye(n * p)
    dB[offset + n * p:] = 1j * np.eye(n * p)
    return dA, dB.reshape(N, n, p)


# ============================================================
# Residual
# ============================================================
def residual_LR(x, beta, Lambda_c, D, F22_target, RTF12_target):
    '''
    Residual for Lambda+R fit.
    '''
    n, p = D.shape
    Lambda, R = unpack_params(x, n, p)

    H = build_H(Lambda, Lambda_c, D, R)
    F = F_of_H(H, beta).T

    F22 = F[n:, n:]
    F12 = F[:n, n:]
    RTF12 = R.T@F12

    iu = np.triu_indices(n)
    iu_strict = np.triu_indices(n, k=1)

    # Standard Physical Residual
    res = np.concatenate([
        (F22.real[iu] - F22_target.real[iu]),
        (F22.imag[iu_strict] - F22_target.imag[iu_strict]),
        (RTF12.real - RTF12_target.real).ravel(),
        (RTF12.imag - RTF12_target.imag).ravel()
    ])
    return res

def residual_LcD(x, beta, Lambda, R, F11_target, F12D_target):
    '''
    Residual for Lambda_c+D fit
    '''
    n, p = R.shape
    Lambda_c, D = unpack_params(x, n, p)

    H = build_H(Lambda, Lambda_c, D, R)
    F = F_of_H(H, beta).T

    F11 = F[:n, :n]
    F12 = F[:n, n:]
    F12D = F12@D

    iu = np.triu_indices(n)
    iu_strict = np.triu_indices(n, k=1)

    # Standard Physical Residual
    res = np.concatenate([
        (F11.real[iu] - F11_target.real[iu]),
        (F11.imag[iu_strict] - F11_target.imag[iu_strict]),
        (F12D.real - F12D_target.real).ravel(),
        (F12D.imag - F12D_target.imag).ravel()
    ])
    return res


# ============================================================
# Jacobian via Frechet derivative
# ============================================================
def jacobian_LR(x, beta, Lambda_c, D, F22_target, RTF12_target):
    r'''
    Jacobian of residual_LR with respect to the packed (Lambda, R), F and all columns from one eigh of H.

    Called by jacobian_LR_movement (the least-squares fit of update_self_energy_thermal_penalty). With
    F = f(H)^T, H = build_H(Lambda, Lambda_c, D, R), the column of the unit direction (dLambda, dR) is

        dH = [[dLambda, dR D^T], [(dR D^T)^dag, 0]],
        dF = (dF_spectral(H, dH))^T,
        d(F22) = dF[n:, n:],   d(R^T F12) = R^T dF[:n, n:] + dR^T F[:n, n:],

    reduced as in residual_LR (Hermitian F22: Re upper triangle, Im strict upper triangle; R^T F12 full). The
    directions run in chunks of JACOBIAN_CHUNK, so the work arrays do not grow with the number of parameters.
    '''
    n, p = D.shape
    Lambda, R = unpack_params(x, n, p)
    iu = np.triu_indices(n)
    iu_strict = np.triu_indices(n, k=1)

    # 1. H = V diag(e) V^dag, F = (V diag(f) V^dag)^T (= F_of_H(H, beta).T)
    H = build_H(Lambda, Lambda_c, D, R)
    spectral = fermi_divided_differences(H, beta)
    _, U, f, _ = spectral
    F = ((U * f) @ U.conj().T).T
    F12 = F[:n, n:]

    dLambda, dR = unit_perturbations(n, p)
    N = len(dLambda)
    J_T = np.empty((N, len(iu[0]) + len(iu_strict[0]) + 2 * n * p))
    # Unit directions k in chunks of JACOBIAN_CHUNK, which bounds the dH and dF stacks
    for k_start in range(0, N, JACOBIAN_CHUNK):
        k_range = slice(k_start, min(k_start + JACOBIAN_CHUNK, N))
        dLambda_chunk, dR_chunk = dLambda[k_range], dR[k_range]

        # 2. dH_k = [[dLambda_k, dR_k D^T], [(dR_k D^T)^dag, 0]]
        dH = np.zeros((len(dLambda_chunk), 2 * n, 2 * n), dtype=complex)
        dH[:, :n, :n] = dLambda_chunk
        dH12 = dR_chunk @ D.T
        dH[:, :n, n:] = dH12
        dH[:, n:, :n] = dH12.conj().transpose(0, 2, 1)

        # 3. dF_k = (dF_spectral(H, dH_k))^T
        dF = dF_spectral(H, dH, beta, spectral=spectral).transpose(0, 2, 1)

        # 4. d(F22)_k = dF_k[n:, n:],  d(R^T F12)_k = R^T dF_k[:n, n:] + dR_k^T F12
        dF22 = dF[:, n:, n:]
        dRTF12 = R.T @ dF[:, :n, n:] + dR_chunk.transpose(0, 2, 1) @ F12

        # 5. rows in the order of residual_LR, one column per k
        J_T[k_range] = np.concatenate([
            dF22.real[:, iu[0], iu[1]],
            dF22.imag[:, iu_strict[0], iu_strict[1]],
            dRTF12.real.reshape(len(dLambda_chunk), -1),
            dRTF12.imag.reshape(len(dLambda_chunk), -1)
        ], axis=1)
    return J_T.T

def jacobian_LcD(x, beta, Lambda, R, F11_target, F12D_target):
    r'''
    Jacobian of residual_LcD with respect to the packed (Lambda_c, D), F and all columns from one eigh of H.

    Called by jacobian_LcD_movement (the least-squares fit of update_hybridization_thermal_penalty). With
    F = f(H)^T, H = build_H(Lambda, Lambda_c, D, R), the column of the unit direction (dLambda_c, dD) is

        dH = [[0, R dD^T], [(R dD^T)^dag, -dLambda_c]],
        dF = (dF_spectral(H, dH))^T,
        d(F11) = dF[:n, :n],   d(F12 D) = dF[:n, n:] D + F[:n, n:] dD,

    reduced as in residual_LcD (Hermitian F11: Re upper triangle, Im strict upper triangle; F12 D full). The
    directions run in chunks of JACOBIAN_CHUNK, so the work arrays do not grow with the number of parameters.
    '''
    n, p = R.shape
    Lambda_c, D = unpack_params(x, n, p)
    iu = np.triu_indices(n)
    iu_strict = np.triu_indices(n, k=1)

    # 1. H = V diag(e) V^dag, F = (V diag(f) V^dag)^T (= F_of_H(H, beta).T)
    H = build_H(Lambda, Lambda_c, D, R)
    spectral = fermi_divided_differences(H, beta)
    _, U, f, _ = spectral
    F = ((U * f) @ U.conj().T).T
    F12 = F[:n, n:]

    dLambda_c, dD = unit_perturbations(n, p)
    N = len(dLambda_c)
    J_T = np.empty((N, len(iu[0]) + len(iu_strict[0]) + 2 * n * p))
    # Unit directions k in chunks of JACOBIAN_CHUNK, which bounds the dH and dF stacks
    for k_start in range(0, N, JACOBIAN_CHUNK):
        k_range = slice(k_start, min(k_start + JACOBIAN_CHUNK, N))
        dLambda_c_chunk, dD_chunk = dLambda_c[k_range], dD[k_range]

        # 2. dH_k = [[0, R dD_k^T], [(R dD_k^T)^dag, -dLambda_c_k]]
        dH = np.zeros((len(dLambda_c_chunk), 2 * n, 2 * n), dtype=complex)
        dH[:, n:, n:] = -dLambda_c_chunk
        dH12 = R @ dD_chunk.transpose(0, 2, 1)
        dH[:, :n, n:] = dH12
        dH[:, n:, :n] = dH12.conj().transpose(0, 2, 1)

        # 3. dF_k = (dF_spectral(H, dH_k))^T
        dF = dF_spectral(H, dH, beta, spectral=spectral).transpose(0, 2, 1)

        # 4. d(F11)_k = dF_k[:n, :n],  d(F12 D)_k = dF_k[:n, n:] D + F12 dD_k
        dF11 = dF[:, :n, :n]
        dF12D = dF[:, :n, n:] @ D + F12 @ dD_chunk

        # 5. rows in the order of residual_LcD, one column per k
        J_T[k_range] = np.concatenate([
            dF11.real[:, iu[0], iu[1]],
            dF11.imag[:, iu_strict[0], iu_strict[1]],
            dF12D.real.reshape(len(dLambda_c_chunk), -1),
            dF12D.imag.reshape(len(dLambda_c_chunk), -1)
        ], axis=1)
    return J_T.T


# ---------- Movement-penalized residuals / jacobians for Lambda+R -----------
def residual_LR_movement(x, beta, Lambda_c, D, F22_target, RTF12_target, x0, alpha=1e-5):
    """Original residuals + movement-penalty residuals sqrt(2*alpha)*(x-x0)."""
    r = residual_LR(x, beta, Lambda_c, D, F22_target, RTF12_target)
    # movement residuals
    if alpha <= 0.0:
        return r
    factor = np.sqrt(2.0 * alpha)
    r_move = factor * (x - x0)
    return np.concatenate([r, r_move])

def jacobian_LR_movement(x, beta, Lambda_c, D, F22_target, RTF12_target, x0, alpha=1e-5):
    """Stack original Jacobian with movement-penalty rows: factor * I."""
    J = jacobian_LR(x, beta, Lambda_c, D, F22_target, RTF12_target)  # shape (m, N)
    if alpha <= 0.0:
        return J
    N = x.size
    factor = np.sqrt(2.0 * alpha)
    # create extra rows: factor * identity (N rows x N cols)
    Jpen = factor * np.eye(N, dtype=float)
    # stack (original m x N) with (N x N) => (m+N, N)
    return np.vstack([J, Jpen])

def solve_F_dF_LR_with_movement(beta, Lambda_c, D, Lambda0, R0, F22_target, RTF12_target,
                                alpha=1e-5, use_analytic_jac=True, max_nfev=200, verbose=0, tol=1e-9):
    """
    Least-squares solve with quadratic movement penalty around starting x0.
    alpha: penalty strength for sum_j (x_j-x0_j)^2 in objective.
    If use_analytic_jac True, provides jacobian via jacobian_LR_movement to least_squares.
    """
    x0 = pack_params(Lambda0, R0)
    if use_analytic_jac:
        sol = least_squares(
            residual_LR_movement,
            x0,
            jac=jacobian_LR_movement,
            args=(beta, Lambda_c, D, F22_target, RTF12_target, x0, alpha),
            method="trf",
            x_scale="jac",
            max_nfev=max_nfev,
            ftol=tol,
            xtol=tol,
            gtol=tol,
            verbose=verbose
        )
    else:
        sol = least_squares(
            residual_LR_movement,
            x0,
            jac='2-point',
            args=(beta, Lambda_c, D, F22_target, RTF12_target, x0, alpha),
            method="trf",
            x_scale="jac",
            max_nfev=max_nfev,
            ftol=tol,
            xtol=tol,
            gtol=tol,
            verbose=verbose
        )
    Lambda_sol, R_sol = unpack_params(sol.x, Lambda0.shape[0], R0.shape[1])
    return sol, Lambda_sol, R_sol

def update_self_energy_thermal_penalty(Lambda0, R0, Lambda_c, D, F22_target, RTF12_target,
                             beta=200, alpha=1e-5, method="dF", verbose=0, tol=1e-9):
    """
    Replacement of new_self_energy that includes movement penalty alpha * ||x-x0||^2.
    method is passed to choose between analytic/numeric jacobian inside the solver.
    verbose: passed to least_squares (0=silent, 1=final, 2=per-iteration).
    """
    sol, Lambda_sol, R_sol = solve_F_dF_LR_with_movement(
        beta, Lambda_c, D, Lambda0, R0, F22_target, RTF12_target,
        alpha=alpha, use_analytic_jac=(method=="dF"), verbose=verbose, tol=tol
    )
    return Lambda_sol, R_sol


# ---------- Movement-penalized residuals / jacobians for Lambda_c + D -----------
def residual_LcD_movement(x, beta, Lambda, R, F11_target, F12D_target, x0, alpha=1e-5):
    r = residual_LcD(x, beta, Lambda, R, F11_target, F12D_target)
    if alpha <= 0.0:
        return r
    factor = np.sqrt(2.0 * alpha)
    r_move = factor * (x - x0)
    return np.concatenate([r, r_move])

def jacobian_LcD_movement(x, beta, Lambda, R, F11_target, F12D_target, x0, alpha=1e-5):
    J = jacobian_LcD(x, beta, Lambda, R, F11_target, F12D_target)  # (m, N)
    if alpha <= 0.0:
        return J
    N = x.size
    factor = np.sqrt(2.0 * alpha)
    Jpen = factor * np.eye(N, dtype=float)
    return np.vstack([J, Jpen])

def solve_F_dF_LcD_with_movement(beta, Lambda, R, Lambda_c0, D0, F11_target, F12D_target,
                                 alpha=1e-5, use_analytic_jac=True, max_nfev=200, verbose=0, tol=1e-9):
    x0 = pack_params(Lambda_c0, D0)
    if use_analytic_jac:
        sol = least_squares(
            residual_LcD_movement,
            x0,
            jac=jacobian_LcD_movement,
            args=(beta, Lambda, R, F11_target, F12D_target, x0, alpha),
            method="trf",
            x_scale="jac",
            max_nfev=max_nfev,
            ftol=tol,
            xtol=tol,
            gtol=tol,
            verbose=verbose
        )
    else:
        sol = least_squares(
            residual_LcD_movement,
            x0,
            jac='2-point',
            args=(beta, Lambda, R, F11_target, F12D_target, x0, alpha),
            method="trf",
            x_scale="jac",
            max_nfev=max_nfev,
            ftol=tol,
            xtol=tol,
            gtol=tol,
            verbose=verbose
        )
    Lambda_c_sol, D_sol = unpack_params(sol.x, D0.shape[0], D0.shape[1])
    return sol, Lambda_c_sol, D_sol

def update_hybridization_thermal_penalty(Lambda_c0, D0, Lambda, R, F11_target, F12D_target,
                               beta=200, alpha=1e-5, method="dF", verbose=0, tol=1e-9):
    '''
    Function to update the hybridization function parameters D and Lambda_c.
    verbose: passed to least_squares (0=silent, 1=final, 2=per-iteration).
    '''
    sol, Lambda_c_sol, D_sol = solve_F_dF_LcD_with_movement(
        beta, Lambda, R, Lambda_c0, D0, F11_target, F12D_target,
        alpha=alpha, use_analytic_jac=(method=="dF"), verbose=verbose, tol=tol
    )
    return Lambda_c_sol, D_sol

