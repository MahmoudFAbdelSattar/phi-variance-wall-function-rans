#!/usr/bin/env python3
"""
================================================================================
A Pointwise φ-Variance Wall Function with Calibrated RANS Inputs for
Turbulent Channel Flow

Supplementary Material

This code reproduces the results presented in the paper. It implements:
  1. Loading of DNS data for turbulent channel flow at Re_τ = 180, 395, 590
     (Moser, Kim & Mansour, 1999).
  2. Solution of the k-ε and Wilcox k-ω turbulence models for the same
     Reynolds numbers using a finite-volume channel solver with wall functions.
  3. Evaluation of the proposed φ-variance wall function at the first interior
     RANS node, with and without the correction parameter C_u.
  4. Calibration of C_u for each Re_τ, and assessment of a constant C_u.
  5. Generation of tables and figures (relative errors, optimal C_u vs Re_τ).

Directory structure:
    ./chan180/profiles/
    ./chan395/profiles/
    ./chan590/profiles/
    ./figures/          (created automatically)
    ./results/          (created automatically)

Requirements:
    numpy, scipy, matplotlib

Usage:
    Local run:
        Ensure the DNS folders (chan180, chan395, chan590) are present.
        Run: python main_analysis.py

    Google Colab:
        If the DNS folders are not present, you will be prompted to upload
        a compressed archive (chandata.tar.gz or .zip). Alternatively, run
        the separate helper script `colab_upload_data.py` first.

Outputs:
    figures/pointwise_error_vs_Cu.png
    figures/optimal_Cu_vs_Re.png
    results/pointwise_results.json

Author: [Author Name]
Affiliation: [Institution]
Date: 2025
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import os, tarfile, zipfile, json
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# 0. DATA PREPARATION (for Google Colab or local run)
# ============================================================
def prepare_data():
    """
    Ensure that the DNS data folders are present.
    If not, and if running in Google Colab, prompt for upload.
    Otherwise, raise a helpful error with instructions.
    """
    expected = [180, 395, 590]
    if all(os.path.isdir(f'chan{Re}') for Re in expected):
        print("DNS data folders found locally.")
        return

    # Detect if running in Google Colab
    try:
        import google.colab
        IN_COLAB = True
    except ImportError:
        IN_COLAB = False

    if IN_COLAB:
        from google.colab import files
        print("DNS data folders not found. Please upload chandata.tar.gz or .zip.")
        uploaded = files.upload()
        for fname in uploaded.keys():
            if fname.endswith('.tar.gz') or fname.endswith('.tgz'):
                with tarfile.open(fname, 'r:gz') as t:
                    t.extractall('.')
                os.remove(fname)
            elif fname.endswith('.zip'):
                with zipfile.ZipFile(fname, 'r') as z:
                    z.extractall('.')
                os.remove(fname)
            else:
                raise ValueError(f"Unsupported file format: {fname}")
        if not all(os.path.isdir(f'chan{Re}') for Re in expected):
            raise FileNotFoundError("Extraction failed or folders are missing.")
        print("DNS data successfully extracted.")
    else:
        raise FileNotFoundError(
            "DNS data folders (chan180, chan395, chan590) not found in the current directory.\n"
            "Please download the data from the official source (e.g., Moser, Kim & Mansour, 1999)\n"
            "and place the extracted folders in the working directory.\n"
            "Alternatively, run the separate helper script `colab_upload_data.py` in Google Colab."
        )

# ============================================================
# 1. DNS LOADER
# ============================================================
def load_dns(Re_tau):
    """
    Load DNS profiles for a given Re_tau.

    Parameters
    ----------
    Re_tau : int
        Friction Reynolds number (180, 395, or 590).

    Returns
    -------
    dict with keys: y_plus, uu (streamwise variance), U_plus, Re_tau.
    """
    folder = f'chan{Re_tau}/profiles'
    rs = np.loadtxt(os.path.join(folder, f'chan{Re_tau}.reystress'),
                    comments='#')
    means = np.loadtxt(os.path.join(folder, f'chan{Re_tau}.means'),
                       comments='#')
    # Column layout (verified from the dataset):
    #   rs:    0: y/h, 1: y+, 2: <u'^2>, 3: <v'^2>, 4: <w'^2>, 5: <u'v'>
    #   means: 0: y/h, 1: y+, 2: U+
    y_plus = rs[:, 1]
    uu     = rs[:, 2]
    U_plus = means[:, 2]
    mask = y_plus > 0.0  # exclude the wall point (y+=0)
    return {
        'y_plus': y_plus[mask],
        'uu': uu[mask],
        'U_plus': U_plus[mask],
        'Re_tau': Re_tau,
    }

# ============================================================
# 2. RANS SOLVERS
# ============================================================
def solve_k_epsilon(Re_tau, ny=200, n_iter=4000, tol=1e-6):
    """
    Solve the standard k-ε model for fully developed channel flow using
    a finite-volume method with wall functions.

    Parameters
    ----------
    Re_tau : int
        Friction Reynolds number.
    ny : int
        Number of interior cells.
    n_iter : int
        Maximum number of iterations.
    tol : float
        Convergence tolerance on the relative change in U.

    Returns
    -------
    dict containing profiles of y+, U+, k+, ν_t+, and dU/dy.
    """
    nu = 1.0 / Re_tau
    delta = 1.0
    kappa, E_wall = 0.41, 9.0
    c_mu, s_k, s_e = 0.09, 1.0, 1.3
    c1e, c2e = 1.44, 1.92
    ufU, ufk, ufe = 0.7, 0.5, 0.5

    # Grid generation (stretched near the wall)
    y_n = np.zeros(ny + 3)
    y_n[0], y_n[1] = 0.0, 0.1
    dy = (delta - 2*y_n[1]) / ny
    y_n[2] = 2*y_n[1] + 0.5*dy
    for i in range(3, ny+2):
        y_n[i] = y_n[i-1] + dy
    y_n[ny+2] = delta

    y_f = np.zeros(ny+2)
    y_f[0], y_f[1] = 0.0, 2*y_n[1]
    for i in range(2, ny+2):
        y_f[i] = y_f[i-1] + dy
    n = len(y_n)

    # Initialisation
    us = 1.0
    U = np.zeros(n)
    for i in range(1, n):
        yp = y_n[i]*us/nu
        U[i] = us * min(yp, np.log(E_wall*yp)/kappa) if yp > 0 else 0.0
    U[-1] = U[-2]
    k = np.full(n, us**2)
    eps = np.full(n, us**3/(kappa*y_n[-2]))
    eps[1] = us**3/(kappa*y_n[1])
    k[0] = 0.0
    k[1] = us**2 / np.sqrt(c_mu)
    k[-1] = k[-2]
    eps[0] = eps[1]
    eps[-1] = eps[-2]
    nu_t = np.zeros(n)
    nu_t_f = np.zeros(n-1)

    # Iterative solution
    for it in range(n_iter):
        Uo, ko, eo, us_old = U.copy(), k.copy(), eps.copy(), us
        # Eddy viscosity
        for i in range(1, n):
            nu_t[i] = min(c_mu * k[i]**2 / max(eps[i], 1e-12), 100.0*nu)
        nu_t[0] = 0.0
        # Eddy viscosity at faces (harmonic mean)
        for i in range(n-1):
            if i == 0:
                nu_t_f[i] = nu_t[0]
            elif i == n-2:
                nu_t_f[i] = nu_t[-1]
            else:
                d1 = y_f[i] - y_f[i-1]
                d2 = y_f[i+1] - y_f[i]
                den = d1 / max(nu_t[i], 1e-12) + d2 / max(nu_t[i+1], 1e-12)
                nu_t_f[i] = (d1 + d2) / den

        # Momentum equation
        Un = U.copy()
        for i in range(1, n-1):
            dV = y_f[i] - y_f[i-1]
            if i == 1:
                aN = (nu + nu_t_f[i]) / (y_n[i+1] - y_n[i])
                Un[i] = U[i+1] - (us_old**2 - dV) / aN
            elif i == n-2:
                aS = (nu + nu_t_f[i-1]) / (y_n[i] - y_n[i-1])
                Un[i] = (aS * U[i-1] + dV) / aS
            else:
                aN = (nu + nu_t_f[i]) / (y_n[i+1] - y_n[i])
                aS = (nu + nu_t_f[i-1]) / (y_n[i] - y_n[i-1])
                Un[i] = (aN * U[i+1] + aS * U[i-1] + dV) / (aN + aS)
        U = U + ufU * (Un - U)
        U[0] = 0.0
        U[-1] = U[-2]

        # Velocity gradient
        dU = np.zeros(n)
        dU[0] = (U[1] - U[0]) / y_n[1]
        for i in range(1, n-1):
            dU[i] = (U[i+1] - U[i-1]) / (y_n[i+1] - y_n[i-1])
        Pk = nu_t * dU**2
        Pk[0] = 0.0

        # Friction velocity (from wall function)
        arg = E_wall * us * y_n[1] / nu
        if arg > 1.0:
            usn = kappa * U[1] / np.log(arg)
            if usn > 0:
                us = 0.5 * us + 0.5 * usn

        # k equation
        kn = k.copy()
        for i in range(2, n-1):
            dV = y_f[i] - y_f[i-1]
            aN = (nu + nu_t_f[i] / s_k) / (y_n[i+1] - y_n[i])
            aS = (nu + nu_t_f[i-1] / s_k) / (y_n[i] - y_n[i-1])
            Sp = -eo[i] / max(ko[i], 1e-12) * dV
            Su = Pk[i] * dV
            aP = aN + aS - Sp
            if i == n-2:
                kn[i] = (aS * k[i-1] + Su) / aP
            else:
                kn[i] = (aN * k[i+1] + aS * k[i-1] + Su) / aP
        k = k + ufk * (kn - k)
        k[0] = 0.0
        k[1] = us**2 / np.sqrt(c_mu)
        k[-1] = k[-2]
        k = np.maximum(k, 1e-12)

        # ε equation
        en = eps.copy()
        for i in range(2, n-1):
            dV = y_f[i] - y_f[i-1]
            aN = (nu + nu_t_f[i] / s_e) / (y_n[i+1] - y_n[i])
            aS = (nu + nu_t_f[i-1] / s_e) / (y_n[i] - y_n[i-1])
            Sp = -c2e * eo[i] / max(ko[i], 1e-12) * dV
            Su = c1e * Pk[i] * eo[i] / max(ko[i], 1e-12) * dV
            aP = aN + aS - Sp
            if i == n-2:
                en[i] = (aS * eps[i-1] + Su) / aP
            else:
                en[i] = (aN * eps[i+1] + aS * eps[i-1] + Su) / aP
        eps = eps + ufe * (en - eps)
        eps[1] = us**3 / (kappa * y_n[1])
        eps[0] = eps[1]
        eps[-1] = eps[-2]
        eps = np.maximum(eps, 1e-12)

        # Convergence check
        ch = np.max(np.abs(U - Uo)) / (np.max(np.abs(U)) + 1e-10)
        if ch < tol and it > 200:
            break

    # Convert to wall units
    y_plus = y_n * us / nu
    U_plus = U / us
    k_plus = k / us**2
    nu_t_plus = nu_t / nu
    dU_dy = np.gradient(U_plus, y_plus)
    return {
        'y_plus': y_plus,
        'U_plus': U_plus,
        'k_plus': k_plus,
        'nu_t_plus': nu_t_plus,
        'dU_dy': dU_dy,
    }

def solve_k_omega_wilcox(Re_tau, ny=200, n_iter=4000, tol=1e-6):
    """
    Solve the Wilcox k-ω model for fully developed channel flow.

    This solver is included for completeness. The returned dictionary
    has the same keys as solve_k_epsilon.
    """
    nu = 1.0 / Re_tau
    delta = 1.0
    kappa, E_wall = 0.41, 9.0
    alpha = 5.0 / 9.0
    beta = 3.0 / 40.0
    bstar = 9.0 / 100.0
    sig_k = sig_w = 0.5
    ufU, ufk, ufw = 0.5, 0.4, 0.25

    # Grid (same as k-ε)
    y_n = np.zeros(ny + 3)
    y_n[0], y_n[1] = 0.0, 0.1
    dy = (delta - 2*y_n[1]) / ny
    y_n[2] = 2*y_n[1] + 0.5*dy
    for i in range(3, ny+2):
        y_n[i] = y_n[i-1] + dy
    y_n[ny+2] = delta
    y_f = np.zeros(ny+2)
    y_f[0], y_f[1] = 0.0, 2*y_n[1]
    for i in range(2, ny+2):
        y_f[i] = y_f[i-1] + dy
    n = len(y_n)

    # Initialisation
    us = 1.0
    U = np.zeros(n)
    for i in range(1, n):
        yp = y_n[i]*us/nu
        U[i] = us * min(yp, np.log(E_wall*yp)/kappa) if yp > 0 else 0.0
    U[-1] = U[-2]
    k = np.full(n, us**2 / np.sqrt(bstar))
    omega = np.full(n, us / (np.sqrt(bstar) * kappa * y_n[-2]))
    omega[1] = us / (np.sqrt(bstar) * kappa * y_n[1])
    k[0] = 0.0
    k[1] = us**2 / np.sqrt(bstar)
    k[-1] = k[-2]
    omega[0] = omega[1]
    omega[-1] = omega[-2]
    omega = np.clip(omega, 1e-3, 1e6)
    nu_t = np.zeros(n)
    nu_t_f = np.zeros(n-1)

    for it in range(n_iter):
        Uo, ko, wo, us_old = U.copy(), k.copy(), omega.copy(), us
        # Eddy viscosity
        for i in range(1, n):
            nu_t[i] = min(k[i] / max(omega[i], 1e-12), 100.0*nu)
        nu_t[0] = 0.0
        # Eddy viscosity at faces
        for i in range(n-1):
            if i == 0:
                nu_t_f[i] = nu_t[0]
            elif i == n-2:
                nu_t_f[i] = nu_t[-1]
            else:
                d1 = y_f[i] - y_f[i-1]
                d2 = y_f[i+1] - y_f[i]
                den = d1 / max(nu_t[i], 1e-12) + d2 / max(nu_t[i+1], 1e-12)
                nu_t_f[i] = (d1 + d2) / den

        # Momentum equation
        Un = U.copy()
        for i in range(1, n-1):
            dV = y_f[i] - y_f[i-1]
            if i == 1:
                aN = (nu + nu_t_f[i]) / (y_n[i+1] - y_n[i])
                Un[i] = U[i+1] - (us_old**2 - dV) / aN
            elif i == n-2:
                aS = (nu + nu_t_f[i-1]) / (y_n[i] - y_n[i-1])
                Un[i] = (aS * U[i-1] + dV) / aS
            else:
                aN = (nu + nu_t_f[i]) / (y_n[i+1] - y_n[i])
                aS = (nu + nu_t_f[i-1]) / (y_n[i] - y_n[i-1])
                Un[i] = (aN * U[i+1] + aS * U[i-1] + dV) / (aN + aS)
        U = U + ufU * (Un - U)
        U[0] = 0.0
        U[-1] = U[-2]

        # Velocity gradient
        dU = np.zeros(n)
        dU[0] = (U[1] - U[0]) / y_n[1]
        for i in range(1, n-1):
            dU[i] = (U[i+1] - U[i-1]) / (y_n[i+1] - y_n[i-1])
        Pk = nu_t * dU**2
        Pk[0] = 0.0

        # Friction velocity
        arg = E_wall * us * y_n[1] / nu
        if arg > 1.0:
            usn = kappa * U[1] / np.log(arg)
            if usn > 0:
                us = 0.5 * us + 0.5 * usn

        # k equation
        kn = k.copy()
        for i in range(2, n-1):
            dV = y_f[i] - y_f[i-1]
            aN = (nu + sig_k * nu_t_f[i]) / (y_n[i+1] - y_n[i])
            aS = (nu + sig_k * nu_t_f[i-1]) / (y_n[i] - y_n[i-1])
            Pk_lim = min(Pk[i], 10.0 * bstar * ko[i] * wo[i])
            Sp = -bstar * wo[i] * dV
            Su = Pk_lim * dV
            aP = aN + aS - Sp
            if i == n-2:
                kn[i] = (aS * k[i-1] + Su) / aP
            else:
                kn[i] = (aN * k[i+1] + aS * k[i-1] + Su) / aP
        k = k + ufk * (kn - k)
        k[0] = 0.0
        k[1] = us**2 / np.sqrt(bstar)
        k[-1] = k[-2]
        k = np.maximum(k, 1e-12)

        # omega equation
        wn = omega.copy()
        for i in range(2, n-1):
            dV = y_f[i] - y_f[i-1]
            aN = (nu + sig_w * nu_t_f[i]) / (y_n[i+1] - y_n[i])
            aS = (nu + sig_w * nu_t_f[i-1]) / (y_n[i] - y_n[i-1])
            Pk_lim = min(Pk[i], 10.0 * bstar * ko[i] * wo[i])
            Sp = -beta * wo[i] * dV
            Su = (alpha / max(nu_t[i], 1e-12)) * Pk_lim * dV
            aP = aN + aS - Sp
            if i == n-2:
                wn[i] = (aS * omega[i-1] + Su) / aP
            else:
                wn[i] = (aN * omega[i+1] + aS * omega[i-1] + Su) / aP
        omega = omega + ufw * (wn - omega)
        omega[1] = us / (np.sqrt(bstar) * kappa * y_n[1])
        omega[0] = omega[1]
        omega[-1] = omega[-2]
        omega = np.clip(omega, 1e-3, 1e6)

        # Convergence check
        ch = np.max(np.abs(U - Uo)) / (np.max(np.abs(U)) + 1e-10)
        if ch < tol and it > 200:
            break

    # Convert to wall units
    y_plus = y_n * us / nu
    U_plus = U / us
    k_plus = k / us**2
    nu_t_plus = nu_t / nu
    dU_dy = np.gradient(U_plus, y_plus)
    return {
        'y_plus': y_plus,
        'U_plus': U_plus,
        'k_plus': k_plus,
        'nu_t_plus': nu_t_plus,
        'dU_dy': dU_dy,
    }

# ============================================================
# 3. φ-VARIANCE WALL FUNCTION (POINTWISE)
# ============================================================
def phi_variance_at_point(U_plus, k_plus, nu_t_plus, dU_dy,
                          C_u=0.0, U_ref=1.0):
    """
    Evaluate the φ-variance wall function at a single point.

    The streamwise variance is estimated as:
        <u'^2>_model = (2/3) k * [1 + C_u * (ν_t/k) |dU/dy|]  (if C_u != 0)
    and the φ-variance is:
        <φ'^2> = <u'^2>_model / (1 + U^2)

    Parameters
    ----------
    U_plus : float
        Mean velocity in wall units at the evaluation point.
    k_plus : float
        Turbulent kinetic energy in wall units.
    nu_t_plus : float
        Eddy viscosity in wall units.
    dU_dy : float
        Mean velocity gradient in wall units.
    C_u : float, optional
        Correction parameter (default 0.0, i.e., no correction).
    U_ref : float, optional
        Reference velocity in wall units (default 1.0).

    Returns
    -------
    phi_sq : float
        Modelled φ-variance.
    """
    uu_model = (2.0 / 3.0) * k_plus
    if C_u != 0.0:
        corr = 1.0 + C_u * nu_t_plus * abs(dU_dy) / max(k_plus, 1e-12)
        corr = np.clip(corr, 0.1, 10.0)  # prevent unphysical values
        uu_model *= corr
    phi_sq = uu_model / (U_ref**2 + U_plus**2)
    return phi_sq

# ============================================================
# 4. MAIN ROUTINE
# ============================================================
def main():
    print("="*80)
    print(" Pointwise φ-Variance Wall Function with Calibrated RANS Inputs")
    print("="*80)

    # Create output directories
    os.makedirs('figures', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    prepare_data()
    Re_taus = [180, 395, 590]
    U_ref = 1.0
    C_u_values = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    results = []
    optimal_Cu = {}

    for Re_tau in Re_taus:
        print(f"\n===== Re_τ = {Re_tau} =====")
        # Load DNS
        dns = load_dns(Re_tau)
        # Solve k-ε (primary model)
        print("  Solving k-ε ...")
        ke = solve_k_epsilon(Re_tau)
        # The first interior node is index 1
        y_eval = ke['y_plus'][1]
        print(f"  First interior RANS node: y+ = {y_eval:.3f}")

        # Interpolate DNS to y_eval
        f_uu = interp1d(dns['y_plus'], dns['uu'], kind='linear',
                        fill_value='extrapolate')
        f_U = interp1d(dns['y_plus'], dns['U_plus'], kind='linear',
                       fill_value='extrapolate')
        uu_dns = float(f_uu(y_eval))
        U_dns = float(f_U(y_eval))
        phi_sq_dns = uu_dns / (U_ref**2 + U_dns**2)
        print(f"  DNS at y+={y_eval:.3f}: U+={U_dns:.4f}, <u'^2>={uu_dns:.4f}, "
              f"<φ'^2>={phi_sq_dns:.6f}")

        # RANS values at the same node
        U_rans = ke['U_plus'][1]
        k_rans = ke['k_plus'][1]
        nu_t_rans = ke['nu_t_plus'][1]
        dU_rans = ke['dU_dy'][1]
        print(f"  RANS at y+={y_eval:.3f}: U+={U_rans:.4f}, k+={k_rans:.4f}, "
              f"ν_t+={nu_t_rans:.4f}, dU/dy={dU_rans:.4f}")

        # Evaluate for each C_u
        for C_u in C_u_values:
            phi_model = phi_variance_at_point(U_rans, k_rans, nu_t_rans,
                                              dU_rans, C_u, U_ref)
            error = phi_model - phi_sq_dns
            rel_err = abs(error) / abs(phi_sq_dns) if phi_sq_dns != 0 else np.nan
            results.append({
                'Re_tau': Re_tau,
                'y_eval': y_eval,
                'C_u': C_u,
                'phi_model': phi_model,
                'phi_dns': phi_sq_dns,
                'error': error,
                'rel_err': rel_err,
                'U_rans': U_rans,
                'k_rans': k_rans,
                'nu_t_rans': nu_t_rans,
                'dU_rans': dU_rans,
                'uu_dns': uu_dns,
                'U_dns': U_dns,
            })

        # Find optimal C_u (minimum relative error)
        sub = [r for r in results if r['Re_tau'] == Re_tau]
        best = min(sub, key=lambda x: x['rel_err'])
        optimal_Cu[Re_tau] = best['C_u']
        print(f"  Optimal C_u = {best['C_u']:.2f}, "
              f"rel_err = {best['rel_err']:.3f}")

    # Print summary table
    print("\n" + "="*80)
    print(" POINTWISE EVALUATION SUMMARY")
    print("="*80)
    print(f"{'Re_τ':>6} {'y_eval':>8} {'C_u_opt':>8} {'rel_err_opt':>12}")
    print("-"*80)
    for Re_tau in Re_taus:
        sub = [r for r in results if r['Re_tau'] == Re_tau]
        best = min(sub, key=lambda x: x['rel_err'])
        print(f"{Re_tau:6d} {best['y_eval']:8.2f} {best['C_u']:8.2f} "
              f"{best['rel_err']:12.3f}")

    # Test with a constant C_u = 1.5
    print("\n" + "="*80)
    print(" TEST WITH CONSTANT C_u = 1.5")
    print("="*80)
    print(f"{'Re_τ':>6} {'rel_err':>10}")
    for Re_tau in Re_taus:
        sub = [r for r in results if r['Re_tau'] == Re_tau and r['C_u'] == 1.5]
        if sub:
            print(f"{Re_tau:6d} {sub[0]['rel_err']:10.3f}")

    # Fit a simple relation for C_u(Re_tau)
    Re_arr = np.array(Re_taus, dtype=float)
    Cu_arr = np.array([optimal_Cu[Re] for Re in Re_taus])
    def fit_func(Re, a, b):
        return a + b / Re
    try:
        popt, _ = curve_fit(fit_func, Re_arr, Cu_arr, p0=[1.0, 200.0])
        print("\n" + "="*80)
        print(" EMPIRICAL FIT FOR OPTIMAL C_u")
        print("="*80)
        print(f"  C_u(Re_τ) = {popt[0]:.3f} + {popt[1]:.3f} / Re_τ")
    except Exception as e:
        print(f"  Curve fit failed: {e}")

    # Plot relative error vs C_u for each Re_tau
    fig, ax = plt.subplots(figsize=(8, 5))
    for Re_tau in Re_taus:
        sub = [r for r in results if r['Re_tau'] == Re_tau]
        C_vals = [r['C_u'] for r in sub]
        err_vals = [r['rel_err'] for r in sub]
        ax.plot(C_vals, err_vals, 'o-', label=f'Re_τ = {Re_tau}')
    ax.set_xlabel(r'$C_u$')
    ax.set_ylabel('Relative error')
    ax.set_title('Relative error in φ-variance at first RANS node')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out1 = os.path.join('figures', 'pointwise_error_vs_Cu.png')
    plt.savefig(out1, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  -> {out1}")

    # Plot optimal C_u vs Re_tau
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(Re_taus, [optimal_Cu[Re] for Re in Re_taus], 'o', label='Optimal C_u')
    Re_fit = np.linspace(150, 650, 100)
    if 'popt' in locals():
        ax.plot(Re_fit, fit_func(Re_fit, *popt), '-',
                label=f'Fit: {popt[0]:.2f} + {popt[1]:.1f}/Re')
    ax.set_xlabel(r'$Re_\tau$')
    ax.set_ylabel('Optimal $C_u$')
    ax.set_title('Optimal correction parameter vs $Re_\\tau$')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out2 = os.path.join('figures', 'optimal_Cu_vs_Re.png')
    plt.savefig(out2, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  -> {out2}")

    # Save results to JSON
    json_out = os.path.join('results', 'pointwise_results.json')
    with open(json_out, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n  -> {json_out}")

if __name__ == "__main__":
    main()