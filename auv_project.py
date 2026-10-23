#!/usr/bin/env python3
"""
AUV Underwater Navigation Without GPS — Complete 5-Module Math Prototype
Covers CAT II & CAT III Objectives 1-5, Hypotheses H1 & H2, and Modules 1-5:
  - Module 1: Gauss-Jordan Elimination & Matrix Rank (Acoustic Trilateration & Matrix Inverse)
  - Module 2: Vector Spaces, Basis & 2D/3D Rotation Matrices Rz(psi)*Ry(theta)*Rx(phi)
  - Module 3: Eigenvalues, Eigenvectors & Diagonalization of Error Covariance Matrix
  - Module 4: Probability, Gaussian Noise, Monte Carlo Simulation & Kalman Filter
  - Module 5: Linear Regression, Parameter Estimation, R^2 & Standard Error (Design Rule)
"""

import math
import random


# ==============================================================================
# MODULE 1: MATRICES, RANK & GAUSS-JORDAN ELIMINATION
# ==============================================================================
def gauss_jordan_solve(A, b):
    """Solve Ax = b and compute Rank(A) using Gauss-Jordan Elimination."""
    n = len(A)
    aug = [row[:] + [b[i]] for i, row in enumerate(A)]
    rank = 0
    for col in range(n):
        pivot = max(range(rank, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            continue
        aug[rank], aug[pivot] = aug[pivot], aug[rank]
        div = aug[rank][col]
        aug[rank] = [val / div for val in aug[rank]]
        for r in range(n):
            if r != rank and abs(aug[r][col]) > 1e-12:
                factor = aug[r][col]
                aug[r] = [aug[r][c] - factor * aug[rank][c] for c in range(n + 1)]
        rank += 1
    x = [aug[i][n] for i in range(n)]
    return x, rank, aug


def invert_2x2_gauss_jordan(M):
    """Invert a 2x2 matrix [M | I] -> [I | M^-1] via Gauss-Jordan."""
    aug = [
        [M[0][0], M[0][1], 1.0, 0.0],
        [M[1][0], M[1][1], 0.0, 1.0],
    ]
    for col in range(2):
        div = aug[col][col]
        aug[col] = [v / div for v in aug[col]]
        other = 1 - col
        factor = aug[other][col]
        aug[other] = [aug[other][c] - factor * aug[col][c] for c in range(4)]
    return [[aug[0][2], aug[0][3]], [aug[1][2], aug[1][3]]]


# ==============================================================================
# MODULE 2: VECTOR SPACES, BASIS & 2D/3D ROTATION MATRICES
# ==============================================================================
def rotation_matrix_2d(psi_deg):
    """Return 2x2 orthogonal rotation matrix R(psi)."""
    psi = math.radians(psi_deg)
    c, s = math.cos(psi), math.sin(psi)
    return [[c, -s], [s, c]]


def rotation_matrix_3d(psi_deg, theta_deg, phi_deg):
    """Return 3x3 rotation matrix R = Rz(psi) * Ry(theta) * Rx(phi)."""
    psi = math.radians(psi_deg)      # Yaw (Z)
    theta = math.radians(theta_deg)  # Pitch (Y)
    phi = math.radians(phi_deg)      # Roll (X)
    cz, sz = math.cos(psi), math.sin(psi)
    cy, sy = math.cos(theta), math.sin(theta)
    cx, sx = math.cos(phi), math.sin(phi)
    return [
        [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
        [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
        [-sy,     cy * sx,                cy * cx],
    ]


def mat_vec_mul(R, v):
    return [sum(R[i][j] * v[j] for j in range(len(v))) for i in range(len(R))]


def mat_mul_2x2(A, B):
    return [
        [A[0][0] * B[0][0] + A[0][1] * B[1][0], A[0][0] * B[0][1] + A[0][1] * B[1][1]],
        [A[1][0] * B[0][0] + A[1][1] * B[1][0], A[1][0] * B[0][1] + A[1][1] * B[1][1]],
    ]


# ==============================================================================
# MODULE 3: EIGENVALUES, EIGENVECTORS & DIAGONALIZATION (2x2 COVARIANCE MATRIX)
# ==============================================================================
def eigen_2x2_symmetric(Sigma):
    """
    Solve det(Sigma - lambda*I) = 0 for a symmetric 2x2 covariance matrix:
    Sigma = [[a, b], [b, d]]
    Returns eigenvalues (lambda1 >= lambda2) and normalized eigenvectors (v1, v2).
    """
    a, b, d = Sigma[0][0], Sigma[0][1], Sigma[1][1]
    trace = a + d
    det = a * d - b * b
    disc = math.sqrt(max(0.0, (trace / 2.0) ** 2 - det))
    l1 = trace / 2.0 + disc
    l2 = trace / 2.0 - disc

    if abs(b) > 1e-12:
        v1 = [b, l1 - a]
        n1 = math.hypot(v1[0], v1[1])
        v1 = [v1[0] / n1, v1[1] / n1]
        v2 = [-v1[1], v1[0]]
    else:
        v1 = [1.0, 0.0] if a >= d else [0.0, 1.0]
        v2 = [0.0, 1.0] if a >= d else [1.0, 0.0]
    return (l1, l2), (v1, v2)


# ==============================================================================
# MODULE 4: PROBABILITY, MONTE CARLO SIMULATION & KALMAN FILTER
# ==============================================================================
def simulate_mission(
    steps=120,
    dt=1.0,
    true_speed=2.0,
    true_heading_deg=30.0,
    sigma_speed=0.15,
    sigma_heading_deg=2.0,
    bias_heading_deg=1.5,
    fix_interval=None,
    sigma_fix=1.2,
    rng=None,
):
    """
    Simulate 2D AUV trajectory over `steps` seconds.
    If `fix_interval` is given, run a Linear Kalman Filter fusing Dead Reckoning
    with periodic acoustic position fixes every `fix_interval` seconds.
    """
    rng = rng or random.Random(42)
    true_p = [0.0, 0.0]
    est_p = [0.0, 0.0]

    # Process covariance Q (per step) and measurement covariance R_fix
    q_step = (sigma_speed * dt) ** 2 + (true_speed * dt * math.radians(sigma_heading_deg)) ** 2
    Sigma = [[0.01, 0.0], [0.0, 0.01]]
    R_fix = [[sigma_fix ** 2, 0.0], [0.0, sigma_fix ** 2]]

    errors = []
    for k in range(1, steps + 1):
        # True motion in Earth frame
        R_true = rotation_matrix_2d(true_heading_deg)
        dp_true = mat_vec_mul(R_true, [true_speed * dt, 0.0])
        true_p = [true_p[0] + dp_true[0], true_p[1] + dp_true[1]]

        # Noisy sensor measurements (Gaussian random variables + bias)
        meas_speed = true_speed + rng.gauss(0.0, sigma_speed)
        meas_heading = true_heading_deg + bias_heading_deg + rng.gauss(0.0, sigma_heading_deg)
        R_meas = rotation_matrix_2d(meas_heading)
        dp_meas = mat_vec_mul(R_meas, [meas_speed * dt, 0.0])

        # Kalman Predict Step
        est_p = [est_p[0] + dp_meas[0], est_p[1] + dp_meas[1]]
        Sigma[0][0] += q_step
        Sigma[1][1] += q_step

        # Kalman Update Step (when acoustic beacon ping arrives)
        if fix_interval and (k % fix_interval == 0):
            z = [true_p[0] + rng.gauss(0.0, sigma_fix), true_p[1] + rng.gauss(0.0, sigma_fix)]
            S = [[Sigma[0][0] + R_fix[0][0], Sigma[0][1]], [Sigma[1][0], Sigma[1][1] + R_fix[1][1]]]
            S_inv = invert_2x2_gauss_jordan(S)
            K = mat_mul_2x2(Sigma, S_inv)
            innov = [z[0] - est_p[0], z[1] - est_p[1]]
            est_p = [
                est_p[0] + K[0][0] * innov[0] + K[0][1] * innov[1],
                est_p[1] + K[1][0] * innov[0] + K[1][1] * innov[1],
            ]
            I_minus_K = [[1.0 - K[0][0], -K[0][1]], [-K[1][0], 1.0 - K[1][1]]]
            Sigma = mat_mul_2x2(I_minus_K, Sigma)

        err = math.hypot(est_p[0] - true_p[0], est_p[1] - true_p[1])
        errors.append(err)

    return true_p, est_p, errors, Sigma


def monte_carlo_rmse(runs=300, **kwargs):
    """Run Monte Carlo simulation and return final position error covariance & mean RMSE."""
    sq_errors = []
    final_dx, final_dy = [], []
    for seed in range(runs):
        rng = random.Random(1000 + seed)
        true_p, est_p, errors, _ = simulate_mission(rng=rng, **kwargs)
        # Steady-state RMSE across the mission
        rmse = math.sqrt(sum(e * e for e in errors) / len(errors))
        sq_errors.append(rmse)
        final_dx.append(est_p[0] - true_p[0])
        final_dy.append(est_p[1] - true_p[1])

    mean_rmse = sum(sq_errors) / len(sq_errors)
    mx = sum(final_dx) / runs
    my = sum(final_dy) / runs
    sxx = sum((x - mx) ** 2 for x in final_dx) / (runs - 1)
    syy = sum((y - my) ** 2 for y in final_dy) / (runs - 1)
    sxy = sum((final_dx[i] - mx) * (final_dy[i] - my) for i in range(runs)) / (runs - 1)
    return mean_rmse, [[sxx, sxy], [sxy, syy]]


# ==============================================================================
# MODULE 5: LINEAR REGRESSION, PARAMETER ESTIMATION & STANDARD ERROR
# ==============================================================================
def linear_regression(x, y):
    """
    Ordinary Least Squares Regression: y_hat = beta0 + beta1 * x
    Returns beta0, beta1, r_squared, std_error_slope
    """
    n = len(x)
    mx = sum(x) / n
    my = sum(y) / n
    ss_xy = sum((x[i] - mx) * (y[i] - my) for i in range(n))
    ss_xx = sum((x[i] - mx) ** 2 for i in range(n))
    ss_yy = sum((y[i] - my) ** 2 for i in range(n))

    beta1 = ss_xy / ss_xx
    beta0 = my - beta1 * mx

    residuals = [y[i] - (beta0 + beta1 * x[i]) for i in range(n)]
    ss_res = sum(r * r for r in residuals)
    r_squared = 1.0 - (ss_res / ss_yy)
    s_err_est = math.sqrt(ss_res / (n - 2))
    se_beta1 = s_err_est / math.sqrt(ss_xx)
    return beta0, beta1, r_squared, se_beta1


# ==============================================================================
# RUN ALL 5 MODULES & SELF-CHECK
# ==============================================================================
def main():
    print("=" * 74)
    print("AUV NAVIGATION WITHOUT GPS — 5-MODULE MATHEMATICAL PROTOTYPE")
    print("=" * 74)

    # 1. MODULE 1: Gauss-Jordan Elimination for Acoustic Beacon Fix
    # Linearized trilateration system A * p = b for 3 seabed transponders
    A_beacon = [
        [2.0, 1.0, -1.0],
        [-3.0, -1.0, 2.0],
        [-2.0, 1.0, 2.0],
    ]
    b_beacon = [8.0, -11.0, -3.0]
    pos_fix, rank, rref = gauss_jordan_solve(A_beacon, b_beacon)
    assert rank == 3 and [round(v, 4) for v in pos_fix] == [2.0, 3.0, -1.0]
    print("\n[MODULE 1] Gauss-Jordan Elimination & Matrix Rank (Acoustic Fix)")
    print(f"  Rank(A) = {rank} (Full rank -> unique 3D acoustic position fix)")
    print(f"  Solved Beacon Position Fix (X, Y, Z) = ({pos_fix[0]:.2f}, {pos_fix[1]:.2f}, {pos_fix[2]:.2f}) m")

    # 2. MODULE 2: 2D & 3D Coordinate Transformations (Objective 1)
    p0 = [0.0, 0.0]
    dp1 = mat_vec_mul(rotation_matrix_2d(30.0), [10.0, 0.0])
    p1 = [p0[0] + dp1[0], p0[1] + dp1[1]]
    dp2 = mat_vec_mul(rotation_matrix_2d(120.0), [6.0, 2.0])
    p2 = [p1[0] + dp2[0], p1[1] + dp2[1]]
    R3 = rotation_matrix_3d(psi_deg=30.0, theta_deg=-15.0, phi_deg=0.0)
    dp3d = mat_vec_mul(R3, [10.0, 0.0, 0.0])
    assert abs(math.hypot(*dp1) - 10.0) < 1e-9, "Orthogonal matrix must preserve norm"
    print("\n[MODULE 2] Coordinate Transformations & Rotation Matrices (Objective 1)")
    print(f"  2D Move 1 (psi=30 deg, surge=10m)        -> p1 = ({p1[0]:.2f}, {p1[1]:.2f}) m")
    print(f"  2D Move 2 (psi=120 deg, surge=6m, sway=2m)-> p2 = ({p2[0]:.2f}, {p2[1]:.2f}) m")
    print(f"  3D Dive   (psi=30 deg, pitch=-15 deg, 10m)-> dP = ({dp3d[0]:.2f}, {dp3d[1]:.2f}, {dp3d[2]:.2f}) m")

    # 3. MODULE 4 & H1: Monte Carlo Sensitivity Analysis (Objectives 2 & 3)
    rmse_speed_only, _ = monte_carlo_rmse(
        sigma_speed=0.15, sigma_heading_deg=0.0, bias_heading_deg=0.0, fix_interval=None
    )
    rmse_heading_only, cov_dr = monte_carlo_rmse(
        sigma_speed=0.0, sigma_heading_deg=2.0, bias_heading_deg=1.5, fix_interval=None
    )
    rmse_both_dr, cov_full = monte_carlo_rmse(
        sigma_speed=0.15, sigma_heading_deg=2.0, bias_heading_deg=1.5, fix_interval=None
    )
    assert rmse_heading_only > rmse_speed_only, "H1 check: Heading error must dominate speed error"
    print("\n[MODULE 4] Monte Carlo Error Growth & Sensitivity Analysis (Objectives 2 & 3, H1)")
    print(f"  Speed Error Only   (sigma_v=0.15 m/s)           : RMSE = {rmse_speed_only:.2f} m")
    print(f"  Heading Error Only (sigma_psi=2 deg, bias=1.5)  : RMSE = {rmse_heading_only:.2f} m")
    print(f"  Combined Uncorrected Dead Reckoning             : RMSE = {rmse_both_dr:.2f} m")
    print("  -> H1 Confirmed: Heading error dominates speed error over distance travelled.")

    # 4. MODULE 3: Eigenvalues & Eigenvectors of Error Covariance Matrix
    (l1, l2), (v1, v2) = eigen_2x2_symmetric(cov_full)
    print("\n[MODULE 3] Eigenvalues & Eigenvectors of Final Error Covariance Matrix Sigma")
    print(f"  Sigma = [[{cov_full[0][0]:.3f}, {cov_full[0][1]:.3f}], [{cov_full[1][0]:.3f}, {cov_full[1][1]:.3f}]]")
    print(f"  Eigenvalues  : lambda_1 = {l1:.3f} (Major std = {math.sqrt(l1):.2f} m), lambda_2 = {l2:.3f} (Minor std = {math.sqrt(l2):.2f} m)")
    print(f"  Eigenvectors : v_1 = [{v1[0]:.3f}, {v1[1]:.3f}] (Cross-track axis), v_2 = [{v2[0]:.3f}, {v2[1]:.3f}]")

    # 5. MODULE 4 & 5: Kalman Filter Fix Intervals & Linear Regression Design Rule (Objectives 4 & 5, H2)
    intervals = [5, 10, 20, 30, 40, 60]
    kf_rmses = []
    for dt_fix in intervals:
        rmse_kf, _ = monte_carlo_rmse(fix_interval=dt_fix)
        kf_rmses.append(rmse_kf)

    beta0, beta1, r2, se_b1 = linear_regression(intervals, kf_rmses)
    assert kf_rmses[0] < kf_rmses[-1] < rmse_both_dr, "H2 check: Kalman filter bounds error & increases with interval"
    print("\n[MODULE 4 & 5] Kalman Filter & Linear Regression Design Rule (Objectives 4 & 5, H2)")
    for dt_fix, val in zip(intervals, kf_rmses):
        print(f"  Fix Interval dT = {dt_fix:2d} s -> Steady-State Kalman RMSE = {val:.2f} m")
    print(f"\n  Least-Squares Design Rule : RMSE = {beta0:.3f} + {beta1:.4f} * dT_fix")
    print(f"  Goodness of Fit           : R^2 = {r2:.4f}, Standard Error(beta_1) = {se_b1:.5f}")
    target_err = 2.0
    max_interval = (target_err - beta0) / beta1
    print(f"  Practical Rule (Target <= {target_err:.1f} m): Acoustic ping required every <= {max_interval:.1f} seconds.")
    print("=" * 74)


if __name__ == "__main__":
    main()
