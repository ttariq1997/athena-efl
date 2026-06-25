#ifndef CHARACTERISTIC_FIELDS_RMHD_HPP_
#define CHARACTERISTIC_FIELDS_RMHD_HPP_

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <limits>
#include <string>

#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../eos/eos.hpp"
#include "../mesh/mesh.hpp"

namespace characterisiticfields::rmhd {

// Numerical tolerances for SRMHD eigensystem calculations
namespace constants {
  constexpr Real EPSILON_SMALL = 1.0e-42;              // Prevent division by zero
  constexpr Real CS2_MIN = 1.0e-12;                    // Minimum sound speed squared
  constexpr Real CS2_MAX_OFFSET = 1.0e-12;             // Max cs2 = 1.0 - this value
  constexpr Real QUARTIC_TOL = 1.0e-14;                // Quartic coefficient tolerance
  constexpr Real DEGENERACY_TOL_BASE = 1.0e-10;        // Base Type I degeneracy tolerance
  constexpr Real DEGENERACY_TOL_SCALE = 1.0e-6;        // Scaling factor for degeneracy
  constexpr Real MATRIX_SINGULAR_TOL = 1.0e-18;        // Matrix inversion singularity
  constexpr Real BIORTHOGONAL_TOL = 1.0e-6;            // L·R biorthogonality tolerance
  constexpr Real GAMMA_ADI_MIN = 1.0e-12;              // Minimum gamma - 1.0
  constexpr Real PRESSURE_MIN = 1.0e-14;               // Minimum pressure for calculations
}

enum HOReconKindRMHD {
  HO_RECON_WENO5 = 0,
  HO_RECON_WENO5Z = 1,
  HO_RECON_CS5 = 2
};

inline Real ReconstructScalarHO(const Real char_flx[5], const int rec_kind) {
  constexpr Real optimw[3] = {1.0/10.0, 3.0/5.0, 3.0/10.0};
  constexpr Real epsl = constants::EPSILON_SMALL;
  constexpr Real othreeotwo = 13.0/12.0;

  const Real fipt = char_flx[4];
  const Real fipo = char_flx[3];
  const Real fi   = char_flx[2];
  const Real fimo = char_flx[1];
  const Real fimt = char_flx[0];

  if (rec_kind == HO_RECON_CS5) {
    return (2.0 * fimt - 13.0 * fimo + 47.0 * fi + 27.0 * fipo - 3.0 * fipt) / 60.0;
  }

  Real a[3], b[3], w[3], fk[3];
  b[0] = othreeotwo * SQR(fimt - 2.0 * fimo + fi)
       + 0.25 * SQR(fimt - 4.0 * fimo + 3.0 * fi);
  b[1] = othreeotwo * SQR(fimo - 2.0 * fi + fipo)
       + 0.25 * SQR(fimo - fipo);
  b[2] = othreeotwo * SQR(fi - 2.0 * fipo + fipt)
       + 0.25 * SQR(3.0 * fi - 4.0 * fipo + fipt);

  if (rec_kind == HO_RECON_WENO5Z) {
    for (int j = 0; j < 3; ++j) {
      a[j] = optimw[j] * (1.0 + std::abs(b[0] - b[2]) / (b[j] + epsl));
    }
  } else {
    for (int j = 0; j < 3; ++j) {
      a[j] = optimw[j] / SQR(epsl + b[j]);
    }
  }

  const Real inv_sum = 1.0 / (a[0] + a[1] + a[2]);
  for (int j = 0; j < 3; ++j) {
    w[j] = a[j] * inv_sum;
  }

  fk[0] = (2.0 * fimt - 7.0 * fimo + 11.0 * fi) / 6.0;
  fk[1] = (-fimo + 5.0 * fi + 2.0 * fipo) / 6.0;
  fk[2] = (2.0 * fi + 5.0 * fipo - fipt) / 6.0;
  return w[0] * fk[0] + w[1] * fk[1] + w[2] * fk[2];
}

constexpr int NRMHD = 7;
constexpr int NCOV_RMHD = 10;

enum ReducedRMHDIndex {
  ID_R   = 0,
  ISN_R  = 1,
  IST1_R = 2,
  IST2_R = 3,
  ITAU_R = 4,
  IBT1_R = 5,
  IBT2_R = 6
};

enum CovariantRMHDIndex {
  IU0_COV = 0,
  IUN_COV = 1,
  IUT1_COV = 2,
  IUT2_COV = 3,
  IB0_COV = 4,
  IBN_COV = 5,
  IBT1_COV = 6,
  IBT2_COV = 7,
  IP_COV = 8,
  IS_COV = 9
};

// The high-order RMHD eigensystem below is written in the shifted-energy basis
// [D, S_n, S_t1, S_t2, tau, B_t1, B_t2], with
//   tau = E - D,
// where Athena stores the conserved energy slot as the full total energy E.
// Whenever data are packed from / written back to Athena arrays, only the
// energy component requires this linear change of variables.

struct RMHDDirMap {
  int ivn;
  int ivt1;
  int ivt2;
  int isn;
  int ist1;
  int ist2;
  int ibn;
  int ibt1;
  int ibt2;
};

struct RMHDState {
  Real rho = 0.0;
  Real pgas = 0.0;
  Real gamma_adi = 0.0;
  Real h = 0.0;
  Real rhoh = 0.0;
  Real cs2 = 0.0;
  Real Emhd = 0.0;

  Real W = 0.0;
  Real util_n = 0.0;
  Real util_t1 = 0.0;
  Real util_t2 = 0.0;
  Real v_n = 0.0;
  Real v_t1 = 0.0;
  Real v_t2 = 0.0;

  Real D = 0.0;
  Real S_n = 0.0;
  Real S_t1 = 0.0;
  Real S_t2 = 0.0;
  Real tau = 0.0;

  Real Bn = 0.0;
  Real Bt1 = 0.0;
  Real Bt2 = 0.0;

  Real b0 = 0.0;
  Real b_n = 0.0;
  Real b_t1 = 0.0;
  Real b_t2 = 0.0;
  Real bsq = 0.0;
  Real ptot = 0.0;
};

inline void FillConservedFromFinalizedStateSRMHD(RMHDState &state) {
  state.D = state.rho * state.W;
  const Real wtot_u02 = (state.rhoh + state.bsq) * SQR(state.W);
  state.S_n  = (state.rhoh + state.bsq) * state.W * state.util_n  - state.b0 * state.b_n;
  state.S_t1 = (state.rhoh + state.bsq) * state.W * state.util_t1 - state.b0 * state.b_t1;
  state.S_t2 = (state.rhoh + state.bsq) * state.W * state.util_t2 - state.b0 * state.b_t2;
  state.tau = wtot_u02 - SQR(state.b0) - (state.pgas + 0.5 * state.bsq) - state.D;
}

struct AntonTangentialDataSRMHD {
  Real alpha1[4] = {};
  Real alpha2[4] = {};
  Real alpha11 = 0.0;
  Real alpha12 = 0.0;
  Real alpha22 = 0.0;
  Real g1 = 0.0;
  Real g2 = 0.0;
  Real f1 = 0.0;
  Real f2 = 0.0;
  Real bt[4] = {};
  Real bt_dir[4] = {};
  Real abs_bt = 0.0;
  Real ghat1 = 0.0;
  Real ghat2 = 0.0;
};

// Thread-local "last failure metric" passed through the eigsys-failure path
// into the L = R^{-1} fallback (read at GetEigenVectorSRMHD post-LeftFail).
inline Real &LastEigenFailureMetricSRMHD() {
  static thread_local Real metric = std::numeric_limits<Real>::quiet_NaN();
  return metric;
}

inline void SetLastEigenFailureSRMHD(
    const int /*code*/,
    const Real metric = std::numeric_limits<Real>::quiet_NaN()) {
  LastEigenFailureMetricSRMHD() = metric;
}

// Forward declarations for functions that have callers preceding their definitions.
inline Real CheckEigenSystemSRMHD(const Real (&L)[NRMHD][NRMHD],
                                  const Real (&R)[NRMHD][NRMHD]);
inline void GetLeftMagnetosonicRenormBranchSRMHD(const Real lambda[NRMHD],
                                                 bool use_typeii_form[NRMHD]);

inline RMHDDirMap GetRMHDDirMap(const int ivx) {
  switch (ivx) {
    case IVX:
      return {IVX, IVY, IVZ, IM1, IM2, IM3, IB1, IB2, IB3};
    case IVY:
      return {IVY, IVZ, IVX, IM2, IM3, IM1, IB2, IB3, IB1};
    default:
      return {IVZ, IVX, IVY, IM3, IM1, IM2, IB3, IB1, IB2};
  }
}

inline Real QuadraticRootRMHD(const Real a1, const Real a0,
                              const bool greater_root) {
  // Same quadratic-root helper used elsewhere in the codebase; see also
  // src/pgen/unsupported/gr_linear_wave.cpp and Numerical Recipes (5.6).
  if (a1*a1 < 4.0*a0) {
    return -a1 / 2.0;
  }
  if (greater_root) {
    if (a1 >= 0.0) {
      return -2.0 * a0 / (a1 + std::sqrt(a1*a1 - 4.0*a0));
    } else {
      return (-a1 + std::sqrt(a1*a1 - 4.0*a0)) / 2.0;
    }
  } else {
    if (a1 >= 0.0) {
      return (-a1 - std::sqrt(a1*a1 - 4.0*a0)) / 2.0;
    } else {
      return -2.0 * a0 / (a1 - std::sqrt(a1*a1 - 4.0*a0));
    }
  }
}

inline Real CubicRootRealRMHD(const Real a2, const Real a1, const Real a0) {
  // Same cubic-root construction used in src/pgen/unsupported/gr_linear_wave.cpp
  // and based on Numerical Recipes (5.6).
  const Real q = (a2*a2 - 3.0*a1) / 9.0;
  const Real r = (2.0*a2*a2*a2 - 9.0*a1*a2 + 27.0*a0) / 54.0;
  if (r*r - q*q*q < 0.0) {
    const Real theta = std::acos(r / std::sqrt(q*q*q));
    return -2.0 * std::sqrt(q) * std::cos(theta / 3.0) - a2 / 3.0;
  }

  const Real a = -copysign(1.0, r)
               * std::cbrt(std::abs(r) + std::sqrt(r*r - q*q*q));
  const Real b = (a != 0.0) ? q / a : 0.0;
  return a + b - a2 / 3.0;
}

// Cubic solver for degenerate quartic case (when leading coefficient → 0)
inline void SolveCubicForMagnetosonicSRMHD(const Real a2, const Real a1, const Real a0,
                                           Real &lambda_fm, Real &lambda_sm,
                                           Real &lambda_sp, Real &lambda_fp) {
  // Solve x³ + a2*x² + a1*x + a0 = 0
  // Using Cardano's formula with numerical stability improvements
  const Real Q = (3.0 * a1 - SQR(a2)) / 9.0;
  const Real R = (9.0 * a2 * a1 - 27.0 * a0 - 2.0 * a2 * a2 * a2) / 54.0;
  const Real discriminant = SQR(R) + Q * Q * Q;

  if (discriminant > 0) {
    // One real root, two complex conjugates
    const Real sqrt_disc = std::sqrt(discriminant);
    const Real S = std::cbrt(R + sqrt_disc);
    const Real T = std::cbrt(R - sqrt_disc);
    const Real real_root = S + T - a2 / 3.0;

    // For Type I degeneracy, this is typically a fast wave
    // Slow waves have collapsed to contact speed
    if (real_root < 0) {
      lambda_fm = real_root;
      lambda_sm = 0.0;  // Collapsed to contact
      lambda_sp = 0.0;  // Collapsed to contact
      lambda_fp = -real_root;  // Assume symmetry
    } else {
      lambda_fm = -real_root;  // Assume symmetry
      lambda_sm = 0.0;  // Collapsed to contact
      lambda_sp = 0.0;  // Collapsed to contact
      lambda_fp = real_root;
    }
  } else {
    // Three real roots
    const Real theta = std::acos(R / std::sqrt(-Q * Q * Q));
    const Real sqrt_Q = std::sqrt(-Q);
    const Real offset = a2 / 3.0;

    Real roots[3];
    roots[0] = 2.0 * sqrt_Q * std::cos(theta / 3.0) - offset;
    roots[1] = 2.0 * sqrt_Q * std::cos((theta + 2.0 * M_PI) / 3.0) - offset;
    roots[2] = 2.0 * sqrt_Q * std::cos((theta + 4.0 * M_PI) / 3.0) - offset;

    // Sort roots
    if (roots[0] > roots[1]) std::swap(roots[0], roots[1]);
    if (roots[1] > roots[2]) std::swap(roots[1], roots[2]);
    if (roots[0] > roots[1]) std::swap(roots[0], roots[1]);

    // Assign based on physical interpretation
    lambda_fm = roots[0];  // Fast minus
    lambda_sm = roots[1];  // Slow minus (or collapsed)
    lambda_sp = roots[1];  // Slow plus (or collapsed)
    lambda_fp = roots[2];  // Fast plus
  }
}

inline void QuarticRootsRMHD(const Real a3, const Real a2, const Real a1,
                             const Real a0, Real *px1, Real *px2,
                             Real *px3, Real *px4) {
  // Same quartic solver strategy used in src/pgen/unsupported/gr_linear_wave.cpp.
  // It is used here to recover the four SR-RMHD magnetosonic eigenvalues.
  const Real b2 = a2 - 3.0/8.0 * SQR(a3);
  const Real b1 = a1 - 1.0/2.0 * a2 * a3 + 1.0/8.0 * a3 * SQR(a3);
  const Real b0 = a0 - 1.0/4.0 * a1 * a3 + 1.0/16.0 * a2 * SQR(a3)
                - 3.0/256.0 * SQR(SQR(a3));

  const Real c2 = -b2;
  const Real c1 = -4.0 * b0;
  const Real c0 = 4.0 * b0 * b2 - SQR(b1);

  const Real z0 = CubicRootRealRMHD(c2, c1, c0);
  const Real d1 = (z0 - b2 > 0.0) ? std::sqrt(z0 - b2) : 0.0;
  const Real e1 = -d1;

  Real rad = SQR(z0) / 4.0 - b0;
  if (rad < 0.0) rad = 0.0;
  const Real srad = std::sqrt(rad);

  Real d0, e0;
  if (b1 < 0.0) {
    d0 = z0 / 2.0 + srad;
    e0 = z0 / 2.0 - srad;
  } else {
    d0 = z0 / 2.0 - srad;
    e0 = z0 / 2.0 + srad;
  }

  const Real y1 = QuadraticRootRMHD(d1, d0, false);
  const Real y2 = QuadraticRootRMHD(d1, d0, true);
  const Real y3 = QuadraticRootRMHD(e1, e0, false);
  const Real y4 = QuadraticRootRMHD(e1, e0, true);

  *px1 = std::min(y1, y3) - a3 / 4.0;
  const Real mid_1 = std::max(y1, y3) - a3 / 4.0;
  *px4 = std::max(y2, y4) - a3 / 4.0;
  const Real mid_2 = std::min(y2, y4) - a3 / 4.0;
  *px2 = std::min(mid_1, mid_2);
  *px3 = std::max(mid_1, mid_2);
}

inline void GetHydroSoundSpeedsSRMHD(const RMHDState &state,
                                     Real &lambda_m, Real &lambda_p) {
  const Real cs2 = std::max(constants::CS2_MIN,
                            std::min((Real)1.0 - constants::CS2_MAX_OFFSET, state.cs2));
  const Real vx = state.v_n;
  const Real vt2 = SQR(state.v_t1) + SQR(state.v_t2);
  const Real v2 = SQR(vx) + vt2;
  const Real denom = 1.0 - v2 * cs2;
  const Real radicand =
      std::max((Real)0.0,
               (1.0 - v2) * (1.0 - v2 * cs2 - SQR(vx) * (1.0 - cs2)));
  const Real root = std::sqrt(cs2 * radicand);
  if (std::abs(denom) < constants::QUARTIC_TOL) {
    lambda_m = vx;
    lambda_p = vx;
    return;
  }
  lambda_m = (vx * (1.0 - cs2) - root) / denom;
  lambda_p = (vx * (1.0 - cs2) + root) / denom;
}

// Main function to compute SRMHD eigenvalues (wave speeds)
// Returns 7 eigenvalues: λ = {fast-, Alfven-, slow-, entropy, slow+, Alfven+, fast+}
inline void GetEigenValuesSRMHD(const RMHDState &state, Real lambda[NRMHD]) {
  // References:
  // 1. Antón et al. 2010, "Relativistic MHD: Renormalized eigenvectors..."
  //    - Entropy wave: λ = v_n (contact/material wave)
  //    - Alfvén waves: eq. (38) - intermediate speed waves
  //    - Magnetosonic waves: quartic N_4 = 0, eq. (35) - fast/slow MHD waves
  //
  // Algorithm handles three degeneracy types:
  // - Type I: Bn → 0 (perpendicular propagation) - 5 waves collapse to v_n
  // - Type II: |b_t| → 0 (parallel propagation) - tangential field vanishes
  // - Type II': cs² = b²/E (special resonance) - characteristic speed coincidence
  const Real cs2 = std::max(constants::CS2_MIN,
                            std::min((Real)1.0 - constants::CS2_MAX_OFFSET, state.cs2));
  const Real Emhd = state.rhoh + state.bsq;
  const Real B_lab =
      std::sqrt(std::max((Real)0.0,
                         SQR(state.Bn) + SQR(state.Bt1) + SQR(state.Bt2)));
  const Real hydro_tol =
      1.0e-12 * std::max({(Real)1.0, std::sqrt(std::abs(state.rhoh)),
                          std::sqrt(std::abs(state.pgas + state.rho))});
  const Real B_com =
      std::sqrt(std::max((Real)0.0,
                         SQR(state.b_n) + SQR(state.b_t1) + SQR(state.b_t2)));
  const Real field_scale =
      std::max({(Real)1.0, B_lab, B_com, std::abs(state.b0)});
  // Adaptive tolerance that also considers transverse field strength
  const Real Bt_scale = std::sqrt(SQR(state.Bt1) + SQR(state.Bt2));
  const Real type_i_tol = std::max(constants::DEGENERACY_TOL_BASE,
                                   constants::DEGENERACY_TOL_SCALE * std::max(field_scale, 0.1 * Bt_scale));

  // Entropy/contact speed from a = W (v_n - lambda) = 0.
  lambda[3] = state.v_n;

  // Exact or near-exact zero-field limit: the SR-RMHD characteristic structure
  // collapses to SR hydrodynamics plus five advected transverse/contact modes.
  // The quartic still carries the hydrodynamic sound roots, but repeated zero
  // roots are numerically fragile in the generic quartic solver and can be
  // misordered in the mixed Type-I/Type-II' limit. Handle this limit
  // explicitly with the analytic SRHD sound speeds.
  if (B_lab <= hydro_tol) {
    Real lambda_hm = state.v_n;
    Real lambda_hp = state.v_n;
    GetHydroSoundSpeedsSRMHD(state, lambda_hm, lambda_hp);
    lambda[0] = std::max((Real)-1.0, std::min((Real)1.0, lambda_hm));
    lambda[1] = state.v_n;
    lambda[2] = state.v_n;
    lambda[4] = state.v_n;
    lambda[5] = state.v_n;
    lambda[6] = std::max((Real)-1.0, std::min((Real)1.0, lambda_hp));
    return;
  }

  // Alfvén speeds, Antón et al. 2010 eq. (38):
  // lambda_a,\pm = (b_n \pm sqrt(rho h + b^2) u_n)
  //              / (b^0 \pm sqrt(rho h + b^2) u^0).
  const Real sqrt_E = std::sqrt(std::max((Real)0.0, Emhd));
  const Real den_a_1 = state.b0 + sqrt_E * state.W;
  const Real den_a_2 = state.b0 - sqrt_E * state.W;
  const Real lambda_a_1 =
      (std::abs(den_a_1) > constants::QUARTIC_TOL)
          ? (state.b_n + sqrt_E * state.util_n) / den_a_1
          : lambda[3];
  const Real lambda_a_2 =
      (std::abs(den_a_2) > constants::QUARTIC_TOL)
          ? (state.b_n - sqrt_E * state.util_n) / den_a_2
          : lambda[3];
  lambda[1] = std::min(lambda_a_1, lambda_a_2);
  lambda[5] = std::max(lambda_a_1, lambda_a_2);

  const Real factor_a = state.rhoh * (1.0 / cs2 - 1.0);
  const Real factor_b = -(state.rhoh + state.bsq / cs2);
  const Real gamma_2 = SQR(state.W);
  const Real gamma_4 = SQR(gamma_2);
  const Real vx = state.v_n;

  // Magnetosonic quartic coefficients from Antón et al. 2010 eq. (35),
  // expanded directly as a polynomial in lambda. The coefficient form follows
  // the existing SR-RMHD implementation in src/pgen/unsupported/gr_linear_wave.cpp.
  const Real coeff_4 = factor_a * gamma_4
                     - factor_b * gamma_2
                     - SQR(state.b0);
  const Real coeff_3 = -4.0 * factor_a * gamma_4 * vx
                     + 2.0 * factor_b * gamma_2 * vx
                     + 2.0 * state.b0 * state.b_n;
  const Real coeff_2 = 6.0 * factor_a * gamma_4 * SQR(vx)
                     + factor_b * gamma_2 * (1.0 - SQR(vx))
                     + SQR(state.b0) - SQR(state.b_n);
  const Real coeff_1 = -4.0 * factor_a * gamma_4 * vx * SQR(vx)
                     - 2.0 * factor_b * gamma_2 * vx
                     - 2.0 * state.b0 * state.b_n;
  const Real coeff_0 = factor_a * gamma_4 * SQR(SQR(vx))
                     + factor_b * gamma_2 * SQR(vx)
                     + SQR(state.b_n);

  // Check if quartic is genuinely quartic or degenerate
  const Real coeff_scale = std::max({std::abs(coeff_3),
                                     std::abs(coeff_2),
                                     std::abs(coeff_1),
                                     std::abs(coeff_0), (Real)1.0});

  if (std::abs(coeff_4) > constants::QUARTIC_TOL * coeff_scale) {
    // Standard quartic case
    QuarticRootsRMHD(coeff_3 / coeff_4, coeff_2 / coeff_4,
                     coeff_1 / coeff_4, coeff_0 / coeff_4,
                     &lambda[0], &lambda[2], &lambda[4], &lambda[6]);
  } else if (std::abs(coeff_3) > constants::QUARTIC_TOL * coeff_scale) {
    // Degenerate to cubic: coeff_3*λ³ + coeff_2*λ² + coeff_1*λ + coeff_0 = 0
    // This happens near Type I degeneracy where slow waves collapse
    SolveCubicForMagnetosonicSRMHD(coeff_2 / coeff_3, coeff_1 / coeff_3,
                                   coeff_0 / coeff_3,
                                   lambda[0], lambda[2], lambda[4], lambda[6]);
  } else if (std::abs(coeff_2) > constants::QUARTIC_TOL * coeff_scale) {
    // Degenerate to quadratic: coeff_2*λ² + coeff_1*λ + coeff_0 = 0
    // Extreme degeneracy case
    const Real discriminant = SQR(coeff_1) - 4.0 * coeff_2 * coeff_0;
    if (discriminant >= 0) {
      const Real sqrt_disc = std::sqrt(discriminant);
      lambda[0] = (-coeff_1 - sqrt_disc) / (2.0 * coeff_2);
      lambda[6] = (-coeff_1 + sqrt_disc) / (2.0 * coeff_2);
    } else {
      // No real roots - use contact speed
      lambda[0] = state.v_n;
      lambda[6] = state.v_n;
    }
    lambda[2] = state.v_n;  // Collapsed
    lambda[4] = state.v_n;  // Collapsed
  } else {
    // Complete degeneracy - all waves at contact speed or use hydro limit
    if (std::abs(state.Bn) <= type_i_tol && cs2 > constants::QUARTIC_TOL) {
      // Use hydro sound speeds as approximation
      Real lambda_hm = state.v_n;
      Real lambda_hp = state.v_n;
      GetHydroSoundSpeedsSRMHD(state, lambda_hm, lambda_hp);
      lambda[0] = lambda_hm;
      lambda[6] = lambda_hp;
    } else {
      lambda[0] = state.v_n;
      lambda[6] = state.v_n;
    }
    lambda[2] = state.v_n;
    lambda[4] = state.v_n;
  }

  // Type-I degeneracy (Bn -> 0): the two Alfven waves, the entropy/contact
  // wave, and the two slow magnetosonic waves all propagate at lambda = v_n.
  // The quartic still supplies the fast pair, but the repeated middle family
  // is numerically fragile and can be misassigned by the generic quartic path.
  if (std::abs(state.Bn) <= type_i_tol) {
    const Real lambda_fast_m = std::min(lambda[0], lambda[6]);
    const Real lambda_fast_p = std::max(lambda[0], lambda[6]);
    lambda[0] = lambda_fast_m;
    lambda[1] = state.v_n;
    lambda[2] = state.v_n;
    lambda[3] = state.v_n;
    lambda[4] = state.v_n;
    lambda[5] = state.v_n;
    lambda[6] = lambda_fast_p;
  }

  // Clamp eigenvalues strictly inside the light cone: |λ| < 1.
  // The magnetosonic eigenvector formulas divide by G = 1 - λ², which vanishes
  // at |λ| = 1. Physically, the fast magnetosonic speed approaches c in
  // magnetically dominated states (σ → ∞) but never reaches it for finite
  // temperature (cs² > 0). The margin 1e-12 is well below double precision
  // but keeps G ≳ 2e-12, preventing division-by-zero in the eigenvectors.
  constexpr Real LAMBDA_MAX = 1.0 - 1.0e-12;
  for (int n = 0; n < NRMHD; ++n) {
    lambda[n] = std::max(-LAMBDA_MAX, std::min(LAMBDA_MAX, lambda[n]));
  }
}

inline void FinalizeStateSRMHD(MeshBlock *pmb, RMHDState &state,
                              const bool rebuild_conserved = true) {
  state.W = std::sqrt(1.0 + SQR(state.util_n)
                    + SQR(state.util_t1) + SQR(state.util_t2));
  const Real oo_W = 1.0 / state.W;

  state.v_n  = state.util_n  * oo_W;
  state.v_t1 = state.util_t1 * oo_W;
  state.v_t2 = state.util_t2 * oo_W;

  state.b0 = state.Bn * state.util_n + state.Bt1 * state.util_t1
           + state.Bt2 * state.util_t2;
  state.b_n  = (state.Bn  + state.b0 * state.util_n)  * oo_W;
  state.b_t1 = (state.Bt1 + state.b0 * state.util_t1) * oo_W;
  state.b_t2 = (state.Bt2 + state.b0 * state.util_t2) * oo_W;
  state.bsq = -SQR(state.b0) + SQR(state.b_n)
            + SQR(state.b_t1) + SQR(state.b_t2);
  state.ptot = state.pgas + 0.5 * state.bsq;

  // Ideal-gas (gamma-law) thermodynamics
  const Real gamma_adi = pmb->peos->GetGamma();
  state.gamma_adi = gamma_adi;
  const Real gamma_adi_red = gamma_adi / (gamma_adi - 1.0);
  state.rhoh = state.rho + gamma_adi_red * state.pgas;
  state.h = state.rhoh / state.rho;
  state.cs2 = gamma_adi * state.pgas / state.rhoh;
  state.Emhd = state.rhoh + state.bsq;
  if (rebuild_conserved) {
    FillConservedFromFinalizedStateSRMHD(state);
  }
}

inline Real GetGammaLawGammaSRMHD(const RMHDState &state) {
  if (std::isfinite(state.gamma_adi) && state.gamma_adi > 1.0 + constants::GAMMA_ADI_MIN) {
    return state.gamma_adi;
  }
  if (state.pgas > constants::PRESSURE_MIN) {
    const Real gamma_adi = state.cs2 * state.rhoh / state.pgas;
    if (std::isfinite(gamma_adi) && gamma_adi > 1.0 + constants::GAMMA_ADI_MIN) {
      return gamma_adi;
    }
  }
  return 2.0;
}

inline void PackReducedState(const AthenaArray<Real> &cons,
                             const AthenaArray<Real> &bcc,
                             const int k, const int j, const int i,
                             const int ivx,
                             Real u[NRMHD]) {
  const RMHDDirMap map = GetRMHDDirMap(ivx);

  u[ID_R]   = cons(IDN, k, j, i);
  u[ISN_R]  = cons(map.isn,  k, j, i);
  u[IST1_R] = cons(map.ist1, k, j, i);
  u[IST2_R] = cons(map.ist2, k, j, i);
  u[ITAU_R] = cons(IEN, k, j, i) - cons(IDN, k, j, i);
  u[IBT1_R] = bcc(map.ibt1, k, j, i);
  u[IBT2_R] = bcc(map.ibt2, k, j, i);
}

inline void GetReducedStateSRMHD(const int k, const int j,
                                 const int ivx, const int il, const int iu,
                                 const AthenaArray<Real> &cons,
                                 const AthenaArray<Real> &bcc,
                                 AthenaArray<Real> &u) {
  for (int i = il; i <= iu; ++i) {
    Real u_loc[NRMHD];
    PackReducedState(cons, bcc, k, j, i, ivx, u_loc);
    for (int n = 0; n < NRMHD; ++n) {
      u(n, k, j, i) = u_loc[n];
    }
  }
}

// Cell-centered reduced flux. Uses cell-centered primitives AND cell-centered
// magnetic field components (bcc) throughout. This matches SRHD's convention
// (state entirely at cell center) and makes the flux array directly reusable
// by the HO stencil builder — no face-Bn override, no per-cell rebuild.
inline void GetFluxesSRMHD(const int k, const int j,
                           const int ivx, const int il, const int iu,
                           const AthenaArray<Real> &prim,
                           const AthenaArray<Real> &bcc,
                           const AthenaArray<Real> &u,
                           AthenaArray<Real> &f) {
  const RMHDDirMap map = GetRMHDDirMap(ivx);

  for (int i = il; i <= iu; ++i) {
    const Real util_n  = prim(map.ivn,  k, j, i);
    const Real util_t1 = prim(map.ivt1, k, j, i);
    const Real util_t2 = prim(map.ivt2, k, j, i);
    const Real pgas = prim(IPR, k, j, i);

    // All B components from cell-centered bcc for internal consistency
    // within the cell state.
    const Real bn  = bcc(map.ibn,  k, j, i);
    const Real bt1 = u(IBT1_R, k, j, i);
    const Real bt2 = u(IBT2_R, k, j, i);

    const Real W = std::sqrt(1.0 + SQR(util_n) + SQR(util_t1) + SQR(util_t2));
    const Real oo_W = 1.0 / W;

    const Real v_n  = util_n  * oo_W;
    const Real v_t1 = util_t1 * oo_W;
    const Real v_t2 = util_t2 * oo_W;

    const Real b0 = bn * util_n + bt1 * util_t1 + bt2 * util_t2;
    const Real b_n  = (bn  + b0 * util_n)  * oo_W;
    const Real b_t1 = (bt1 + b0 * util_t1) * oo_W;
    const Real b_t2 = (bt2 + b0 * util_t2) * oo_W;
    const Real b_sq = -SQR(b0) + SQR(b_n) + SQR(b_t1) + SQR(b_t2);
    const Real ptot = pgas + 0.5 * b_sq;

    f(ID_R,   k, j, i) = u(ID_R,   k, j, i) * v_n;
    f(ISN_R,  k, j, i) = u(ISN_R,  k, j, i) * v_n - b_n  * bn * oo_W + ptot;
    f(IST1_R, k, j, i) = u(IST1_R, k, j, i) * v_n - b_t1 * bn * oo_W;
    f(IST2_R, k, j, i) = u(IST2_R, k, j, i) * v_n - b_t2 * bn * oo_W;
    f(ITAU_R, k, j, i) =
        u(ITAU_R, k, j, i) * v_n + ptot * v_n - b0 * bn * oo_W;
    f(IBT1_R, k, j, i) = bt1 * v_n - bn * v_t1;
    f(IBT2_R, k, j, i) = bt2 * v_n - bn * v_t2;
  }
}

inline void GetReducedFluxFromStateSRMHD(const RMHDState &state,
                                         Real f[NRMHD]) {
  f[ID_R]   = state.D * state.v_n;
  f[ISN_R]  = state.S_n * state.v_n - state.b_n  * state.Bn  / state.W + state.ptot;
  f[IST1_R] = state.S_t1 * state.v_n - state.b_t1 * state.Bn / state.W;
  f[IST2_R] = state.S_t2 * state.v_n - state.b_t2 * state.Bn / state.W;
  f[ITAU_R] = state.tau * state.v_n + state.ptot * state.v_n
            - state.b0 * state.Bn / state.W;
  f[IBT1_R] = state.Bt1 * state.v_n - state.Bn * state.v_t1;
  f[IBT2_R] = state.Bt2 * state.v_n - state.Bn * state.v_t2;
}

inline void PackReducedStateFromStateSRMHD(const RMHDState &state,
                                           Real u[NRMHD]) {
  u[ID_R]   = state.D;
  u[ISN_R]  = state.S_n;
  u[IST1_R] = state.S_t1;
  u[IST2_R] = state.S_t2;
  u[ITAU_R] = state.tau;
  u[IBT1_R] = state.Bt1;
  u[IBT2_R] = state.Bt2;
}

// Build the 6-cell HO stencil at face (k, j, i) by reading from precomputed
// per-cell reduced-basis arrays. cons_stencil gets cell-averaged conserved
// (u_red), flx_stencil gets cell-centered flux (f_red, built with cell-center
// Bn matching SRHD convention), lambda_stencil gets per-cell eigenvalues.
// No per-cell state rebuild — all quantities are looked up from arrays that
// were filled once in RusanovFluxDir's top-of-function precompute loop.
inline void BuildFaceCompatibleStencilDataSRMHD(
    const int k, const int j, const int i,
    const int ivx,
    const AthenaArray<Real> &u,
    const AthenaArray<Real> &f,
    const AthenaArray<Real> &lambda,
    Real cons_stencil[6][NRMHD],
    Real flx_stencil[6][NRMHD],
    Real lambda_stencil[6][NRMHD]) {
  for (int s = 0; s < 6; ++s) {
    const int offset = s - 3;  // [-3, -2, -1, 0, +1, +2]
    int kk = k, jj = j, ii = i;
    switch (ivx) {
      case IVX: ii = i + offset; break;
      case IVY: jj = j + offset; break;
      default:  kk = k + offset; break;
    }
    for (int n = 0; n < NRMHD; ++n) {
      cons_stencil[s][n]   = u(n, kk, jj, ii);
      flx_stencil[s][n]    = f(n, kk, jj, ii);
      lambda_stencil[s][n] = lambda(n, kk, jj, ii);
    }
  }
}

#if GENERAL_RELATIVITY
// -----------------------------------------------------------------------------
// Antón 2006 tetrad-frame GRMHD pipeline for high-order characteristic
// reconstruction. Three-step recipe (Antón et al. 2006 §3.4, arXiv:astro-ph/
// 0506063, lines 1260–1264):
//   i)   Transform 6-cell stencil primitives into locally Minkowskian coords
//        at the face via Coordinates::StencilPrimToLocal[1|2|3]. Avg state at
//        the face comes from L (stencil[2]) and R (stencil[3]) post-transform
//        averaging — no separate L/R PrimToLocal call needed.
//   ii)  Build per-cell reduced state/flux/eigenvalues using the SR pipeline
//        (unchanged — correct in the local Minkowski frame by the equivalence
//        principle).
//   iii) After ReconFluxRMHD writes the reduced face flux in tetrad coords,
//        call FluxToGlobal[1|2|3] to transform back to global coordinates.
// Steps (i) and (ii) live in BuildFaceCompatibleStencilDataTetradGRMHD below;
// step (iii) is invoked by the caller (rusanov_mhd_rel.cpp) after the HO
// reconstruction completes, via CallFluxToGlobalSingle.
// -----------------------------------------------------------------------------

// Build an SR RMHDState from a single entry in the stack-allocated stencil
// array produced by Coordinates::StencilPrimToLocal[1|2|3]. The slot layout
// matches PrimToLocal[1|2|3]: tetrad-frame velocity components are direction-
// permuted into IVX/IVY/IVZ (so map.{ivn,ivt1,ivt2} reads the correct
// face-normal/tangential components), and tang B^(y),B^(z) live in IBY/IBZ.
inline void BuildRMHDStateFromStencilEntry(
    MeshBlock *pmb,
    const Real stencil_entry[NWAVE],
    const Real bbx_entry,
    const int ivx,
    RMHDState &state,
    const bool refinalize = true) {
  const RMHDDirMap map = GetRMHDDirMap(ivx);

  state.rho     = stencil_entry[IDN];
  state.pgas    = stencil_entry[IPR];
  state.util_n  = stencil_entry[map.ivn];
  state.util_t1 = stencil_entry[map.ivt1];
  state.util_t2 = stencil_entry[map.ivt2];
  state.Bn      = bbx_entry;
  state.Bt1     = stencil_entry[IBY];
  state.Bt2     = stencil_entry[IBZ];

  if (refinalize) {
    FinalizeStateSRMHD(pmb, state, true);
  }
}

// Dispatcher for FluxToGlobal: routes to the correct Coordinates member based
// on ivx. Operates on a single face (il=iu=i).
inline void CallFluxToGlobalSingle(MeshBlock *pmb,
                                   const int k, const int j, const int i,
                                   const int ivx,
                                   const AthenaArray<Real> &cons,
                                   const AthenaArray<Real> &bbx,
                                   AthenaArray<Real> &flux,
                                   AthenaArray<Real> &ey,
                                   AthenaArray<Real> &ez) {
  switch (ivx) {
    case IVX:
      pmb->pcoord->FluxToGlobal1(k, j, i, i, cons, bbx, flux, ey, ez);
      break;
    case IVY:
      pmb->pcoord->FluxToGlobal2(k, j, i, i, cons, bbx, flux, ey, ez);
      break;
    default:
      pmb->pcoord->FluxToGlobal3(k, j, i, i, cons, bbx, flux, ey, ez);
      break;
  }
}

// Tetrad-frame version of BuildFaceCompatibleStencilDataSRMHD. Follows
// Antón 2006 §3.4 step (i)+(ii): transforms each stencil cell into the local
// Minkowski frame at face (k,j,i), then uses the SR pipeline to compute the
// reduced state, physical flux, and characteristic eigenvalues.
//
// Also produces the tetrad-frame avg state used as the eigensystem
// linearization point — built by averaging the L (s=2) and R (s=3) entries
// of the already-transformed stencil. This avoids a redundant L/R PrimToLocal
// call (eliminated GetStateAvgTetradGRMHD). Result is byte-identical to the
// post-transform L/R avg the legacy code produced.
//
// Outputs:
//   cons_stencil[6][NRMHD], flx_stencil[6][NRMHD], lambda_stencil[6][NRMHD]
//   avg_state           — tetrad-frame avg, eigensystem linearization point
//   out_tetrad_cons(*,i) — tetrad cons in NWAVE layout (input to FluxToGlobal)
//   out_tetrad_bbx(i)    — tetrad face-normal B (input to FluxToGlobal)
inline void BuildFaceCompatibleStencilDataTetradGRMHD(
    MeshBlock *pmb,
    const int k, const int j, const int i,
    const int ivx,
    const AthenaArray<Real> &prim,
    const AthenaArray<Real> &bn_face,
    const AthenaArray<Real> &bcc,
    Real cons_stencil[6][NRMHD],
    Real flx_stencil[6][NRMHD],
    Real lambda_stencil[6][NRMHD],
    RMHDState &avg_state,
    AthenaArray<Real> &out_tetrad_cons,
    AthenaArray<Real> &out_tetrad_bbx) {
  Real stencil_prim[6][NWAVE];
  Real stencil_bbx[6];
  const Real bn_target = bn_face(k, j, i);

  // (i) Single stencil-wide tetrad transform via Coordinates virtual dispatch.
  switch (ivx) {
    case IVX:
      pmb->pcoord->StencilPrimToLocal1(k, j, i, bn_target, prim, bcc,
                                       stencil_prim, stencil_bbx);
      break;
    case IVY:
      pmb->pcoord->StencilPrimToLocal2(k, j, i, bn_target, prim, bcc,
                                       stencil_prim, stencil_bbx);
      break;
    default:
      pmb->pcoord->StencilPrimToLocal3(k, j, i, bn_target, prim, bcc,
                                       stencil_prim, stencil_bbx);
      break;
  }

  // (ii) Per-cell SR pipeline (unchanged math) on the tetrad-frame primitives.
  for (int s = 0; s < 6; ++s) {
    RMHDState state{};
    BuildRMHDStateFromStencilEntry(pmb, stencil_prim[s], stencil_bbx[s], ivx,
                                   state, /*refinalize=*/true);
    PackReducedStateFromStateSRMHD(state, cons_stencil[s]);
    GetReducedFluxFromStateSRMHD(state, flx_stencil[s]);
    GetEigenValuesSRMHD(state, lambda_stencil[s]);
  }

  // (iii) Tetrad-frame avg state at the face — average the L (s=2) and R
  // (s=3) entries of the already-transformed stencil. Mathematically
  // equivalent to the legacy GetStateAvgTetradGRMHD which transformed L/R via
  // a separate PrimToLocal call and averaged post-transform; we save that
  // duplicated work since the same transformation result is already in
  // stencil_prim[2,3] / stencil_bbx[2,3].
  // 2nd-order (L+R)/2 average of stencil[2] and stencil[3] post-tetrad.
  Real avg_prim[NWAVE];
  for (int n = 0; n < NWAVE; ++n) {
    avg_prim[n] = 0.5 * (stencil_prim[2][n] + stencil_prim[3][n]);
  }
  Real avg_bbx = 0.5 * (stencil_bbx[2] + stencil_bbx[3]);
  BuildRMHDStateFromStencilEntry(pmb, avg_prim, avg_bbx, ivx,
                                 avg_state, /*refinalize=*/true);

  // (iv) Populate tetrad_cons and tetrad_bbx for FluxToGlobal consumption.
  // FluxToGlobal expects Athena's NWAVE layout:
  //   cons(IDN)         = D
  //   cons(IEN)         = tau + D       (i.e. total SR energy E = T^00)
  //   cons(map.isn)     = S_n           (face-normal momentum)
  //   cons(map.ist1/2)  = S_t1, S_t2    (tangential momenta, direction-permuted)
  //   cons(IBY/IBZ)     = Bt1, Bt2 in tetrad frame
  const RMHDDirMap map = GetRMHDDirMap(ivx);
  out_tetrad_cons(IDN, i)      = avg_state.D;
  out_tetrad_cons(IEN, i)      = avg_state.tau + avg_state.D;
  out_tetrad_cons(map.isn,  i) = avg_state.S_n;
  out_tetrad_cons(map.ist1, i) = avg_state.S_t1;
  out_tetrad_cons(map.ist2, i) = avg_state.S_t2;
  out_tetrad_cons(IBY, i)      = avg_state.Bt1;
  out_tetrad_cons(IBZ, i)      = avg_state.Bt2;
  out_tetrad_bbx(i) = avg_bbx;
}
#endif  // GENERAL_RELATIVITY

inline void GetMaximalWaveSpeedStencilSRMHD(const Real lambda_stencil[6][NRMHD],
                                            Real lambda_max[NRMHD],
                                            const int s_min = 0,
                                            const int s_max = 5) {
  // s_min/s_max bound the stencil range over which the per-characteristic max
  // is taken. CS5 and WENO5/WENO5Z use the full [0,5] (default).
  for (int m = 0; m < NRMHD; ++m) {
    lambda_max[m] = 0.0;
    for (int s = s_min; s <= s_max; ++s) {
      lambda_max[m] = std::max(lambda_max[m], std::abs(lambda_stencil[s][m]));
    }
    // Causality clamp: |lambda| <= 1 (c=1 in Athena units). Eigenvalue roundoff
    // or ill-conditioned states can push |lambda| slightly above 1; clamping
    // prevents spurious over-dissipation in the LF split.
    lambda_max[m] = std::min(lambda_max[m], (Real)1.0);
  }
}

// High-order characteristic reconstruction of interface fluxes
// Algorithm:
// 1. Project physical fluxes to characteristic variables: char = L · (F ± λ_max·U)
// 2. Apply WENO5/WENO5Z reconstruction to each characteristic field
// 3. Combine left/right going waves: char_flux = flux^+ + flux^-
// 4. Transform back to physical space later: F = R · char_flux
inline void ReconCharFieldsStencilSRMHD(const Real flx_stencil[6][NRMHD],
                                        const Real cons_stencil[6][NRMHD],
                                        const Real lambda_max[NRMHD],
                                        const Real (&L_eig)[NRMHD][NRMHD],
                                        Real char_flx[NRMHD],
                                        const int ho_recon_kind) {
  constexpr Real fac = 0.5;  // Factor for local Lax-Friedrichs splitting

  // Loop over each characteristic field (7 waves in SRMHD)
  for (int m = 0; m < NRMHD; ++m) {
    Real flux_stencil_p[5] = {};  // Right-going characteristic flux
    Real flux_stencil_m[5] = {};  // Left-going characteristic flux
    const Real amax = lambda_max[m];  // Max wave speed for this characteristic

    // Project to characteristic space and apply Lax-Friedrichs splitting
    // F^± = 0.5 * L · (F ± λ_max·U)
    for (int s = 0; s < 5; ++s) {
      const int sp = s;      // Index for + direction stencil
      const int sm = 5 - s;  // Index for - direction stencil (reversed)
      for (int l = 0; l < NRMHD; ++l) {
        flux_stencil_p[s] += fac * L_eig[m][l]
                           * (flx_stencil[sp][l] + amax * cons_stencil[sp][l]);
        flux_stencil_m[s] += fac * L_eig[m][l]
                           * (flx_stencil[sm][l] - amax * cons_stencil[sm][l]);
      }
    }

    // Apply high-order reconstruction (WENO5/WENO5Z) and combine
    char_flx[m] = ReconstructScalarHO(flux_stencil_p, ho_recon_kind)
                + ReconstructScalarHO(flux_stencil_m, ho_recon_kind);
  }
}

inline void GetStateCellSRMHD(MeshBlock *pmb,
                              const int k, const int j, const int i,
                              const int ivx,
                              const AthenaArray<Real> &prim,
                              const AthenaArray<Real> &bcc,
                              const AthenaArray<Real> &u,
                              RMHDState &state) {
  const RMHDDirMap map = GetRMHDDirMap(ivx);

  state.rho = prim(IDN, k, j, i);
  state.pgas = prim(IPR, k, j, i);

  state.util_n  = prim(map.ivn,  k, j, i);
  state.util_t1 = prim(map.ivt1, k, j, i);
  state.util_t2 = prim(map.ivt2, k, j, i);

  state.Bn  = bcc(map.ibn,  k, j, i);
  state.Bt1 = bcc(map.ibt1, k, j, i);
  state.Bt2 = bcc(map.ibt2, k, j, i);

  state.D    = u(ID_R,   k, j, i);
  state.S_n  = u(ISN_R,  k, j, i);
  state.S_t1 = u(IST1_R, k, j, i);
  state.S_t2 = u(IST2_R, k, j, i);
  state.tau  = u(ITAU_R, k, j, i);

  FinalizeStateSRMHD(pmb, state, false);
}

inline void GetEigenValuesCellsSRMHD(MeshBlock *pmb,
                                     const int k, const int j,
                                     const int ivx, const int il, const int iu,
                                     const AthenaArray<Real> &prim,
                                     const AthenaArray<Real> &bcc,
                                     const AthenaArray<Real> &u,
                                     AthenaArray<Real> &lambda) {
  for (int i = il; i <= iu; ++i) {
    RMHDState state{};
    Real lambda_loc[NRMHD] = {};
    GetStateCellSRMHD(pmb, k, j, i, ivx, prim, bcc, u, state);
    GetEigenValuesSRMHD(state, lambda_loc);

    for (int n = 0; n < NRMHD; ++n) {
      lambda(n, k, j, i) = lambda_loc[n];
    }
  }
}

// skip_finalize: when true, only fills primitives and Bn in avg without
// calling FinalizeStateSRMHD. Useful when the caller will immediately
// re-finalize with a different metric.
inline void GetStateAvgSRMHD(MeshBlock *pmb,
                             const int k, const int j, const int i,
                             const int ivx,
                             const AthenaArray<Real> &prim,
                             const AthenaArray<Real> &bn_face,
                             const AthenaArray<Real> &bcc,
                             const AthenaArray<Real> &u,
                             RMHDState &avg,
                             const bool skip_finalize = false) {
  const RMHDDirMap map = GetRMHDDirMap(ivx);

  int kl = k, jl = j, il = i;
  int kr = k, jr = j, ir = i;

  switch (ivx) {
    case IVX:
      il = i - 1;
      ir = i;
      break;
    case IVY:
      jl = j - 1;
      jr = j;
      break;
    case IVZ:
      kl = k - 1;
      kr = k;
      break;
  }

  // SRHD-style averaging for robustness across shocks & to reduce bias on
  // smooth flows with large tangential 4-velocities:
  //   (i)  convert stored 4-velocity u^i = W·v^i to 3-velocity v^i per cell,
  //        average v^i, then rebuild util^i_avg = W_avg * v_avg
  //   (ii) average specific internal energy e = p/(rho*(gamma-1)), then
  //        reconstruct p_avg = (gamma-1)·rho_avg·e_avg
  // See SRHD GetStateAvgRHD (CharacteristicFieldsRHD.hpp L217-271) for
  // documented rationale: "physically bounded average |v|<1 across shocks"
  // and "more physical enthalpy across discontinuities".
  const Real un_L  = prim(map.ivn,  kl, jl, il);
  const Real ut1_L = prim(map.ivt1, kl, jl, il);
  const Real ut2_L = prim(map.ivt2, kl, jl, il);
  const Real WL    = std::sqrt(1.0 + SQR(un_L) + SQR(ut1_L) + SQR(ut2_L));
  const Real vn_L  = un_L  / WL;
  const Real vt1_L = ut1_L / WL;
  const Real vt2_L = ut2_L / WL;

  const Real un_R  = prim(map.ivn,  kr, jr, ir);
  const Real ut1_R = prim(map.ivt1, kr, jr, ir);
  const Real ut2_R = prim(map.ivt2, kr, jr, ir);
  const Real WR    = std::sqrt(1.0 + SQR(un_R) + SQR(ut1_R) + SQR(ut2_R));
  const Real vn_R  = un_R  / WR;
  const Real vt1_R = ut1_R / WR;
  const Real vt2_R = ut2_R / WR;

  const Real rhoL  = prim(IDN, kl, jl, il);
  const Real rhoR  = prim(IDN, kr, jr, ir);
  const Real pL    = prim(IPR, kl, jl, il);
  const Real pR    = prim(IPR, kr, jr, ir);

  const Real vn_avg  = 0.5 * (vn_L  + vn_R);
  const Real vt1_avg = 0.5 * (vt1_L + vt1_R);
  const Real vt2_avg = 0.5 * (vt2_L + vt2_R);
  const Real v2_avg  = SQR(vn_avg) + SQR(vt1_avg) + SQR(vt2_avg);
  const Real W_avg   = 1.0 / std::sqrt(std::max((Real)1.0e-30, 1.0 - v2_avg));

  avg.rho = 0.5 * (rhoL + rhoR);

  const Real gamma_adi = pmb->peos->GetGamma();
  const Real gmo = gamma_adi - 1.0;
  const Real eps_L = pL / (gmo * std::max(rhoL, (Real)1.0e-30));
  const Real eps_R = pR / (gmo * std::max(rhoR, (Real)1.0e-30));
  const Real eps   = 0.5 * (eps_L + eps_R);
  avg.pgas = gmo * avg.rho * eps;

  avg.util_n  = W_avg * vn_avg;
  avg.util_t1 = W_avg * vt1_avg;
  avg.util_t2 = W_avg * vt2_avg;

  // Normal B at the face: use the face-centered value (CT-pinned, div-B=0).
  // This is a FACE quantity, not a cell-center quantity, so it correctly
  // uses bn_face. Tangential B components come from cell-center averages.
  avg.Bn  = bn_face(k, j, i);
  avg.Bt1 = 0.5 * (bcc(map.ibt1, kl, jl, il) + bcc(map.ibt1, kr, jr, ir));
  avg.Bt2 = 0.5 * (bcc(map.ibt2, kl, jl, il) + bcc(map.ibt2, kr, jr, ir));

  const Real bn_l = bcc(map.ibn, kl, jl, il);
  const Real bn_r = bcc(map.ibn, kr, jr, ir);
  const Real bn_scale = std::max(
      {(Real)1.0, std::abs(avg.Bn), std::abs(bn_l), std::abs(bn_r)});
  const bool use_face_compatible_conserved =
      (std::abs(avg.Bn - bn_l) > 1.0e-14 * bn_scale) ||
      (std::abs(avg.Bn - bn_r) > 1.0e-14 * bn_scale);

  if (skip_finalize) {
    // Caller will finalize with a different metric (e.g., GR).
    // Just fill the conserved variables if Bn matches, otherwise leave
    // them for the caller's FinalizeState to recompute.
    if (!use_face_compatible_conserved) {
      avg.D    = 0.5 * (u(ID_R,   kl, jl, il) + u(ID_R,   kr, jr, ir));
      avg.S_n  = 0.5 * (u(ISN_R,  kl, jl, il) + u(ISN_R,  kr, jr, ir));
      avg.S_t1 = 0.5 * (u(IST1_R, kl, jl, il) + u(IST1_R, kr, jr, ir));
      avg.S_t2 = 0.5 * (u(IST2_R, kl, jl, il) + u(IST2_R, kr, jr, ir));
      avg.tau  = 0.5 * (u(ITAU_R, kl, jl, il) + u(ITAU_R, kr, jr, ir));
    }
  } else {
    if (use_face_compatible_conserved) {
      FinalizeStateSRMHD(pmb, avg, true);
    } else {
      avg.D    = 0.5 * (u(ID_R,   kl, jl, il) + u(ID_R,   kr, jr, ir));
      avg.S_n  = 0.5 * (u(ISN_R,  kl, jl, il) + u(ISN_R,  kr, jr, ir));
      avg.S_t1 = 0.5 * (u(IST1_R, kl, jl, il) + u(IST1_R, kr, jr, ir));
      avg.S_t2 = 0.5 * (u(IST2_R, kl, jl, il) + u(IST2_R, kr, jr, ir));
      avg.tau  = 0.5 * (u(ITAU_R, kl, jl, il) + u(ITAU_R, kr, jr, ir));

      FinalizeStateSRMHD(pmb, avg, false);
    }
  }
}

inline bool InvertMatrixRMHD(const Real (&A)[NRMHD][NRMHD],
                             Real (&Ainv)[NRMHD][NRMHD]) {
  long double aug[NRMHD][2*NRMHD] = {};
  for (int i = 0; i < NRMHD; ++i) {
    for (int j = 0; j < NRMHD; ++j) {
      aug[i][j] = static_cast<long double>(A[i][j]);
      aug[i][j + NRMHD] = (i == j ? 1.0L : 0.0L);
    }
  }

  for (int col = 0; col < NRMHD; ++col) {
    int pivot = col;
    long double piv_abs = std::abs(aug[col][col]);
    for (int row = col + 1; row < NRMHD; ++row) {
      const long double cand = std::abs(aug[row][col]);
      if (cand > piv_abs) {
        piv_abs = cand;
        pivot = row;
      }
    }
    if (piv_abs < constants::MATRIX_SINGULAR_TOL) return false;

    if (pivot != col) {
      for (int j = 0; j < 2*NRMHD; ++j) std::swap(aug[col][j], aug[pivot][j]);
    }

    const long double piv = aug[col][col];
    for (int j = 0; j < 2*NRMHD; ++j) aug[col][j] /= piv;

    for (int row = 0; row < NRMHD; ++row) {
      if (row == col) continue;
      const long double fac = aug[row][col];
      if (fac == 0.0L) continue;
      for (int j = 0; j < 2*NRMHD; ++j) aug[row][j] -= fac * aug[col][j];
    }
  }

  for (int i = 0; i < NRMHD; ++i) {
    for (int j = 0; j < NRMHD; ++j) {
      Ainv[i][j] = static_cast<Real>(aug[i][j + NRMHD]);
    }
  }
  return true;
}

inline bool GetAntonTangentialDataSRMHD(const RMHDState &s, const Real lambda,
                                        AntonTangentialDataSRMHD &data) {
  const Real u0 = s.W;
  const Real ux = s.util_n;
  const Real uy = s.util_t1;
  const Real uz = s.util_t2;
  const Real bx = s.Bn;
  const Real by = s.Bt1;
  const Real bz = s.Bt2;

  data.alpha1[0] = uz;
  data.alpha1[1] = lambda * uz;
  data.alpha1[2] = 0.0;
  data.alpha1[3] = u0 - lambda * ux;

  data.alpha2[0] = -uy;
  data.alpha2[1] = -lambda * uy;
  data.alpha2[2] = lambda * ux - u0;
  data.alpha2[3] = 0.0;

  data.alpha11 = -SQR(data.alpha1[0]) + SQR(data.alpha1[1])
               + SQR(data.alpha1[2]) + SQR(data.alpha1[3]);
  data.alpha12 = -data.alpha1[0]*data.alpha2[0] + data.alpha1[1]*data.alpha2[1]
               + data.alpha1[2]*data.alpha2[2] + data.alpha1[3]*data.alpha2[3];
  data.alpha22 = -SQR(data.alpha2[0]) + SQR(data.alpha2[1])
               + SQR(data.alpha2[2]) + SQR(data.alpha2[3]);

  const Real den = 1.0 - lambda * s.v_n;
  const Real den_basis = data.alpha11*data.alpha22 - SQR(data.alpha12);
  if (std::abs(den) < 1.0e-14) {
    SetLastEigenFailureSRMHD(10, den);
    return false;
  }
  if (std::abs(den_basis) < 1.0e-14) {
    SetLastEigenFailureSRMHD(11, den_basis);
    return false;
  }

  data.g1 = (by + lambda * s.v_t1 / den * bx) / u0;
  data.g2 = (bz + lambda * s.v_t2 / den * bx) / u0;
  const Real gnorm = std::sqrt(SQR(data.g1) + SQR(data.g2));
  if (gnorm > 1.0e-14) {
    data.f1 = data.g1 / gnorm;
    data.f2 = data.g2 / gnorm;
  } else {
    // Type II degeneracy (|g| → 0): pick the tangential basis from a physical
    // direction in priority order — velocity, lab B, fluid B, coord-aligned.
    const Real vt_norm = std::sqrt(SQR(s.v_t1) + SQR(s.v_t2));
    const Real Bt_norm = std::sqrt(SQR(s.Bt1) + SQR(s.Bt2));
    const Real bt_norm = std::sqrt(SQR(s.b_t1) + SQR(s.b_t2));

    if (vt_norm > 1.0e-14) {
      // Use velocity direction for continuity
      data.f1 = s.v_t1 / vt_norm;
      data.f2 = s.v_t2 / vt_norm;
    } else if (Bt_norm > 1.0e-14) {
      // Use lab-frame magnetic field direction
      data.f1 = s.Bt1 / Bt_norm;
      data.f2 = s.Bt2 / Bt_norm;
    } else if (bt_norm > 1.0e-14) {
      // Use fluid-frame magnetic field direction
      data.f1 = s.b_t1 / bt_norm;
      data.f2 = s.b_t2 / bt_norm;
    } else {
      // Last resort - but this should rarely happen
      // Use coordinate-aligned direction to be deterministic
      data.f1 = 1.0;
      data.f2 = 0.0;
    }
  }

  const Real c1 = (data.g1 * data.alpha12 + data.g2 * data.alpha22) / den_basis * u0 * den;
  const Real c2 = -(data.g1 * data.alpha11 + data.g2 * data.alpha12) / den_basis * u0 * den;
  for (int mu = 0; mu < 4; ++mu) {
    data.bt[mu] = c1 * data.alpha1[mu] + c2 * data.alpha2[mu];
  }

  Real bt_sq = -SQR(data.bt[0]) + SQR(data.bt[1]) + SQR(data.bt[2]) + SQR(data.bt[3]);
  bt_sq = std::max((Real)0.0, bt_sq);
  data.abs_bt = std::sqrt(bt_sq);

  // Antón et al. 2010 eq. (75): use the renormalized direction b_t / |b_t|
  // directly from the alpha-basis and (f1,f2). In exact Type-II states the
  // basis is not unique; when a preferred tangential line is supplied from
  // neighboring nondegenerate states, use it to continue the basis smoothly.
  const Real combo = SQR(data.f1) * data.alpha11
                   + 2.0 * data.f1 * data.f2 * data.alpha12
                   + SQR(data.f2) * data.alpha22;
  const Real dir_norm_sq = den_basis * combo;
  if (dir_norm_sq <= 1.0e-14) {
    SetLastEigenFailureSRMHD(12, dir_norm_sq);
    return false;
  }
  const Real dir_norm = std::sqrt(dir_norm_sq);
  const Real coeff1 = (data.f1 * data.alpha12 + data.f2 * data.alpha22) / dir_norm;
  const Real coeff2 = -(data.f1 * data.alpha11 + data.f2 * data.alpha12) / dir_norm;
  for (int mu = 0; mu < 4; ++mu) {
    data.bt_dir[mu] = coeff1 * data.alpha1[mu] + coeff2 * data.alpha2[mu];
  }

  if (data.abs_bt > 1.0e-14) {
    data.ghat1 = data.g1 / data.abs_bt;
    data.ghat2 = data.g2 / data.abs_bt;
  } else {
    // Antón et al. 2010 eq. (154): limit of g_i / |b_t| used by the
    // Type-II-renormalized left magnetosonic eigenvectors.
    const Real num_sq = std::max((Real)0.0,
                                 SQR(u0 - lambda * ux)
                                 - (1.0 - SQR(lambda)) * (SQR(uy) + SQR(uz)));
    const Real denom_eq154 = std::max((Real)1.0e-14, combo);
    const Real scale = std::sqrt(num_sq / denom_eq154);
    data.ghat1 = data.f1 * scale;
    data.ghat2 = data.f2 * scale;
  }

  return true;
}

struct PaperProductSummarySRMHD {
  Real max_diag_err = 0.0;
  Real max_offdiag = 0.0;
  int i_diag = -1;
  int i_off = -1;
  int j_off = -1;
};

inline void MultiplyMatVecPaperSRMHD(const Real (&M)[NRMHD][NCOV_RMHD],
                                     const Real vec[NCOV_RMHD],
                                     Real out[NRMHD]) {
  for (int i = 0; i < NRMHD; ++i) {
    out[i] = 0.0;
    for (int j = 0; j < NCOV_RMHD; ++j) out[i] += M[i][j] * vec[j];
  }
}

inline Real DotRowColPaperSRMHD(const Real row[NRMHD],
                                const Real (&R)[NRMHD][NRMHD],
                                const int col) {
  Real sum = 0.0;
  for (int k = 0; k < NRMHD; ++k) sum += row[k] * R[k][col];
  return sum;
}

inline bool NormalizeLeftRowsPaperSRMHD(Real (&L)[NRMHD][NRMHD],
                                        const Real (&R)[NRMHD][NRMHD]) {
  for (int i = 0; i < NRMHD; ++i) {
    const Real dot = DotRowColPaperSRMHD(L[i], R, i);
    if (std::abs(dot) < 1.0e-14) return false;
    for (int j = 0; j < NRMHD; ++j) L[i][j] /= dot;
  }
  return true;
}

inline PaperProductSummarySRMHD SummarizeProductPaperSRMHD(
    const Real (&L)[NRMHD][NRMHD], const Real (&R)[NRMHD][NRMHD]) {
  PaperProductSummarySRMHD out;
  for (int i = 0; i < NRMHD; ++i) {
    for (int j = 0; j < NRMHD; ++j) {
      Real sum = 0.0;
      for (int k = 0; k < NRMHD; ++k) sum += L[i][k] * R[k][j];
      if (i == j) {
        const Real err = std::abs(sum - 1.0);
        if (err > out.max_diag_err) {
          out.max_diag_err = err;
          out.i_diag = i;
        }
      } else if (std::abs(sum) > out.max_offdiag) {
        out.max_offdiag = std::abs(sum);
        out.i_off = i;
        out.j_off = j;
      }
    }
  }
  return out;
}

inline Real ComputePaperBOverASRMHD(const RMHDState &s, const Real lambda,
                                    const bool negative_class) {
  const Real a = s.util_n - lambda * s.W;
  const Real G = 1.0 - SQR(lambda);
  const Real factor_a = s.rhoh * (1.0 / s.cs2 - 1.0);
  const Real factor_b = -(s.rhoh + s.bsq / s.cs2);
  const Real radicand = std::max((Real)0.0, -factor_b - factor_a * SQR(a) / G);
  const Real sign = negative_class ? 1.0 : -1.0;
  return sign * std::sqrt(radicand);
}

inline void BuildPaperdUdUtSRMHD(const RMHDState &s,
                                 Real (&M)[NRMHD][NCOV_RMHD]) {
  const Real gamma_adi = s.gamma_adi;
  const Real drho_dp_s = s.rho / (gamma_adi * s.pgas);
  const Real drho_ds_p = -s.rho / gamma_adi;
  const Real drhoh_dp_s = drho_dp_s + gamma_adi / (gamma_adi - 1.0);
  const Real drhoh_ds_p = drho_ds_p;

  for (int i = 0; i < NRMHD; ++i) {
    for (int j = 0; j < NCOV_RMHD; ++j) M[i][j] = 0.0;
  }

  M[ID_R][IU0_COV] = s.rho;
  M[ID_R][IP_COV] = drho_dp_s * s.W;
  M[ID_R][IS_COV] = drho_ds_p * s.W;

  M[ISN_R][IU0_COV] = s.Emhd * s.util_n;
  M[ISN_R][IUN_COV] = s.Emhd * s.W;
  M[ISN_R][IB0_COV] = -2.0 * s.b0 * s.W * s.util_n - s.b_n;
  M[ISN_R][IBN_COV] = 2.0 * s.b_n * s.W * s.util_n - s.b0;
  M[ISN_R][IBT1_COV] = 2.0 * s.b_t1 * s.W * s.util_n;
  M[ISN_R][IBT2_COV] = 2.0 * s.b_t2 * s.W * s.util_n;
  M[ISN_R][IP_COV] = drhoh_dp_s * s.W * s.util_n;
  M[ISN_R][IS_COV] = drhoh_ds_p * s.W * s.util_n;

  M[IST1_R][IU0_COV] = s.Emhd * s.util_t1;
  M[IST1_R][IUT1_COV] = s.Emhd * s.W;
  M[IST1_R][IB0_COV] = -2.0 * s.b0 * s.W * s.util_t1 - s.b_t1;
  M[IST1_R][IBN_COV] = 2.0 * s.b_n * s.W * s.util_t1;
  M[IST1_R][IBT1_COV] = 2.0 * s.b_t1 * s.W * s.util_t1 - s.b0;
  M[IST1_R][IBT2_COV] = 2.0 * s.b_t2 * s.W * s.util_t1;
  M[IST1_R][IP_COV] = drhoh_dp_s * s.W * s.util_t1;
  M[IST1_R][IS_COV] = drhoh_ds_p * s.W * s.util_t1;

  M[IST2_R][IU0_COV] = s.Emhd * s.util_t2;
  M[IST2_R][IUT2_COV] = s.Emhd * s.W;
  M[IST2_R][IB0_COV] = -2.0 * s.b0 * s.W * s.util_t2 - s.b_t2;
  M[IST2_R][IBN_COV] = 2.0 * s.b_n * s.W * s.util_t2;
  M[IST2_R][IBT1_COV] = 2.0 * s.b_t1 * s.W * s.util_t2;
  M[IST2_R][IBT2_COV] = 2.0 * s.b_t2 * s.W * s.util_t2 - s.b0;
  M[IST2_R][IP_COV] = drhoh_dp_s * s.W * s.util_t2;
  M[IST2_R][IS_COV] = drhoh_ds_p * s.W * s.util_t2;

  M[ITAU_R][IU0_COV] = 2.0 * s.Emhd * s.W;
  M[ITAU_R][IB0_COV] = -2.0 * s.b0 * SQR(s.W) - s.b0;
  M[ITAU_R][IBN_COV] = 2.0 * s.b_n * SQR(s.W) - s.b_n;
  M[ITAU_R][IBT1_COV] = 2.0 * s.b_t1 * SQR(s.W) - s.b_t1;
  M[ITAU_R][IBT2_COV] = 2.0 * s.b_t2 * SQR(s.W) - s.b_t2;
  M[ITAU_R][IP_COV] = drhoh_dp_s * SQR(s.W) - 1.0;
  M[ITAU_R][IS_COV] = drhoh_ds_p * SQR(s.W);

  M[IBT1_R][IU0_COV] = s.b_t1;
  M[IBT1_R][IUT1_COV] = -s.b0;
  M[IBT1_R][IB0_COV] = -s.util_t1;
  M[IBT1_R][IBT1_COV] = s.W;

  M[IBT2_R][IU0_COV] = s.b_t2;
  M[IBT2_R][IUT2_COV] = -s.b0;
  M[IBT2_R][IB0_COV] = -s.util_t2;
  M[IBT2_R][IBT2_COV] = s.W;
}

inline bool ComputePaperdZdUSRMHD(const RMHDState &s, Real dZdU[NRMHD]) {
  const Real Z = s.rhoh * SQR(s.W);
  const Real B2 = SQR(s.Bn) + SQR(s.Bt1) + SQR(s.Bt2);
  const Real SB = s.S_n * s.Bn + s.S_t1 * s.Bt1 + s.S_t2 * s.Bt2;
  const Real S2 = SQR(s.S_n) + SQR(s.S_t1) + SQR(s.S_t2);
  const Real A = Z + B2;
  const Real H = (2.0 * Z + B2) / SQR(Z);
  const Real Q = SQR(A) - S2 - SQR(SB) * H;
  if (!(A > 0.0) || !(Q > 0.0)) {
    return false;
  }

  const Real gamma_adi = s.gamma_adi;
  const Real alpha = gamma_adi / (gamma_adi - 1.0);

  const auto eval_F_partial = [&](const Real dD, const Real dSn, const Real dSt1,
                                  const Real dSt2, const Real dTau, const Real dBt1,
                                  const Real dBt2, const Real dZ) {
    const Real dB2 = 2.0 * s.Bt1 * dBt1 + 2.0 * s.Bt2 * dBt2;
    const Real dSB = s.Bn * dSn + s.Bt1 * dSt1 + s.S_t1 * dBt1
                   + s.Bt2 * dSt2 + s.S_t2 * dBt2;
    const Real dS2 = 2.0 * s.S_n * dSn + 2.0 * s.S_t1 * dSt1 + 2.0 * s.S_t2 * dSt2;
    const Real dA = dZ + dB2;
    const Real dH = -2.0 * (Z + B2) / (Z * Z * Z) * dZ + dB2 / (Z * Z);
    const Real dQ = 2.0 * A * dA - dS2 - 2.0 * SB * H * dSB - SQR(SB) * dH;
    const Real du0 = s.W / A * dA - s.W / (2.0 * Q) * dQ;
    const Real dp = dZ - dTau
                  + (1.0 - 1.0 / (2.0 * SQR(s.W))) * dB2
                  + B2 * du0 / (s.W * s.W * s.W)
                  - SB * dSB / SQR(Z)
                  + SQR(SB) * dZ / (Z * Z * Z);
    return dZ - s.W * dD - s.D * du0
         - alpha * (SQR(s.W) * dp + 2.0 * s.pgas * s.W * du0);
  };

  const Real FZ = eval_F_partial(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0);
  if (!std::isfinite(FZ) || std::abs(FZ) < 1.0e-14) {
    return false;
  }

  dZdU[ID_R] = -eval_F_partial(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0) / FZ;
  dZdU[ISN_R] = -eval_F_partial(0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0) / FZ;
  dZdU[IST1_R] = -eval_F_partial(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0) / FZ;
  dZdU[IST2_R] = -eval_F_partial(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0) / FZ;
  dZdU[ITAU_R] = -eval_F_partial(0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0) / FZ;
  dZdU[IBT1_R] = -eval_F_partial(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0) / FZ;
  dZdU[IBT2_R] = -eval_F_partial(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0) / FZ;
  return true;
}

inline bool BuildPaperdVdUSRMHD(const RMHDState &s,
                                Real (&dVdU)[NRMHD][NRMHD]) {
  const Real Z = s.rhoh * SQR(s.W);
  const Real B2 = SQR(s.Bn) + SQR(s.Bt1) + SQR(s.Bt2);
  const Real denom = Z + B2;
  if (!(denom > 0.0)) {
    return false;
  }
  const Real SB = s.S_n * s.Bn + s.S_t1 * s.Bt1 + s.S_t2 * s.Bt2;
  const Real S2 = SQR(s.S_n) + SQR(s.S_t1) + SQR(s.S_t2);
  const Real A = denom;
  const Real H = (2.0 * Z + B2) / SQR(Z);
  const Real Q = SQR(A) - S2 - SQR(SB) * H;
  if (!(Q > 0.0)) {
    return false;
  }

  Real dZdU[NRMHD] = {};
  if (!ComputePaperdZdUSRMHD(s, dZdU)) return false;

  for (int i = 0; i < NRMHD; ++i) {
    for (int j = 0; j < NRMHD; ++j) dVdU[i][j] = 0.0;
  }

  const auto Ucomp = [&](const int i) {
    return (i == 0 ? s.util_n : (i == 1 ? s.util_t1 : s.util_t2));
  };
  const auto Bcomp = [&](const int i) {
    return (i == 0 ? s.Bn : (i == 1 ? s.Bt1 : s.Bt2));
  };
  const auto Scomp = [&](const int i) {
    return (i == 0 ? s.S_n : (i == 1 ? s.S_t1 : s.S_t2));
  };
  const auto bcomp = [&](const int i) {
    return (i == 0 ? s.b_n : (i == 1 ? s.b_t1 : s.b_t2));
  };

  for (int j = 0; j < NRMHD; ++j) {
    const Real dD = (j == ID_R) ? 1.0 : 0.0;
    const Real dSn = (j == ISN_R) ? 1.0 : 0.0;
    const Real dSt1 = (j == IST1_R) ? 1.0 : 0.0;
    const Real dSt2 = (j == IST2_R) ? 1.0 : 0.0;
    const Real dTau = (j == ITAU_R) ? 1.0 : 0.0;
    const Real dBt1 = (j == IBT1_R) ? 1.0 : 0.0;
    const Real dBt2 = (j == IBT2_R) ? 1.0 : 0.0;
    const Real dZ = dZdU[j];

    const Real dB2 = 2.0 * s.Bt1 * dBt1 + 2.0 * s.Bt2 * dBt2;
    const Real dSB = s.Bn * dSn + s.Bt1 * dSt1 + s.S_t1 * dBt1
                   + s.Bt2 * dSt2 + s.S_t2 * dBt2;
    const Real dS2 = 2.0 * s.S_n * dSn + 2.0 * s.S_t1 * dSt1 + 2.0 * s.S_t2 * dSt2;
    const Real dA = dZ + dB2;
    const Real dH = -2.0 * (Z + B2) / (Z * Z * Z) * dZ + dB2 / (Z * Z);
    const Real dQ = 2.0 * A * dA - dS2 - 2.0 * SB * H * dSB - SQR(SB) * dH;
    const Real du0 = s.W / A * dA - s.W / (2.0 * Q) * dQ;
    const Real db0 = (s.W * dSB + SB * du0 - s.b0 * dZ) / Z;

    const auto dScomp = [&](const int i) {
      return (i == 0 ? dSn : (i == 1 ? dSt1 : dSt2));
    };
    const auto dBcomp = [&](const int i) {
      if (i == 0) return (Real)0.0;
      return (i == 1 ? dBt1 : dBt2);
    };

    Real du[3] = {};
    Real db[3] = {};
    for (int i = 0; i < 3; ++i) {
      const Real Si = Scomp(i);
      const Real Bi = Bcomp(i);
      const Real ui = Ucomp(i);
      const Real dSi = dScomp(i);
      const Real dBi = dBcomp(i);
      du[i] = (Si * du0 + s.W * dSi + Bi * db0 + s.b0 * dBi) / A - ui * dA / A;
      const Real bi = bcomp(i);
      db[i] = dBi / s.W + (db0 * ui + s.b0 * du[i]) / s.W - bi * du0 / s.W;
    }

    const Real dp = dZ - dTau
                  + (1.0 - 1.0 / (2.0 * SQR(s.W))) * dB2
                  + B2 * du0 / (s.W * s.W * s.W)
                  - SB * dSB / SQR(Z)
                  + SQR(SB) * dZ / (Z * Z * Z);
    const Real drho = dD / s.W - s.D * du0 / SQR(s.W);

    dVdU[0][j] = du[0];
    dVdU[1][j] = du[1];
    dVdU[2][j] = du[2];
    dVdU[3][j] = db[1];
    dVdU[4][j] = db[2];
    dVdU[5][j] = dp;
    dVdU[6][j] = drho;
  }
  return true;
}

inline void BuildPaperCovariantRightEntropySRMHD(Real col[NCOV_RMHD]) {
  for (int n = 0; n < NCOV_RMHD; ++n) col[n] = 0.0;
  col[IS_COV] = 1.0;
}

inline bool BuildPaperCovariantRightAlfvenSRMHD(
    const RMHDState &s, const Real lambda, const bool plus_branch,
    Real col[NCOV_RMHD]) {
  AntonTangentialDataSRMHD data{};
  if (!GetAntonTangentialDataSRMHD(s, lambda, data)) return false;
  const Real sqrt_E = std::sqrt(std::max((Real)0.0, s.Emhd));
  const Real sgn = plus_branch ? 1.0 : -1.0;
  Real avec[4] = {};
  for (int mu = 0; mu < 4; ++mu) {
    avec[mu] = data.f1 * data.alpha1[mu] + data.f2 * data.alpha2[mu];
  }
  for (int n = 0; n < NCOV_RMHD; ++n) col[n] = 0.0;
  for (int mu = 0; mu < 4; ++mu) {
    col[IU0_COV + mu] = avec[mu];
    col[IB0_COV + mu] = -sgn * sqrt_E * avec[mu];
  }
  return true;
}

inline bool BuildPaperCovariantRightMagnetosonicSRMHD(
    const RMHDState &s, const Real lambda, const bool use_typeii,
    const bool negative_class, Real col[NCOV_RMHD]) {
  AntonTangentialDataSRMHD data{};
  if (!GetAntonTangentialDataSRMHD(s, lambda, data)) return false;
  const Real a = s.util_n - lambda * s.W;
  const Real G = 1.0 - SQR(lambda);
  // With eigenvalue clamping (|λ| ≤ 1 - 1e-12), G ≥ 2e-12.
  // Use a threshold below this to catch only truly degenerate cases.
  if (std::abs(G) < 1.0e-14) {
    return false;
  }
  const Real B_over_a = ComputePaperBOverASRMHD(s, lambda, negative_class);
  const Real phi_up[4] = {lambda, 1.0, 0.0, 0.0};
  const Real u_up[4] = {s.W, s.util_n, s.util_t1, s.util_t2};
  Real eprime[4] = {};
  Real Lprime[4] = {};
  Real C = 0.0;

  if (use_typeii) {
    const Real denom = SQR(a) - (G + SQR(a)) * s.cs2;
    if (std::abs(denom) < constants::QUARTIC_TOL) {
      C = 0.0;
    } else {
      C = -(G + SQR(a)) * s.cs2 / denom * data.abs_bt;
    }
    for (int mu = 0; mu < 4; ++mu) {
      eprime[mu] = -a * C / (s.rhoh * s.cs2 * (G + SQR(a)))
                 * (phi_up[mu] + a * u_up[mu])
                 - B_over_a / s.rhoh * data.bt_dir[mu];
      Lprime[mu] = -B_over_a * C / s.rhoh * u_up[mu]
                 - (1.0 + SQR(a) / G) * data.bt_dir[mu];
    }
  } else {
    const Real denom = s.rhoh * SQR(a) - s.bsq * G;
    const Real bt_coeff = (std::abs(denom) < 1.0e-14) ? 0.0 : (G / denom);
    C = -1.0;
    for (int mu = 0; mu < 4; ++mu) {
      eprime[mu] = a / (s.rhoh * s.cs2 * (G + SQR(a)))
                 * (phi_up[mu] + a * u_up[mu])
                 - B_over_a * bt_coeff * data.bt[mu] / s.rhoh;
      Lprime[mu] = B_over_a / s.rhoh * u_up[mu]
                 - (1.0 + SQR(a) / G) * bt_coeff * data.bt[mu];
    }
  }

  for (int n = 0; n < NCOV_RMHD; ++n) col[n] = 0.0;
  for (int mu = 0; mu < 4; ++mu) {
    col[IU0_COV + mu] = eprime[mu];
    col[IB0_COV + mu] = Lprime[mu];
  }
  col[IP_COV] = C;
  return true;
}

inline void FillReducedEntropyLeftEigenvectorSRMHD(const RMHDState &s,
                                                   Real row[NRMHD]) {
  const Real gamma_adi = GetGammaLawGammaSRMHD(s);
  const Real ds_dp_rho = 1.0 / s.pgas;
  const Real ds_drho_p = -gamma_adi / s.rho;

  row[0] = 0.0;
  row[1] = 0.0;
  row[2] = 0.0;
  row[3] = 0.0;
  row[4] = 0.0;
  row[5] = s.W * ds_dp_rho;
  row[6] = s.W * ds_drho_p;
}

inline void MapReducedLeftEigenvectorSRMHD(const Real reduced[NRMHD],
                                           const Real (&dVdU)[NRMHD][NRMHD],
                                           Real row[NRMHD]) {
  for (int j = 0; j < NRMHD; ++j) {
    row[j] = 0.0;
    for (int a = 0; a < NRMHD; ++a) {
      row[j] += reduced[a] * dVdU[a][j];
    }
  }
}

inline bool FillReducedAlfvenLeftEigenvectorSRMHD(const RMHDState &s,
                                                  const Real lambda,
                                                  const bool plus_branch,
                                                  Real row[NRMHD]) {
  const Real E = s.rhoh + s.bsq;
  const Real sqrt_E = std::sqrt(std::max((Real)0.0, E));
  const Real u0 = s.W;
  const Real ux = s.util_n;
  const Real uy = s.util_t1;
  const Real uz = s.util_t2;
  const Real bx = s.b_n;
  const Real by = s.b_t1;
  const Real bz = s.b_t2;
  const Real a = ux - lambda * u0;
  const Real Bcal = s.b_n - lambda * s.b0;
  const Real sgn = plus_branch ? 1.0 : -1.0;

  AntonTangentialDataSRMHD data{};
  if (!GetAntonTangentialDataSRMHD(s, lambda, data)) return false;

  const Real wa1[NRMHD] = {
      2.0 * (-E * a + s.b0 * Bcal / u0) * uz,
      (-E * u0 * uy + sgn * sqrt_E * (by * u0 - 2.0 * s.b0 * uy)) * uz / u0,
      E * (1.0 + SQR(uy))
          + sgn * sqrt_E * (s.b0 * (u0 - 2.0 * SQR(uz) / u0) + bz * uz - bx * ux),
      uz * (sgn * sqrt_E * uy + by),
      -by * uy - sgn * sqrt_E * (1.0 + SQR(uy)),
      uz,
      0.0
  };

  const Real wa2_base[NRMHD] = {
      2.0 * (-E * a + s.b0 * Bcal / u0) * uy,
      E * (1.0 + SQR(uz))
          + sgn * sqrt_E * (s.b0 * (u0 - 2.0 * SQR(uy) / u0) + by * uy - bx * ux),
      (-E * u0 * uz + sgn * sqrt_E * (bz * u0 - 2.0 * s.b0 * uz)) * uy / u0,
      -bz * uz - sgn * sqrt_E * (1.0 + SQR(uz)),
      uy * (sgn * sqrt_E * uz + bz),
      uy,
      0.0
  };

  for (int n = 0; n < NRMHD; ++n) {
    row[n] = data.f1 * wa1[n] - data.f2 * wa2_base[n];
  }
  return true;
}

inline bool FillReducedMagnetosonicLeftEigenvectorSRMHD(const RMHDState &s,
                                                        const Real lambda,
                                                        const bool use_typeii_form,
                                                        const bool negative_class,
                                                        Real row[NRMHD]) {
  const Real rhoh = s.rhoh;
  const Real cs2 = std::max((Real)1.0e-12, std::min((Real)1.0 - 1.0e-12, s.cs2));
  const Real u0 = s.W;
  const Real ux = s.util_n;
  const Real uy = s.util_t1;
  const Real uz = s.util_t2;
  const Real bx = s.b_n;
  const Real a = ux - lambda * u0;
  const Real G = 1.0 - SQR(lambda);
  const Real den_vel = u0 - lambda * ux;
  const Real eps = 1.0e-14;

  if (std::abs(G) < eps || std::abs(den_vel) < eps) return false;

  AntonTangentialDataSRMHD data{};
  if (!GetAntonTangentialDataSRMHD(s, lambda, data)) return false;

  const Real factor_a = rhoh * (1.0 / cs2 - 1.0);
  const Real factor_b = -(rhoh + s.bsq / cs2);
  const Real radicand = std::max((Real)0.0, -factor_b - factor_a * SQR(a) / G);
  // Keep the reduced left magnetosonic formulas on the same B/a branch as the
  // paper right route. Using a different sign convention here makes the active
  // runtime left/right bases disagree even when both sides build successfully.
  const Real B_over_a = (negative_class ? 1.0 : -1.0) * std::sqrt(radicand);

  const Real H = (G + 2.0 * SQR(a)) * (s.b0 - B_over_a * u0)
               - (bx * lambda - s.b0);

  if (use_typeii_form) {
    const Real denom = SQR(a) - (G + SQR(a)) * cs2;
    const Real bt_ratio =
        (std::abs(denom) < eps) ? 0.0 : ((1.0 - cs2) * data.abs_bt * a / denom);
    const Real bt_flux_ratio =
        (std::abs(denom) < eps) ? 0.0 : (data.abs_bt * G / (rhoh * denom));

    row[0] = (G + SQR(a)) / u0
             * (bt_ratio * (SQR(u0) - SQR(ux))
                + 2.0 * s.Bn * data.bt_dir[0]);
    row[1] = uy * (G + SQR(a)) / (u0 * den_vel)
             * (bt_ratio * ux * (ux * lambda - u0)
                - 2.0 * s.Bn * lambda * data.bt_dir[0])
           + data.ghat1 * H;
    row[2] = uz * (G + SQR(a)) / (u0 * den_vel)
             * (bt_ratio * ux * (ux * lambda - u0)
                - 2.0 * s.Bn * lambda * data.bt_dir[0])
           + data.ghat2 * H;
    row[3] = -data.ghat1 * den_vel;
    row[4] = -data.ghat2 * den_vel;
    row[5] = bt_flux_ratio * (lambda * ux - u0)
           - G / rhoh * B_over_a * data.bt_dir[0];
    row[6] = 0.0;
  } else {
    const Real denom = rhoh * SQR(a) - s.bsq * G;
    const Real bt_coeff = (std::abs(denom) < eps) ? 0.0 : (G / denom);

    row[0] = ((1.0 - cs2) * a / cs2 * (SQR(u0) - SQR(ux))
             + 2.0 * s.Bn * data.bt[0] * bt_coeff) / u0;
    row[1] = uy / (u0 * den_vel)
           * (((1.0 - cs2) * a * ux / cs2) * (ux * lambda - u0)
              - 2.0 * s.Bn * lambda * data.bt[0] * bt_coeff)
           + data.g1 * bt_coeff * H;
    row[2] = uz / (u0 * den_vel)
           * (((1.0 - cs2) * a * ux / cs2) * (ux * lambda - u0)
              - 2.0 * s.Bn * lambda * data.bt[0] * bt_coeff)
           + data.g2 * bt_coeff * H;
    row[3] = -data.g1 * bt_coeff * den_vel;
    row[4] = -data.g2 * bt_coeff * den_vel;
    row[5] = (lambda * ux - u0) * G / (rhoh * (G + SQR(a)) * cs2)
           - G / rhoh * data.bt[0] * B_over_a * bt_coeff;
    row[6] = 0.0;
  }

  return true;
}

inline bool CorrectBiorthogonalitySRMHD(const Real (&R)[NRMHD][NRMHD],
                                        Real (&L)[NRMHD][NRMHD]) {
  // Compute P = L · R
  Real P[NRMHD][NRMHD] = {};
  for (int i = 0; i < NRMHD; ++i)
    for (int j = 0; j < NRMHD; ++j)
      for (int k = 0; k < NRMHD; ++k)
        P[i][j] += L[i][k] * R[k][j];

  // Invert P (7×7) using the existing long-double Gauss-Jordan
  Real Pinv[NRMHD][NRMHD] = {};
  if (!InvertMatrixRMHD(P, Pinv)) return false;

  // L_corrected = Pinv · L
  Real L_new[NRMHD][NRMHD] = {};
  for (int i = 0; i < NRMHD; ++i)
    for (int j = 0; j < NRMHD; ++j)
      for (int k = 0; k < NRMHD; ++k)
        L_new[i][j] += Pinv[i][k] * L[k][j];

  for (int i = 0; i < NRMHD; ++i)
    for (int j = 0; j < NRMHD; ++j)
      L[i][j] = L_new[i][j];
  return true;
}

inline bool BuildPaperLeftMatrixWithChoicesSRMHD(
    const RMHDState &s, const Real lambda[NRMHD],
    const bool magneto_typeii[4], const bool magneto_negative_class[4],
    Real (&L)[NRMHD][NRMHD]) {
  Real dVdU[NRMHD][NRMHD] = {};
  if (!BuildPaperdVdUSRMHD(s, dVdU)) return false;

  Real reduced[NRMHD] = {};
  for (int i = 0; i < NRMHD; ++i) {
    for (int j = 0; j < NRMHD; ++j) L[i][j] = 0.0;
  }

  if (!FillReducedMagnetosonicLeftEigenvectorSRMHD(
          s, lambda[0], magneto_typeii[0], magneto_negative_class[0], reduced))
    return false;
  MapReducedLeftEigenvectorSRMHD(reduced, dVdU, L[0]);

  if (!FillReducedAlfvenLeftEigenvectorSRMHD(s, lambda[1], false, reduced))
    return false;
  MapReducedLeftEigenvectorSRMHD(reduced, dVdU, L[1]);

  if (!FillReducedMagnetosonicLeftEigenvectorSRMHD(
          s, lambda[2], magneto_typeii[1], magneto_negative_class[1], reduced))
    return false;
  MapReducedLeftEigenvectorSRMHD(reduced, dVdU, L[2]);

  FillReducedEntropyLeftEigenvectorSRMHD(s, reduced);
  MapReducedLeftEigenvectorSRMHD(reduced, dVdU, L[3]);

  if (!FillReducedMagnetosonicLeftEigenvectorSRMHD(
          s, lambda[4], magneto_typeii[2], magneto_negative_class[2], reduced))
    return false;
  MapReducedLeftEigenvectorSRMHD(reduced, dVdU, L[4]);

  if (!FillReducedAlfvenLeftEigenvectorSRMHD(s, lambda[5], true, reduced))
    return false;
  MapReducedLeftEigenvectorSRMHD(reduced, dVdU, L[5]);

  if (!FillReducedMagnetosonicLeftEigenvectorSRMHD(
          s, lambda[6], magneto_typeii[3], magneto_negative_class[3], reduced))
    return false;
  MapReducedLeftEigenvectorSRMHD(reduced, dVdU, L[6]);
  return true;
}

inline bool BuildPaperLeftMatrixSRMHD(const RMHDState &s, const Real lambda[NRMHD],
                                      Real (&L)[NRMHD][NRMHD]) {
  bool use_typeii[NRMHD] = {};
  GetLeftMagnetosonicRenormBranchSRMHD(lambda, use_typeii);
  const bool magneto_typeii[4] = {use_typeii[0], use_typeii[2], use_typeii[4], use_typeii[6]};
  const bool magneto_negative_class[4] = {true, true, false, false};
  return BuildPaperLeftMatrixWithChoicesSRMHD(
      s, lambda, magneto_typeii, magneto_negative_class, L);
}

inline bool NormalizeLeftEigenvectorsSRMHD(const Real (&R)[NRMHD][NRMHD],
                                           Real (&L)[NRMHD][NRMHD]) {
  return NormalizeLeftRowsPaperSRMHD(L, R);
}

inline void GetRightMagnetosonicRenormBranchSRMHD(const Real lambda[NRMHD],
                                                  bool use_typeii_form[NRMHD]) {
  for (int n = 0; n < NRMHD; ++n) use_typeii_form[n] = false;

  // Antón et al. 2010 choose the eqs. (71)-(75) branch for the magnetosonic
  // wave closest to the corresponding Alfvén wave in each left/right class,
  // and eqs. (76)-(79) for the other magnetosonic wave.
  use_typeii_form[0] =
      (std::abs(lambda[0] - lambda[1]) <= std::abs(lambda[2] - lambda[1]));
  use_typeii_form[2] = !use_typeii_form[0];
  use_typeii_form[4] =
      (std::abs(lambda[4] - lambda[5]) <= std::abs(lambda[6] - lambda[5]));
  use_typeii_form[6] = !use_typeii_form[4];
}

inline void GetLeftMagnetosonicRenormBranchSRMHD(const Real lambda[NRMHD],
                                                 bool use_typeii_form[NRMHD]) {
  // Keep the analytic left/right magnetosonic bases on the same Antón branch
  // choice. Using different renormalized branches for projection and
  // projection-back breaks the exact duality of the analytic eigensystem and
  // can leak planar states into the orthogonal tangential subspace.
  GetRightMagnetosonicRenormBranchSRMHD(lambda, use_typeii_form);
}

inline bool BuildPaperRightMatrixWithChoicesSRMHD(
    const RMHDState &s, const Real lambda[NRMHD],
    const bool magneto_typeii[4], const bool magneto_negative_class[4],
    Real (&R)[NRMHD][NRMHD]) {
  Real dUdUt[NRMHD][NCOV_RMHD] = {};
  BuildPaperdUdUtSRMHD(s, dUdUt);

  Real cov[NCOV_RMHD] = {};
  Real col[NRMHD] = {};
  for (int i = 0; i < NRMHD; ++i) {
    for (int j = 0; j < NRMHD; ++j) R[i][j] = 0.0;
  }

  if (!BuildPaperCovariantRightMagnetosonicSRMHD(
          s, lambda[0], magneto_typeii[0], magneto_negative_class[0], cov))
    return false;
  MultiplyMatVecPaperSRMHD(dUdUt, cov, col);
  for (int i = 0; i < NRMHD; ++i) R[i][0] = col[i];

  if (!BuildPaperCovariantRightAlfvenSRMHD(s, lambda[1], false, cov)) return false;
  MultiplyMatVecPaperSRMHD(dUdUt, cov, col);
  for (int i = 0; i < NRMHD; ++i) R[i][1] = col[i];

  if (!BuildPaperCovariantRightMagnetosonicSRMHD(
          s, lambda[2], magneto_typeii[1], magneto_negative_class[1], cov))
    return false;
  MultiplyMatVecPaperSRMHD(dUdUt, cov, col);
  for (int i = 0; i < NRMHD; ++i) R[i][2] = col[i];

  BuildPaperCovariantRightEntropySRMHD(cov);
  MultiplyMatVecPaperSRMHD(dUdUt, cov, col);
  for (int i = 0; i < NRMHD; ++i) R[i][3] = col[i];

  if (!BuildPaperCovariantRightMagnetosonicSRMHD(
          s, lambda[4], magneto_typeii[2], magneto_negative_class[2], cov))
    return false;
  MultiplyMatVecPaperSRMHD(dUdUt, cov, col);
  for (int i = 0; i < NRMHD; ++i) R[i][4] = col[i];

  if (!BuildPaperCovariantRightAlfvenSRMHD(s, lambda[5], true, cov)) return false;
  MultiplyMatVecPaperSRMHD(dUdUt, cov, col);
  for (int i = 0; i < NRMHD; ++i) R[i][5] = col[i];

  if (!BuildPaperCovariantRightMagnetosonicSRMHD(
          s, lambda[6], magneto_typeii[3], magneto_negative_class[3], cov))
    return false;
  MultiplyMatVecPaperSRMHD(dUdUt, cov, col);
  for (int i = 0; i < NRMHD; ++i) R[i][6] = col[i];

  return true;
}

inline bool BuildPaperRightMatrixSRMHD(const RMHDState &s, const Real lambda[NRMHD],
                                       Real (&R)[NRMHD][NRMHD]) {
  bool use_typeii[NRMHD] = {};
  GetRightMagnetosonicRenormBranchSRMHD(lambda, use_typeii);
  const bool magneto_typeii[4] = {use_typeii[0], use_typeii[2], use_typeii[4], use_typeii[6]};
  const bool magneto_negative_class[4] = {true, true, false, false};
  return BuildPaperRightMatrixWithChoicesSRMHD(
      s, lambda, magneto_typeii, magneto_negative_class, R);
}

inline bool ScaleRightEigenvectorsSRMHD(Real (&R)[NRMHD][NRMHD]) {
  for (int col = 0; col < NRMHD; ++col) {
    Real norm2 = 0.0;
    Real scale = 0.0;
    for (int row = 0; row < NRMHD; ++row) {
      const Real val = R[row][col];
      norm2 += SQR(val);
      scale = std::max(scale, std::abs(val));
    }
    if (!std::isfinite(norm2) || norm2 < 1.0e-24) {
      SetLastEigenFailureSRMHD(100 + col, norm2);
      return false;
    }
    if (!(scale > 0.0) || !std::isfinite(scale)) {
      SetLastEigenFailureSRMHD(120 + col, scale);
      return false;
    }
    const Real inv_scale = 1.0 / scale;
    for (int row = 0; row < NRMHD; ++row) {
      R[row][col] *= inv_scale;
    }
  }
  return true;
}

inline bool GetRightEigenVectorSRMHD(const RMHDState &avg,
                                     const Real lambda[NRMHD],
                                     Real (&R)[NRMHD][NRMHD]) {
  if (!BuildPaperRightMatrixSRMHD(avg, lambda, R)) return false;
  return ScaleRightEigenvectorsSRMHD(R);
}

inline bool GetLeftEigenVectorSRMHD(const RMHDState &avg,
                                    const Real lambda[NRMHD],
                                    const Real (&R)[NRMHD][NRMHD],
                                    Real (&L)[NRMHD][NRMHD]) {
  // Adaptive biorthogonality tolerance: in magnetically dominated states
  // (high σ = b²/2ρ), the eigenvectors become nearly degenerate (especially
  // at Type I where Bn → 0 and 5 eigenvalues collapse). The covariant-to-
  // conserved Jacobian multiplication amplifies small errors proportionally
  // to the condition number, which scales with σ. Allow larger L·R errors
  // for high-σ states — the characteristic reconstruction still converges
  // because the nearly-degenerate waves carry negligible physical information.
  const Real sigma = (avg.rho > 0.0) ? avg.bsq / (2.0 * avg.rho) : 0.0;
  const Real beta_inv = (avg.pgas > 0.0) ? 0.5 * avg.bsq / avg.pgas : 0.0;
  // Scale tolerance with both σ and inverse plasma beta.
  // At Type I degeneracy (Bn=0) with high magnetization, the nearly-degenerate
  // magnetosonic waves produce L·R errors ~O(σ * machine_eps_amplified).
  // The factor (1 + σ + β⁻¹) captures both rest-mass and thermal magnetization.
  const Real mag_factor = 1.0 + sigma + beta_inv;
  const Real kRuntimeOrthoTol = (mag_factor > 10.0)
      ? std::min(1.0e-2, 1.0e-6 * mag_factor)
      : 1.0e-6;

  if (!BuildPaperLeftMatrixSRMHD(avg, lambda, L)) {
    SetLastEigenFailureSRMHD(200);
    return false;
  }

  // Correct the biorthogonality error from the covariant-to-conserved chain.
  // L_corrected = (L·R)^{-1} · L guarantees L·R = I to machine precision.
  // This is a 7×7 solve — cheaper than full R^{-1} since the analytic L
  // provides a good starting point (P = L·R ≈ I with ~1% error).
  if (!CorrectBiorthogonalitySRMHD(R, L)) {
    // Correction failed (L·R was singular) — fall back to diagonal normalization
    if (!NormalizeLeftEigenvectorsSRMHD(R, L)) {
      SetLastEigenFailureSRMHD(220);
      return false;
    }
  }
  const Real max_lr_err = CheckEigenSystemSRMHD(L, R);
  if (!std::isfinite(max_lr_err) || max_lr_err >= kRuntimeOrthoTol) {
    SetLastEigenFailureSRMHD(221, max_lr_err);
    return false;
  }
  return true;
}

inline Real CheckEigenSystemSRMHD(const Real (&L)[NRMHD][NRMHD],
                                  const Real (&R)[NRMHD][NRMHD]) {
  const PaperProductSummarySRMHD summary = SummarizeProductPaperSRMHD(L, R);
  return std::max(summary.max_diag_err, summary.max_offdiag);
}

// ============================================================================
// EIGENVECTOR CONSTRUCTION VIA COVARIANT ROUTE
// ============================================================================
// Anton et al. (2010) provides both covariant and conserved formulations.
// This implementation uses the covariant route:
//   1. Build eigenvectors in covariant variables (V)
//   2. Transform to conserved variables (U) via Jacobian dU/dV
// ============================================================================
inline bool GetEigenVectorSRMHD(const RMHDState &avg,
                                const Real lambda[NRMHD],
                                Real (&L)[NRMHD][NRMHD],
                                Real (&R)[NRMHD][NRMHD]) {
  SetLastEigenFailureSRMHD(0);

  if (!GetRightEigenVectorSRMHD(avg, lambda, R)) {
    SetLastEigenFailureSRMHD(200);
    return false;
  }

  if (GetLeftEigenVectorSRMHD(avg, lambda, R, L)) {
    SetLastEigenFailureSRMHD(0);
    return true;
  }

  // Left eigenvector construction failed, try matrix inversion as fallback
  const Real left_metric = LastEigenFailureMetricSRMHD();
  if (InvertMatrixRMHD(R, L)) {
    SetLastEigenFailureSRMHD(500, left_metric);
    return true;
  }

  SetLastEigenFailureSRMHD(520, left_metric);
  return false;
}

inline void WriteReducedFlux(const Real fhat[NRMHD],
                             const int k, const int j, const int i,
                             const int ivx,
                             AthenaArray<Real> &flux) {
  const RMHDDirMap map = GetRMHDDirMap(ivx);

  flux(IDN, k, j, i) = fhat[ID_R];
  flux(map.isn,  k, j, i) = fhat[ISN_R];
  flux(map.ist1, k, j, i) = fhat[IST1_R];
  flux(map.ist2, k, j, i) = fhat[IST2_R];
  // Reduced basis evolves tau = E - D, so flux(E) = flux(tau) + flux(D).
  // In GR the caller hands us a tetrad-frame flux; FluxToGlobal then maps
  // (tau+D) -> the global T^i_{,0} convention via the face metric.
  flux(IEN, k, j, i) = fhat[ITAU_R] + fhat[ID_R];

  // Tangential magnetic fluxes are not stored in Hydro::flux. Athena carries
  // those through the separate face-centered EMF arrays used by CT.
}

inline void WriteReducedEMF(const Real fhat[NRMHD],
                            const int k, const int j, const int i,
                            AthenaArray<Real> &emf_t1,
                            AthenaArray<Real> &emf_t2) {
  // In the reduced normal/tangential basis,
  //   F(B_t1) = -EMF_t1, F(B_t2) = EMF_t2
  // where the call site chooses which physical EMF components correspond to
  // the first and second tangential directions for the given interface normal.
  emf_t1(k, j, i) = -fhat[IBT1_R];
  emf_t2(k, j, i) =  fhat[IBT2_R];
}

inline void ProjectBackFluxRMHD(const Real char_flx[NRMHD],
                                const Real (&R_eig)[NRMHD][NRMHD],
                                Real rflx[NRMHD]) {
  for (int m = 0; m < NRMHD; ++m) {
    rflx[m] = 0.0;
  }
  for (int m = 0; m < NRMHD; ++m) {
    for (int n = 0; n < NRMHD; ++n) {
      rflx[m] += R_eig[m][n] * char_flx[n];
    }
  }
}

inline void ReconFluxRMHD(const int k, const int j, const int i, const int ivx,
                          const Real char_flx[NRMHD],
                          const Real (&R_eig)[NRMHD][NRMHD],
                          AthenaArray<Real> &flx,
                          AthenaArray<Real> &emf_t1,
                          AthenaArray<Real> &emf_t2) {
  Real rflx[NRMHD] = {};
  ProjectBackFluxRMHD(char_flx, R_eig, rflx);
  WriteReducedFlux(rflx, k, j, i, ivx, flx);
  WriteReducedEMF(rflx, k, j, i, emf_t1, emf_t2);
}

}  // namespace characterisiticfields::rmhd

#endif  // CHARACTERISTIC_FIELDS_RMHD_HPP_
