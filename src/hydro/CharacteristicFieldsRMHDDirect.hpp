#ifndef HYDRO_CHARACTERISTIC_FIELDS_RMHD_DIRECT_HPP_
#define HYDRO_CHARACTERISTIC_FIELDS_RMHD_DIRECT_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file CharacteristicFieldsRMHDDirect.hpp
//! \brief Direct conserved-variable right eigenvectors for SRMHD (Antón et al. 2010 §5).
//!
//! This header provides an alternate route for the SRMHD 7×7 characteristic
//! decomposition used by the componentwise HO / EFL blending pipeline.  It
//! replaces the covariant-first-then-transform pipeline that lives in
//! CharacteristicFieldsRMHD.hpp with the direct conserved-variable formulas from
//! Antón et al. 2010, ApJS, "Relativistic Magnetohydrodynamics: Renormalized
//! Eigenvectors and Full Wave Decomposition Riemann Solver" §5.
//!
//! Rationale
//! ---------
//! The covariant route builds r-tilde (10 slots) in covariant variables, then
//! multiplies by the 7×10 dU/dU-tilde Jacobian to get the conserved-variable
//! right eigenvector R.  Near degeneracies (b_x → 0, b² → 0, c_s² − b²/E → 0)
//! and at high magnetization (cond(R) → 10⁸ – 10¹²), the Jacobian
//! multiplication accumulates round-off errors that leak into the L·R
//! biorthogonality check downstream.
//!
//! The direct route uses the §5.2 closed-form conserved-variable expressions,
//! avoiding the Jacobian chain entirely.  For the first implementation we
//! obtain L = R⁻¹ via long-double Gauss–Jordan inversion (existing
//! InvertMatrixRMHD helper), deferring the direct §6 left-eigenvector formulas
//! until this route is validated on the SR-MHD rotor problem.
//!
//! Wave ordering (must match GetEigenValuesSRMHD in the base header)
//! ------------------------------------------------------------------
//! Column of R  |  lambda index  |  Wave family
//!     0             lambda[0]      fast magnetosonic minus  (λ⁻_f)
//!     1             lambda[1]      Alfvén minus             (λ⁻_a)
//!     2             lambda[2]      slow magnetosonic minus  (λ⁻_s)
//!     3             lambda[3]      entropic                 (v_n)
//!     4             lambda[4]      slow magnetosonic plus   (λ⁺_s)
//!     5             lambda[5]      Alfvén plus              (λ⁺_a)
//!     6             lambda[6]      fast magnetosonic plus   (λ⁺_f)
//!
//! Row indexing follows the reduced (7-slot) conserved layout defined in the
//! base header (ID_R, ISN_R, IST1_R, IST2_R, ITAU_R, IBT1_R, IBT2_R).  The
//! sweep-normal direction "n" corresponds to "x" in the paper's notation, and
//! tangentials "t1" and "t2" map to "y" and "z" respectively.
//========================================================================================

#include <algorithm>
#include <cmath>
#include <cstdlib>

#include "../athena.hpp"                    // Real, SQR, macros
#include "CharacteristicFieldsRMHD.hpp"     // RMHDState, NRMHD, ID_R, etc.

namespace characterisiticfields::rmhd {
namespace direct {

//----------------------------------------------------------------------------------------
//! Strict Antón-paper tangential-basis helper.
//!
//! Calls the shared GetAntonTangentialDataSRMHD to compute α₁, α₂, α_{ij}, g₁, g₂,
//! b_t^μ and |b_t|, then STRICTLY follows the paper's Type II prescription for
//! (f₁, f₂):
//!
//!   * Non-Type-II  (√(g₁² + g₂²) > threshold):  f₁ = g₁/‖g‖,  f₂ = g₂/‖g‖.
//!   * Type II       (√(g₁² + g₂²) ≤ threshold):  f₁ = f₂ = 1/√2  (paper §5.2 text
//!                                                after Alfvén / magnetosonic
//!                                                subsections).
//!
//! No fallback ladder to v_t / B_t / b_t — paper does not use these.  When (f₁, f₂)
//! is overridden, b_t^μ/|b_t| (data.bt_dir) is recomputed using Antón eq. (75) so
//! the direction is consistent with the strict (1/√2, 1/√2) choice.
//!
//! Threshold matches the Athena++ base helper (1e-14).
//----------------------------------------------------------------------------------------
inline bool GetAntonTangentialDataStrictSRMHD(
    const RMHDState &s, const Real lambda,
    AntonTangentialDataSRMHD &data) {
  if (!GetAntonTangentialDataSRMHD(s, lambda, data)) return false;

  const Real gnorm = std::sqrt(SQR(data.g1) + SQR(data.g2));
  if (gnorm > 1.0e-14) return true;    // non-degenerate — base helper's (f1,f2) already correct

  // Type II: enforce paper's f_{1,2} = 1/√2 and recompute b_t^μ/|b_t| from Antón eq. (75).
  const Real inv_sqrt2 = 1.0 / std::sqrt((Real)2.0);
  data.f1 = inv_sqrt2;
  data.f2 = inv_sqrt2;

  const Real combo = SQR(data.f1) * data.alpha11
                    + 2.0 * data.f1 * data.f2 * data.alpha12
                    + SQR(data.f2) * data.alpha22;
  const Real den_basis = data.alpha11 * data.alpha22 - SQR(data.alpha12);
  const Real dir_norm_sq = den_basis * combo;
  if (dir_norm_sq <= 1.0e-14) return false;

  const Real dir_norm = std::sqrt(dir_norm_sq);
  const Real coeff1 =  (data.f1 * data.alpha12 + data.f2 * data.alpha22) / dir_norm;
  const Real coeff2 = -(data.f1 * data.alpha11 + data.f2 * data.alpha12) / dir_norm;
  for (int mu = 0; mu < 4; ++mu) {
    data.bt_dir[mu] = coeff1 * data.alpha1[mu] + coeff2 * data.alpha2[mu];
  }
  return true;
}

//----------------------------------------------------------------------------------------
//! Antón §5.2 Eq. (eq:entro.cons) — Entropic right eigenvector in conserved variables.
//! R_e = u⁰ · ( ∂ρ/∂s,  u^x ∂(ρh)/∂s,  u^y ∂(ρh)/∂s,  u^z ∂(ρh)/∂s,
//!               u⁰ ∂(ρh)/∂s,  0,  0 )ᵀ
//!
//! For a Γ-law EOS: ∂ρ/∂s|_p = -ρ/Γ,  ∂(ρh)/∂s|_p = ∂ρ/∂s|_p
//----------------------------------------------------------------------------------------
inline void BuildDirectRightEntropicSRMHD(const RMHDState &s,
                                          Real R_col[NRMHD]) {
  const Real gamma_adi = s.gamma_adi;
  const Real drho_ds_p  = -s.rho / gamma_adi;
  const Real drhoh_ds_p = drho_ds_p;           // for Γ-law
  const Real W = s.W;

  R_col[ID_R]   = W * drho_ds_p;
  R_col[ISN_R]  = W * s.util_n  * drhoh_ds_p;
  R_col[IST1_R] = W * s.util_t1 * drhoh_ds_p;
  R_col[IST2_R] = W * s.util_t2 * drhoh_ds_p;
  R_col[ITAU_R] = W * W         * drhoh_ds_p;
  R_col[IBT1_R] = 0.0;
  R_col[IBT2_R] = 0.0;
}

//----------------------------------------------------------------------------------------
//! Antón §5.2 Eq. (eq:alfven.cons) — Alfvén right eigenvector in conserved variables.
//!
//! R_{a,±} = f₁ V_{a,1,±} + f₂ V_{a,2,±}
//!
//! where V_{a,1,±} and V_{a,2,±} are the (7-slot) column-vector expressions given
//! in Section 5.2 of Antón et al. 2010.  Sign convention:
//!   plus_branch = false  →  R_{a,−}  (paper's "−" sign in ±)  — matches lambda[1]
//!   plus_branch = true   →  R_{a,+}  (paper's "+" sign in ±)  — matches lambda[5]
//!
//! Type II degeneracy (g₁ = g₂ = 0): f₁ = f₂ = 1/√2 is enforced by
//! AntonTangentialDataSRMHD (populates s → tangent.f1, f2 via g_normal check).
//! Type I degeneracy (B^n = 0): both V₁ and V₂ are individually regular.
//----------------------------------------------------------------------------------------
inline void BuildDirectRightAlfvenSRMHD(
    const RMHDState &s,
    const AntonTangentialDataSRMHD &tangent,
    const bool plus_branch,
    Real R_col[NRMHD]) {
  const Real sign = plus_branch ? 1.0 : -1.0;    // ± → sign
  const Real sqrtE = std::sqrt(std::max((Real)0.0, s.rhoh + s.bsq));
  const Real E     = s.rhoh + s.bsq;              // 𝓔 = ρh + b²
  const Real W     = s.W;
  const Real u_n   = s.util_n;                    // paper u^x
  const Real u_t1  = s.util_t1;                   // paper u^y
  const Real u_t2  = s.util_t2;                   // paper u^z
  const Real b0    = s.b0;
  const Real b_n   = s.b_n;
  const Real b_t1  = s.b_t1;
  const Real b_t2  = s.b_t2;
  const Real rho   = s.rho;

  //---- V_{a,1,±} (paper Sect. 5.2, seven-vector) ------------------------------------
  Real V1[NRMHD];
  V1[ID_R]   = rho * u_t2;
  V1[ISN_R]  = 2.0 * u_t2 * (E * u_n + sign * sqrtE * b_n);
  V1[IST1_R] = E * u_t1 * u_t2 + sign * sqrtE * b_t1 * u_t2;
  V1[IST2_R] = E * (SQR(W) + SQR(u_t2) - SQR(u_n))
               + sign * sqrtE * (b_t2 * u_t2 + b0 * W - b_n * u_n);
  V1[ITAU_R] = 2.0 * u_t2 * (E * W + sign * sqrtE * b0);
  V1[IBT1_R] = b_t1 * u_t2 + sign * sqrtE * u_t1 * u_t2;
  V1[IBT2_R] = -b_t1 * u_t1 - sign * sqrtE * (1.0 + SQR(u_t1));

  //---- V_{a,2,±} = − ( ... ) --------------------------------------------------------
  Real V2[NRMHD];
  V2[ID_R]   = -(rho * u_t1);
  V2[ISN_R]  = -(2.0 * u_t1 * (E * u_n + sign * sqrtE * b_n));
  V2[IST1_R] = -(E * (SQR(W) + SQR(u_t1) - SQR(u_n))
                  + sign * sqrtE * (b_t1 * u_t1 + b0 * W - b_n * u_n));
  V2[IST2_R] = -(E * u_t1 * u_t2 + sign * sqrtE * b_t2 * u_t1);
  V2[ITAU_R] = -(2.0 * u_t1 * (E * W + sign * sqrtE * b0));
  V2[IBT1_R] = -(-b_t2 * u_t2 - sign * sqrtE * (1.0 + SQR(u_t2)));
  V2[IBT2_R] = -(b_t2 * u_t1 + sign * sqrtE * u_t1 * u_t2);

  const Real f1 = tangent.f1;
  const Real f2 = tangent.f2;
  for (int i = 0; i < NRMHD; ++i) {
    R_col[i] = f1 * V1[i] + f2 * V2[i];
  }
}

//----------------------------------------------------------------------------------------
//! Antón §5.2 — Magnetosonic right eigenvector in conserved variables.
//!
//! use_branch_A = true  → "closer-to-Alfvén" form (paper Eqs. eq:rme..eq:rmC, divide by |b_t|)
//!                       Type II regular; set 𝒞 = 0 at Type II'.
//! use_branch_A = false → "farther-from-Alfvén" form (Sect. 3.3.2 end, divide by (ρh a²−b²G))
//!                       Type II regular; set b_t^ν/(ρh a²−b²G) = 0 at Type II'.
//!
//! negative_class encodes the sign in (𝓑/a)_{m,±} — for the minus-side waves
//! (lambda[0], lambda[2]) it is true; for the plus-side waves (lambda[4], lambda[6])
//! it is false.  This matches ComputePaperBOverASRMHD().
//!
//! Auxiliary quantities per the paper:
//!   a  = W·(v_n − λ)  = util_n − λ·W
//!   𝓑 = b^n − b⁰·λ
//!   G  = 1 − λ²
//!   φ^ν = (λ, 1, 0, 0)  (indexing v-component 0=t, 1=n, 2=t1, 3=t2)
//!   u^ν = (W, u^n, u^t1, u^t2)
//!   b_t^ν, |b_t|, b_t^0  ← from AntonTangentialDataSRMHD (tangent.bt[4], .abs_bt)
//----------------------------------------------------------------------------------------
inline bool BuildDirectRightMagnetosonicSRMHD(
    const RMHDState &s,
    const AntonTangentialDataSRMHD &tangent,
    const Real lambda,
    const bool use_branch_A,
    const bool negative_class,
    Real R_col[NRMHD]) {
  //---- Wave-frame kinematic scalars per paper §3 -----------------------------------
  //   a  = W · (v_n − λ)  =  util_n − λ·W
  //   G  = 1 − λ²
  //   φ^ν = (λ, 1, 0, 0)   (index-raised, ν = 0, n, t1, t2)
  //   (𝓑/a)_{m,±} from ComputePaperBOverASRMHD (paper eq. `bas`)
  const Real a     = s.util_n - lambda * s.W;
  const Real G     = std::max((Real)1.0e-14, 1.0 - SQR(lambda));
  const Real a2    = SQR(a);
  const Real cs2   = s.cs2;
  const Real rhoh  = s.rhoh;
  const Real bsq   = s.bsq;
  const Real h     = s.h;
  const Real B_over_a = ComputePaperBOverASRMHD(s, lambda, negative_class);

  //---- Type II' denominator (paper: kills 𝒞 in Branch A, kills b_t^ν/denom_B in B)
  const Real denom_A  = a2 - (G + a2) * cs2;              // (a² − c_s²(G+a²))
  const Real denom_B  = rhoh * a2 - bsq * G;              // (ρh a² − b² G)
  const bool type_iip = (std::abs(denom_A) < 1.0e-12 * (rhoh + bsq + s.pgas));

  //---- Kinematic components (paper's u^μ, b^μ, φ^μ; μ = 0, x, y, z) ---------------
  const Real u0  = s.W;
  const Real ux  = s.util_n;   // paper u^x  (sweep-normal direction)
  const Real uy  = s.util_t1;  // paper u^y
  const Real uz  = s.util_t2;  // paper u^z
  const Real b0  = s.b0;
  const Real bx  = s.b_n;
  const Real by  = s.b_t1;
  const Real bz  = s.b_t2;
  const Real phi0 = lambda;
  const Real phix = 1.0;

  //---- Perfect-fluid (Γ-law) EOS thermodynamic derivatives ------------------------
  //   ∂ρ/∂p |_s   = ρ / (Γ · p)
  //   ∂(ρh)/∂p |_s = ρ/(Γ p) + Γ/(Γ−1)
  const Real gamma_adi  = s.gamma_adi;
  const Real drho_dp    = s.rho / (gamma_adi * s.pgas);
  const Real drhoh_dp   = drho_dp + gamma_adi / (gamma_adi - 1.0);

  //---- b_t^μ (raw and unit-direction), |b_t| — from tangent basis ------------------
  //  tangent.bt[μ]     = b_t^μ                (raw, dimensional)
  //  tangent.bt_dir[μ] = b_t^μ / |b_t|        (Antón eq. 75, well-defined at Type II)
  //  tangent.abs_bt    = |b_t|
  const Real abs_bt  = tangent.abs_bt;
  const Real *bt     = tangent.bt;             // raw b_t^μ           (Branch B uses these)
  const Real *bt_dir = tangent.bt_dir;         // b_t^μ / |b_t|       (Branch A uses these)

  //=================================================================================
  //  Branch A: paper Eqs. (932)-(936).
  //  "closer-to-Alfvén" magnetosonic; renormalized by |b_t|.
  //  In terms of the pre-renormalized covariant 𝒞:  𝒞 = -(G+a²)c_s²/denom_A · |b_t|
  //=================================================================================
  if (use_branch_A) {
    const Real one_plus_a2_G = 1.0 + a2 / G;
    const Real Ccal          = type_iip
                                 ? (Real)0.0
                                 : -(G + a2) * cs2 / denom_A * abs_bt;

    //--- D  (paper Eq. ~932) --------------------------------------------------------
    //   R_D = (a·|b_t|) / [h · denom_A] · (φ⁰ + a u⁰)
    //         − (𝓑/a)/h · b_t⁰/|b_t|
    //         − |b_t| · (G+a²) c_s² / denom_A · u⁰ · ∂_p ρ
    R_col[ID_R] =
        (type_iip ? (Real)0.0
                  : (a * abs_bt) / (h * denom_A) * (phi0 + a * u0))
        - (B_over_a / h) * bt_dir[0]
        + (type_iip ? (Real)0.0
                    : -abs_bt * (G + a2) * cs2 / denom_A * u0 * drho_dp);

    //--- S^x  (paper Eq. ~933) ------------------------------------------------------
    //   R_{S^x} = (1 + a²/G) { |b_t|/denom_A · [a(u^x φ⁰ + u⁰ φ^x)(1-c_s²) + 2 u⁰ u^x c_s² G]
    //                          − (𝓑/a)[b_t^x u⁰/|b_t| + b_t⁰ u^x/|b_t|]
    //                          + [b_t^x b⁰/|b_t| + b_t⁰ b^x/|b_t|] }
    //             + u⁰ u^x · 𝒞 · ∂_p(ρh)
    R_col[ISN_R] = one_plus_a2_G * (
          (type_iip ? (Real)0.0
                    : abs_bt / denom_A
                      * (a * (ux * phi0 + u0 * phix) * (1.0 - cs2)
                         + 2.0 * u0 * ux * cs2 * G))
        - B_over_a * (bt_dir[1] * u0 + bt_dir[0] * ux)
        + (bt_dir[1] * b0 + bt_dir[0] * bx)
      ) + u0 * ux * Ccal * drhoh_dp;

    //--- S^y  (paper Eq. ~934; j = t1) ----------------------------------------------
    //   R_{S^j} = (1 + a²/G) { u^j |b_t|/denom_A · [a φ⁰(1-c_s²) + 2 u⁰ c_s² G]
    //                          − (𝓑/a)[b_t^j u⁰/|b_t| + b_t⁰ u^j/|b_t|]
    //                          + [b_t^j b⁰/|b_t| + b_t⁰ b^j/|b_t|] }
    //             + u⁰ u^j · 𝒞 · ∂_p(ρh)
    R_col[IST1_R] = one_plus_a2_G * (
          (type_iip ? (Real)0.0
                    : uy * abs_bt / denom_A
                      * (a * phi0 * (1.0 - cs2) + 2.0 * u0 * cs2 * G))
        - B_over_a * (bt_dir[2] * u0 + bt_dir[0] * uy)
        + (bt_dir[2] * b0 + bt_dir[0] * by)
      ) + u0 * uy * Ccal * drhoh_dp;

    //--- S^z  (paper Eq. ~934; j = t2) ----------------------------------------------
    R_col[IST2_R] = one_plus_a2_G * (
          (type_iip ? (Real)0.0
                    : uz * abs_bt / denom_A
                      * (a * phi0 * (1.0 - cs2) + 2.0 * u0 * cs2 * G))
        - B_over_a * (bt_dir[3] * u0 + bt_dir[0] * uz)
        + (bt_dir[3] * b0 + bt_dir[0] * bz)
      ) + u0 * uz * Ccal * drhoh_dp;

    //--- τ  (paper Eq. ~935) --------------------------------------------------------
    //   R_τ = (1 + a²/G) { 2 u⁰|b_t|/denom_A · [a φ⁰(1-c_s²) + u⁰ c_s² G]
    //                       + |b_t|
    //                       + (b⁰ − (𝓑/a) u⁰) · 2 b_t⁰/|b_t| }
    //          + 𝒞 · (∂_p(ρh) (u⁰)² − 1)
    R_col[ITAU_R] = one_plus_a2_G * (
          (type_iip ? (Real)0.0
                    : 2.0 * u0 * abs_bt / denom_A
                      * (a * phi0 * (1.0 - cs2) + u0 * cs2 * G))
        + abs_bt
        + (b0 - B_over_a * u0) * 2.0 * bt_dir[0]
      ) + Ccal * (drhoh_dp * SQR(u0) - 1.0);

    //--- B^y  (paper Eq. ~936; j = t1) ---------------------------------------------
    //   R_{B^j} = (u^x λ_{m,±} − u⁰)/G · b_t^j/|b_t|  +  u^j · b_t⁰/|b_t|
    R_col[IBT1_R] = (ux * lambda - u0) / G * bt_dir[2] + uy * bt_dir[0];

    //--- B^z  (paper Eq. ~936; j = t2) ---------------------------------------------
    R_col[IBT2_R] = (ux * lambda - u0) / G * bt_dir[3] + uz * bt_dir[0];
  }

  //=================================================================================
  //  Branch B: paper Eqs. (~938)-(~942).
  //  "farther-from-Alfvén" magnetosonic; renormalized by (ρh a² − b² G)/G.
  //  In terms of the pre-renormalized covariant 𝒞:  𝒞 = -1.
  //  Type II' prescription: b_t^ν / (ρh a² − b² G) → 0 (kills all tangential terms).
  //=================================================================================
  else {
    //--- D  (paper Eq. ~938) --------------------------------------------------------
    //   R_D = a / [h(G+a²) c_s²] · (φ⁰ + a u⁰)
    //         − (𝓑/a) / h · b_t⁰ · G / denom_B
    //         − u⁰ · ∂_p ρ
    R_col[ID_R] =
          a / (h * (G + a2) * cs2) * (phi0 + a * u0)
        - (type_iip ? (Real)0.0
                    : (B_over_a / h) * bt[0] * G / denom_B)
        - u0 * drho_dp;

    //--- S^x  (paper Eq. ~939) ------------------------------------------------------
    //   R_{S^x} = (1/(G c_s²)) · [a(1-c_s²)(u^x φ⁰ + u⁰ φ^x) + 2 u⁰ u^x c_s² G]
    //             + (G+a²) / denom_B · [b_t^x b⁰ + b_t⁰ b^x − (b_t^x u⁰ + b_t⁰ u^x)(𝓑/a)]
    //             − u⁰ u^x · ∂_p(ρh)
    R_col[ISN_R] =
          (1.0 / (G * cs2))
              * (a * (1.0 - cs2) * (ux * phi0 + u0 * phix)
                 + 2.0 * u0 * ux * cs2 * G)
        + (type_iip ? (Real)0.0
                    : (G + a2) / denom_B
                      * (bt[1] * b0 + bt[0] * bx
                         - (bt[1] * u0 + bt[0] * ux) * B_over_a))
        - u0 * ux * drhoh_dp;

    //--- S^y  (paper Eq. ~940; j = t1) ----------------------------------------------
    R_col[IST1_R] =
          uy / (G * cs2) * (a * phi0 * (1.0 - cs2) + 2.0 * u0 * cs2 * G)
        + (type_iip ? (Real)0.0
                    : (G + a2) / denom_B
                      * (bt[2] * b0 + bt[0] * by
                         - (bt[2] * u0 + bt[0] * uy) * B_over_a))
        - u0 * uy * drhoh_dp;

    //--- S^z  (paper Eq. ~940; j = t2) ----------------------------------------------
    R_col[IST2_R] =
          uz / (G * cs2) * (a * phi0 * (1.0 - cs2) + 2.0 * u0 * cs2 * G)
        + (type_iip ? (Real)0.0
                    : (G + a2) / denom_B
                      * (bt[3] * b0 + bt[0] * bz
                         - (bt[3] * u0 + bt[0] * uz) * B_over_a))
        - u0 * uz * drhoh_dp;

    //--- τ  (paper Eq. ~941) --------------------------------------------------------
    //   R_τ = (1/(G c_s²)) · [2 u⁰(a φ⁰(1-c_s²) + u⁰ c_s² G) + a² − c_s²(G+a²)]
    //          + 2 b_t⁰ (G+a²) / denom_B · [b⁰ − (𝓑/a) u⁰]
    //          − ∂_p(ρh) · (u⁰)² + 1
    R_col[ITAU_R] =
          (1.0 / (G * cs2))
              * (2.0 * u0 * (a * phi0 * (1.0 - cs2) + u0 * cs2 * G)
                 + a2 - cs2 * (G + a2))
        + (type_iip ? (Real)0.0
                    : 2.0 * bt[0] * (G + a2) / denom_B
                      * (b0 - B_over_a * u0))
        - drhoh_dp * SQR(u0) + 1.0;

    //--- B^y  (paper Eq. ~942; j = t1) ---------------------------------------------
    //   R_{B^j} = 1/denom_B · [(u^x λ_{m,±} − u⁰) b_t^j + G u^j b_t⁰]
    R_col[IBT1_R] = type_iip
        ? (Real)0.0
        : ((ux * lambda - u0) * bt[2] + G * uy * bt[0]) / denom_B;

    //--- B^z  (paper Eq. ~942; j = t2) ---------------------------------------------
    R_col[IBT2_R] = type_iip
        ? (Real)0.0
        : ((ux * lambda - u0) * bt[3] + G * uz * bt[0]) / denom_B;
  }

  // Reject NaN / Inf.  Caller decides whether to fall through to LO / componentwise.
  for (int i = 0; i < NRMHD; ++i) {
    if (!std::isfinite(R_col[i])) return false;
  }
  return true;
}

//========================================================================================
// DIRECT §6.3 LEFT EIGENVECTORS IN CONSERVED VARIABLES
// ---------------------------------------------------------------------------------------
// Antón et al. 2010 §6.3 (paper ms.tex lines 1432–1755) gives closed-form left
// eigenvectors directly in the conserved variables U = (D, S^x, S^y, S^z, τ,
// B^y, B^z), avoiding the two-step covariant-then-chain path used in
// CharacteristicFieldsRMHD.hpp.  Each L component has the structure
//
//     L[wave][slot] = C(state, λ) · ∂Z/∂U_slot + explicit_remainder(state, λ)
//
// where Z = ρh (u^0)² and ∂Z/∂U_k is the sensitivity of Z to conserved slot k,
// supplied by ComputePaperdZdUSRMHD in the base header.  The C coefficient is
// wave-family-specific; the remainder mixes state fields, tangential-basis
// data, and (𝓑/a)_{m,±} exactly analogous to the §5.2 R builders above.
//
// Paper coord convention:  x = sweep-normal, y = t1, z = t2.
// Athena reduced 7-slot indexing:  ID_R, ISN_R, IST1_R, IST2_R, ITAU_R,
//                                  IBT1_R, IBT2_R  (n, y, z map to x, y, z).
//
// Paper §6.3 writes L directly in the τ = E − D basis (unlike §5.2 R, which
// needed the T-transform).  So NO basis correction on L rows here.
//
// After all seven rows are built, apply row-by-row normalization
// (NormalizeLeftRowsPaperSRMHD, paper Eq. 1754) so L_i · R_i = 1 exactly.
//========================================================================================

//----------------------------------------------------------------------------------------
//! Antón §6.3 Eqs. (1440)-(1459) — Entropic left eigenvector in conserved variables.
//!
//! Uses the shorthand
//!    A_e = (Z + b²) (∂s/∂p)|_ρ − ρ (1 − W² − (b⁰)²/Z) (∂s/∂ρ)|_p
//! and the Γ-law entropy derivatives (paper §6.3 preamble):
//!    (∂s/∂p)|_ρ = 1/p
//!    (∂s/∂ρ)|_p = −Γ/ρ
//!
//! Every slot has an "A_e · ∂Z/∂U_k / (Z+B²)" term plus a wave-specific remainder.
//----------------------------------------------------------------------------------------
inline void BuildDirectLeftEntropicSRMHD(const RMHDState &s,
                                          const Real dZdU[NRMHD],
                                          Real L_row[NRMHD]) {
  // Γ-law entropy derivatives.  Consistent with FillReducedEntropyLeftEigenvectorSRMHD
  // in the base header (line 1742-1743).
  const Real ds_dp   = 1.0 / s.pgas;              // (∂s/∂p)|_ρ
  const Real ds_drho = -s.gamma_adi / s.rho;      // (∂s/∂ρ)|_p

  // Auxiliary Z = ρh · (u⁰)²  (paper §6.1, base header line 1515).
  const Real Z         = s.rhoh * SQR(s.W);
  const Real B2        = SQR(s.Bn) + SQR(s.Bt1) + SQR(s.Bt2);
  const Real ZplusB2   = Z + B2;
  const Real inv_ZpB2  = 1.0 / ZplusB2;
  const Real inv_Z     = 1.0 / Z;
  const Real inv_W     = 1.0 / s.W;
  const Real W2        = SQR(s.W);

  // Shorthand A_e — appears in every slot's ∂Z coefficient (paper Eq. 1441 body).
  const Real A_e = (Z + s.bsq) * ds_dp
                 - s.rho * (1.0 - W2 - SQR(s.b0) * inv_Z) * ds_drho;

  //--- Slot D  (paper Eq. 1441) ---------------------------------------------------
  //   L_{e,D} = (1/W)(∂s/∂ρ) + A_e · ∂Z/∂D / (Z+B²)
  L_row[ID_R] = inv_W * ds_drho + inv_ZpB2 * A_e * dZdU[ID_R];

  //--- Slots S^x, S^y, S^z  (paper Eq. 1444-1447) ---------------------------------
  //   L_{e,S^i} = (1/(Z+B²)) { A_e · ∂Z/∂S^i
  //                          + (B² u^i − b⁰ B^i)/W · (∂s/∂p)
  //                          − D (u^i + b⁰ B^i / Z) · (∂s/∂ρ) }
  // Athena "n" = paper "x", "t1" = "y", "t2" = "z".
  L_row[ISN_R]  = inv_ZpB2 * (
        A_e * dZdU[ISN_R]
      + (B2 * s.util_n  - s.b0 * s.Bn ) * inv_W * ds_dp
      - s.D * (s.util_n  + s.b0 * s.Bn  * inv_Z) * ds_drho );

  L_row[IST1_R] = inv_ZpB2 * (
        A_e * dZdU[IST1_R]
      + (B2 * s.util_t1 - s.b0 * s.Bt1) * inv_W * ds_dp
      - s.D * (s.util_t1 + s.b0 * s.Bt1 * inv_Z) * ds_drho );

  L_row[IST2_R] = inv_ZpB2 * (
        A_e * dZdU[IST2_R]
      + (B2 * s.util_t2 - s.b0 * s.Bt2) * inv_W * ds_dp
      - s.D * (s.util_t2 + s.b0 * s.Bt2 * inv_Z) * ds_drho );

  //--- Slot τ  (paper Eq. 1453) ---------------------------------------------------
  //   L_{e,τ} = −(∂s/∂p) + A_e · ∂Z/∂τ / (Z+B²)
  L_row[ITAU_R] = -ds_dp + inv_ZpB2 * A_e * dZdU[ITAU_R];

  //--- Slots B^y, B^z  (paper Eq. 1457-1459) --------------------------------------
  //   L_{e,B^i} = (1/(Z+B²)) { A_e · ∂Z/∂B^i
  //                          − ρ (2 B^i (1 − W²) + b⁰ (W S^i / Z + u^i)) · (∂s/∂ρ)
  //                          + ((B² b^i − b⁰ S^i)/W + (2 − 1/W²) Z B^i) · (∂s/∂p) }
  const Real one_minus_W2   = 1.0 - W2;
  const Real two_minus_iW2  = 2.0 - 1.0 / W2;

  L_row[IBT1_R] = inv_ZpB2 * (
        A_e * dZdU[IBT1_R]
      - s.rho * (2.0 * s.Bt1 * one_minus_W2
                 + s.b0 * (s.W * s.S_t1 * inv_Z + s.util_t1)) * ds_drho
      + ((B2 * s.b_t1 - s.b0 * s.S_t1) * inv_W
         + two_minus_iW2 * Z * s.Bt1) * ds_dp );

  L_row[IBT2_R] = inv_ZpB2 * (
        A_e * dZdU[IBT2_R]
      - s.rho * (2.0 * s.Bt2 * one_minus_W2
                 + s.b0 * (s.W * s.S_t2 * inv_Z + s.util_t2)) * ds_drho
      + ((B2 * s.b_t2 - s.b0 * s.S_t2) * inv_W
         + two_minus_iW2 * Z * s.Bt2) * ds_dp );
}

//----------------------------------------------------------------------------------------
//! Antón §6.3 Eqs. (1470)-(1598) — Alfvén left eigenvector in conserved variables.
//!
//! L_{a,±} = f₁ · W_{a,1,±}  +  f₂ · W_{a,2,±}                       (Eq. 1472)
//!
//! Each W_{a,k,±} is a 7-vector with the shape
//!   W_{a,k,±,U_slot} = C_{a,k,±} · ∂Z/∂U_slot + explicit_remainder(state, λ, ±)
//!
//! Sign convention: plus_branch=false → "−" (Alfvén⁻ at λ[1]); true → "+" at λ[5].
//! Type II (|g₁|² + |g₂|² → 0): f₁ = f₂ = 1/√2 (enforced by
//! GetAntonTangentialDataStrictSRMHD upstream).
//!
//! Paper coord convention: x = sweep-normal, y = t1, z = t2.  Athena reduced
//! slots: ID_R (D), ISN_R (S^x), IST1_R (S^y), IST2_R (S^z), ITAU_R (τ),
//! IBT1_R (B^y), IBT2_R (B^z).
//----------------------------------------------------------------------------------------
inline bool BuildDirectLeftAlfvenSRMHD(const RMHDState &s,
                                        const AntonTangentialDataSRMHD &tangent,
                                        const Real lambda,
                                        const bool plus_branch,
                                        const Real dZdU[NRMHD],
                                        Real L_row[NRMHD]) {
  //---- Common precomputed scalars -------------------------------------------------
  const Real sign  = plus_branch ? 1.0 : -1.0;
  const Real E     = s.rhoh + s.bsq;                     // 𝓔 = ρh + b²
  const Real sqrtE = std::sqrt(std::max((Real)0.0, E));

  // Athena → paper index rename for readability.
  const Real u0 = s.W;                                    // u^0
  const Real ux = s.util_n;                               // u^x
  const Real uy = s.util_t1;                              // u^y
  const Real uz = s.util_t2;                              // u^z
  const Real b0 = s.b0;                                   // b^0
  const Real bx = s.b_n;                                  // b^x
  const Real by = s.b_t1;                                 // b^y
  const Real bz = s.b_t2;                                 // b^z
  const Real Bx = s.Bn;                                   // B^x  (constrained-transport normal-B)
  const Real By = s.Bt1;                                  // B^y
  const Real Bz = s.Bt2;                                  // B^z
  const Real Sx = s.S_n;                                  // S^x
  const Real Sy = s.S_t1;                                 // S^y
  const Real Sz = s.S_t2;                                 // S^z

  const Real B2 = SQR(Bx) + SQR(By) + SQR(Bz);            // paper's 𝐁²
  const Real Z  = s.rhoh * SQR(u0);                       // Z = ρh(u⁰)² (paper §6.1)
  const Real inv_Z  = 1.0 / std::max((Real)1.0e-30, Z);
  const Real inv_u0 = 1.0 / u0;

  const Real a    = ux - lambda * u0;                     // paper a
  const Real Bcal = bx - lambda * b0;                     // paper 𝓑
  const Real u0sq = SQR(u0);
  const Real one_minus_u0sq = 1.0 - u0sq;

  // Paper shorthand appearing in several rows.
  const Real E_u0_plus_sign_sqrtE_b0 = E * u0 + sign * sqrtE * b0;
  const Real E_u0_minus_b0sq_over_u0 = E * u0 - SQR(b0) * inv_u0;

  // Cross-products that appear repeatedly.
  const Real cross_yzy = By * uz - Bz * uy;               // B^y u^z − B^z u^y
  const Real cross_yzy_b = by * uz - bz * uy;             // b^y u^z − b^z u^y
  // Paper cross-product identities (multiplication commutes):
  //   (B^y u^z − B^z u^y)  =  (u^z B^y − u^y B^z)  =  +cross_yzy
  //   (B^z u^y − B^y u^z)  =  (u^y B^z − u^z B^y)  =  −cross_yzy
  //   Same pattern for b^μ → cross_yzy_b.

  //=================================================================================
  //  C_{a,1,±} coefficient  (paper Eq. 1584-1590)
  //=================================================================================
  //  C_{a,1,±} = −𝓔·u⁰·u^z·(1 − a·u^x)
  //             − (𝓔·u⁰ ± √𝓔·b⁰) · b⁰/Z · ( b^z(1 + a·u^x) + u^y(b^z u^y − b^y u^z) − 2 b^x a u^z )
  //             + (Z + b²) · u^z
  //             − b^y (b^z u^y − b^y u^z) · ((u⁰)² − 1)
  //             ± √𝓔 · { u^z · ((u⁰)² u^x 𝓑 + b⁰ b^x B^x / Z )
  //                       − ((u⁰)² − 1) · ( b^z + u^y(b^z u^y − b^y u^z) ) }
  //
  //  Note: paper's (b^z u^y − b^y u^z) = −cross_yzy_b.  Every occurrence of
  //  this cross-product below appears with an explicit MINUS sign to match
  //  the paper's convention.
  const Real Ca1 =
      -E * u0 * uz * (1.0 - a * ux)
      - E_u0_plus_sign_sqrtE_b0 * b0 * inv_Z
          * ( bz * (1.0 + a * ux)
              - uy * cross_yzy_b                              // + u^y·(b^z u^y − b^y u^z) = − u^y·cross_yzy_b
              - 2.0 * bx * a * uz )
      + (Z + s.bsq) * uz
      + by * cross_yzy_b * (u0sq - 1.0)                       // − b^y·(b^z u^y − b^y u^z)·… = + b^y·cross_yzy_b·…
      + sign * sqrtE
          * ( uz * ( u0sq * ux * Bcal + b0 * bx * Bx * inv_Z )
              - (u0sq - 1.0) * ( bz - uy * cross_yzy_b ) );   // b^z + u^y·(b^z u^y − b^y u^z) = b^z − u^y·cross_yzy_b

  //=================================================================================
  //  C_{a,2,±} coefficient  (paper Eq. 1592-1598) — y↔z swap of C_{a,1,±}
  //  Paper's (b^y u^z − b^z u^y) = +cross_yzy_b (positive).
  //=================================================================================
  const Real Ca2 =
      -E * u0 * uy * (1.0 - a * ux)
      - E_u0_plus_sign_sqrtE_b0 * b0 * inv_Z
          * ( by * (1.0 + a * ux)
              + uz * cross_yzy_b                          // + u^z·(b^y u^z − b^z u^y) = + u^z·cross_yzy_b
              - 2.0 * bx * a * uy )
      + (Z + s.bsq) * uy
      - bz * cross_yzy_b * (u0sq - 1.0)                   // − b^z·(b^y u^z − b^z u^y)·… = − b^z·cross_yzy_b·…
      + sign * sqrtE
          * ( uy * ( u0sq * ux * Bcal + b0 * bx * Bx * inv_Z )
              - (u0sq - 1.0) * ( by + uz * cross_yzy_b ) );   // b^y + u^z·(b^y u^z − b^z u^y) = b^y + u^z·cross_yzy_b

  //=================================================================================
  //  W_{a,1,±} components (paper Eqs. 1479-1524)
  //=================================================================================
  Real W1[NRMHD];

  // Eq. 1479:  W_{a,1,±,D} = C · ∂Z/∂D
  W1[ID_R] = Ca1 * dZdU[ID_R];

  // Eq. 1483-1486:  W_{a,1,±,S^x}
  //   C·∂Z/∂S^x + (B²u^x − b⁰B^x)·u^z/u⁰ − u^x b^y (B^y u^z − B^z u^y)
  //   + (𝓔u⁰ ± √𝓔 b⁰) · u^z (u^x − 2a)
  //   ± √𝓔 · ( u^x B^z − u^z B^x − u^y u^x (u^z B^y − u^y B^z) )
  W1[ISN_R] =
        Ca1 * dZdU[ISN_R]
      + (B2 * ux - b0 * Bx) * uz * inv_u0
      - ux * by * cross_yzy
      + E_u0_plus_sign_sqrtE_b0 * uz * (ux - 2.0 * a)
      + sign * sqrtE * ( ux * Bz - uz * Bx - uy * ux * cross_yzy );   // (u^z B^y − u^y B^z) = +cross_yzy

  // Eq. 1489-1492:  W_{a,1,±,S^y}
  //   C·∂Z/∂S^y + (B²u^y − b⁰B^y)·u^z/u⁰ + b⁰ b^y u^z − u^y b^y (B^y u^z − B^z u^y)
  //   ± √𝓔 · ( b^y u^z u⁰ + (1 + (u^y)²)(u^y B^z − u^z B^y) )
  W1[IST1_R] =
        Ca1 * dZdU[IST1_R]
      + (B2 * uy - b0 * By) * uz * inv_u0
      + b0 * by * uz
      - uy * by * cross_yzy
      + sign * sqrtE * ( by * uz * u0 + (1.0 + SQR(uy)) * (-cross_yzy) );

  // Eq. 1495-1498:  W_{a,1,±,S^z}
  //   C·∂Z/∂S^z + (B²u^z − b⁰B^z)·u^z/u⁰ − b⁰ b^y u^y − u^z b^y (B^y u^z − B^z u^y)
  //   + 𝓔 (u⁰)² (u⁰ − λ u^x)
  //   ± √𝓔 · ( b⁰ u^x a + b^z u^z u⁰ − u^z u^y (u^z B^y − u^y B^z) )
  W1[IST2_R] =
        Ca1 * dZdU[IST2_R]
      + (B2 * uz - b0 * Bz) * uz * inv_u0
      - b0 * by * uy
      - uz * by * cross_yzy
      + E * u0sq * (u0 - lambda * ux)
      + sign * sqrtE * ( b0 * ux * a + bz * uz * u0 - uz * uy * cross_yzy );   // (u^z B^y − u^y B^z) = +cross_yzy

  // Eq. 1501:  W_{a,1,±,τ} = C·∂Z/∂τ − u^z (Z + B²)
  W1[ITAU_R] = Ca1 * dZdU[ITAU_R] - uz * (Z + B2);

  // Eq. 1505-1514:  W_{a,1,±,B^y}
  {
    const Real bracket1 = (B2 * by - b0 * Sy) * inv_u0
                        + (2.0 - 1.0 / u0sq) * Z * By;    // {…}·u^z prefactor
    const Real cross_z_paper = -cross_yzy;                 // (B^z u^y − B^y u^z) = −cross_yzy
    const Real bracket2 = uy * b0 * inv_u0
                        + 2.0 * By * one_minus_u0sq * inv_u0;
    const Real bracket3 =
          By * uz * inv_u0
        + (1.0 - a * ux) * (b0 * uy - 2.0 * u0sq * By) * uz * inv_u0
        - Sy * u0 * inv_Z * bx * uz * a;
    const Real bracket4 =
          Sy * uz * (1.0 + bx * Bx * u0 * inv_Z)
        - E_u0_minus_b0sq_over_u0 * uy * uz
        + (b0 * uy - 2.0 * u0sq * By) * uz * ux * Bx * inv_u0
        - (Bz + (-cross_yzy) * uy)
            * (uy * b0 * inv_u0 + 2.0 * By * one_minus_u0sq * inv_u0);

    W1[IBT1_R] =
          Ca1 * dZdU[IBT1_R]
        + bracket1 * uz
        + by * cross_z_paper * bracket2
        + E_u0_plus_sign_sqrtE_b0 * bracket3
        - sign * sqrtE * bracket4;
  }

  // Eq. 1517-1524:  W_{a,1,±,B^z}
  {
    const Real bracket1 = (B2 * bz - b0 * Sz) * inv_u0
                        + (2.0 - 1.0 / u0sq) * Z * Bz;
    const Real cross_z_paper = -cross_yzy;                 // (B^z u^y − B^y u^z)
    const Real bracket2 = uz * b0 * inv_u0
                        + 2.0 * Bz * one_minus_u0sq * inv_u0;
    const Real bracket3 =
          Bz * uz * inv_u0
        + (1.0 - a * ux) * (b0 * uz - 2.0 * u0sq * Bz) * uz * inv_u0
        - Sz * u0 * inv_Z * bx * uz * a;
    const Real bracket4 =
          Sz * uz * (1.0 + bx * Bx * u0 * inv_Z)
        + E_u0_minus_b0sq_over_u0 * ( u0sq - lambda * u0 * ux - SQR(uz) )
        + (b0 * uz - 2.0 * u0sq * Bz) * uz * ux * Bx * inv_u0
        - (Bz + (-cross_yzy) * uy)
            * (uz * b0 * inv_u0 + 2.0 * Bz * one_minus_u0sq * inv_u0);

    W1[IBT2_R] =
          Ca1 * dZdU[IBT2_R]
        + bracket1 * uz
        + by * cross_z_paper * bracket2
        + E_u0_plus_sign_sqrtE_b0 * bracket3
        - sign * sqrtE * bracket4;
  }

  //=================================================================================
  //  W_{a,2,±} components (paper Eqs. 1527-1578) — y↔z swap of W_{a,1,±}
  //=================================================================================
  Real W2[NRMHD];

  // Eq. 1527:  W_{a,2,±,D} = C · ∂Z/∂D
  W2[ID_R] = Ca2 * dZdU[ID_R];

  // Eq. 1531-1533:  W_{a,2,±,S^x} — y↔z swap of Eq. 1483-1486
  //   Paper's (B^z u^y − B^y u^z) = −cross_yzy;  (u^y B^z − u^z B^y) = −cross_yzy.
  W2[ISN_R] =
        Ca2 * dZdU[ISN_R]
      + (B2 * ux - b0 * Bx) * uy * inv_u0
      - ux * bz * (-cross_yzy)                            // − u^x b^z (B^z u^y − B^y u^z) = − u^x b^z·(−cross_yzy) = +u^x b^z·cross_yzy
      + E_u0_plus_sign_sqrtE_b0 * uy * (ux - 2.0 * a)
      + sign * sqrtE * ( ux * By - uy * Bx - uz * ux * (-cross_yzy) );  // − u^z u^x (u^y B^z − u^z B^y) = − u^z u^x·(−cross_yzy)

  // Eq. 1537-1540:  W_{a,2,±,S^y} — y↔z swap of Eq. 1495-1498
  W2[IST1_R] =
        Ca2 * dZdU[IST1_R]
      + (B2 * uy - b0 * By) * uy * inv_u0
      - b0 * bz * uz
      - uy * bz * (-cross_yzy)
      + E * u0sq * (u0 - lambda * ux)
      + sign * sqrtE * ( b0 * ux * a + by * uy * u0 - uz * uy * (-cross_yzy) );  // − u^z u^y (u^y B^z − u^z B^y) = − u^z u^y·(−cross_yzy)

  // Eq. 1544-1547:  W_{a,2,±,S^z} — y↔z swap of Eq. 1489-1492
  W2[IST2_R] =
        Ca2 * dZdU[IST2_R]
      + (B2 * uz - b0 * Bz) * uy * inv_u0
      + b0 * bz * uy
      - uz * bz * (-cross_yzy)
      + sign * sqrtE * ( bz * uy * u0 + (1.0 + SQR(uz)) * cross_yzy );

  // Eq. 1550:  W_{a,2,±,τ} = C·∂Z/∂τ − u^y (Z + B²)
  W2[ITAU_R] = Ca2 * dZdU[ITAU_R] - uy * (Z + B2);

  // Eq. 1554-1567:  W_{a,2,±,B^y} — y↔z swap of Eq. 1517-1524 (S^y↔S^z, y↔z, B^y↔B^z)
  {
    const Real bracket1 = (B2 * by - b0 * Sy) * inv_u0
                        + (2.0 - 1.0 / u0sq) * Z * By;
    const Real cross_y_paper = cross_yzy;                 // (B^y u^z − B^z u^y)
    const Real bracket2 = uy * b0 * inv_u0
                        + 2.0 * By * one_minus_u0sq * inv_u0;
    const Real bracket3 =
          By * uy * inv_u0
        + (1.0 - a * ux) * (b0 * uy - 2.0 * u0sq * By) * uy * inv_u0
        - Sy * u0 * inv_Z * bx * uy * a;
    const Real bracket4 =
          Sy * uy * (1.0 + bx * Bx * u0 * inv_Z)
        + E_u0_minus_b0sq_over_u0 * ( u0sq - lambda * u0 * ux - SQR(uy) )
        + (b0 * uy - 2.0 * u0sq * By) * uy * ux * Bx * inv_u0
        - (By + cross_yzy * uz)
            * (uy * b0 * inv_u0 + 2.0 * By * one_minus_u0sq * inv_u0);

    W2[IBT1_R] =
          Ca2 * dZdU[IBT1_R]
        + bracket1 * uy
        + bz * cross_y_paper * bracket2
        + E_u0_plus_sign_sqrtE_b0 * bracket3
        - sign * sqrtE * bracket4;
  }

  // Eq. 1569-1578:  W_{a,2,±,B^z} — y↔z swap of Eq. 1505-1514
  {
    const Real bracket1 = (B2 * bz - b0 * Sz) * inv_u0
                        + (2.0 - 1.0 / u0sq) * Z * Bz;
    const Real cross_y_paper = cross_yzy;
    const Real bracket2 = uz * b0 * inv_u0
                        + 2.0 * Bz * one_minus_u0sq * inv_u0;
    const Real bracket3 =
          Bz * uy * inv_u0
        + (1.0 - a * ux) * (b0 * uz - 2.0 * u0sq * Bz) * uy * inv_u0
        - Sz * u0 * inv_Z * bx * uy * a;
    const Real bracket4 =
          Sz * uy * (1.0 + bx * Bx * u0 * inv_Z)
        - E_u0_minus_b0sq_over_u0 * uy * uz
        + (b0 * uz - 2.0 * u0sq * Bz) * uy * ux * Bx * inv_u0
        - (By + cross_yzy * uz)
            * (uz * b0 * inv_u0 + 2.0 * Bz * one_minus_u0sq * inv_u0);

    W2[IBT2_R] =
          Ca2 * dZdU[IBT2_R]
        + bracket1 * uy
        + bz * cross_y_paper * bracket2
        + E_u0_plus_sign_sqrtE_b0 * bracket3
        - sign * sqrtE * bracket4;
  }

  //=================================================================================
  //  L_{a,±} = f₁ · W_{a,1,±}  −  f₂ · W_{a,2,±}
  //
  //  Paper-consistency derivation:
  //    Paper reduced-cov Eq. 1270 explicitly writes `\bar w_{a,2,±} = −(...)`
  //    with an overall minus.  Paper R (conserved) applies the SAME convention:
  //    V_{a,2,±} = −(...) (paper Eq. 900-916), and code carries this negation
  //    in BuildDirectRightAlfvenSRMHD line 175-182.  Under this convention R
  //    passes every test.
  //
  //    Paper §6.3 W_{a,2,±} (Eqs. 1527-1598) drops the negation in the
  //    conserved-variable formulas — paper's Ca2 (Eq. 1592) has same-sign
  //    structure as Ca1[y↔z], not negated form.  Paper's Eq. 1472
  //    `L = f₁W₁ + f₂W₂` is therefore INTERNALLY INCONSISTENT with paper's
  //    own reduced-cov Eq. 1270 and with the R-side V_{a,2,±} convention.
  //
  //    To restore paper's overall convention (matching Eq. 1270 and R's
  //    V_{a,2,±}=−(...)), we apply the negation here.  Empirically this
  //    fixes L·R at S2, S9, S11, S24 to machine precision without biorth.
  //=================================================================================
  const Real f1 = tangent.f1;
  const Real f2 = tangent.f2;
  for (int k = 0; k < NRMHD; ++k) {
    L_row[k] = f1 * W1[k] - f2 * W2[k];
    if (!std::isfinite(L_row[k])) return false;
  }
  return true;
}

//----------------------------------------------------------------------------------------
//! Antón §6.3 Eqs. (1607)-(1750) — Magnetosonic left eigenvector in conserved variables.
//!
//! Paper defines the "raw" magnetosonic L (Eqs. 1624-1706) as
//!   L_{m,±,U_slot} = C_{m,±} · ∂Z/∂U_slot + [b_t² / (a² − (G+a²) c_s²)] · {...}
//!                    + b^0_t · {...} + g₁ · {...} + g₂ · {...}
//!
//! Then the Type II renormalization (Eqs. 1710-1750) selects one of two
//! substitution schemes based on which pair of magnetosonic waves we're on:
//!
//!   Branch A (closer to Alfvén — use_branch_A=true, "eq. eff1"):
//!     b_t²/(a²−(G+a²)c_s²) → |b_t|/(a²−(G+a²)c_s²)
//!     b^0_t → b^0_t / |b_t|
//!     g_i   → g_i / |b_t|
//!
//!   Branch B (farther from Alfvén — use_branch_A=false, "eq. eff2"):
//!     b_t²/(a²−(G+a²)c_s²) → 1
//!     b^0_t → b^0_t · G(G+a²)c_s² / (ρh a² − b² G)
//!     g_i   → g_i · G(G+a²)c_s² / (ρh a² − b² G)
//!
//! The code path below applies the appropriate substitutions in-line.
//! Type II' guards (denom_A ≈ 0 for Branch A, denom_B ≈ 0 for Branch B) zero
//! out the offending prefactors as in §5.2 R.
//----------------------------------------------------------------------------------------
inline bool BuildDirectLeftMagnetosonicSRMHD(
    const RMHDState &s,
    const AntonTangentialDataSRMHD &tangent,
    const Real lambda,
    const bool use_branch_A,
    const bool negative_class,
    const Real dZdU[NRMHD],
    Real L_row[NRMHD]) {
  //---- Precomputed scalars --------------------------------------------------------
  const Real u0 = s.W;
  const Real ux = s.util_n;
  const Real uy = s.util_t1;
  const Real uz = s.util_t2;
  const Real b0 = s.b0;
  const Real bx = s.b_n;
  const Real by = s.b_t1;
  const Real bz = s.b_t2;
  const Real Bx = s.Bn;
  const Real By = s.Bt1;
  const Real Bz = s.Bt2;
  const Real Sx = s.S_n;
  const Real Sy = s.S_t1;
  const Real Sz = s.S_t2;
  const Real rhoh = s.rhoh;
  const Real cs2  = s.cs2;

  const Real B2   = SQR(Bx) + SQR(By) + SQR(Bz);
  const Real Z    = rhoh * SQR(u0);
  const Real inv_Z = 1.0 / std::max((Real)1.0e-30, Z);
  const Real inv_rhoh = 1.0 / std::max((Real)1.0e-30, rhoh);
  const Real inv_u0 = 1.0 / u0;
  const Real u0sq = SQR(u0);

  const Real a  = ux - lambda * u0;
  const Real G  = std::max((Real)1.0e-14, 1.0 - SQR(lambda));
  const Real a2 = SQR(a);
  const Real Bcal = bx - lambda * b0;
  const Real B_over_a = ComputePaperBOverASRMHD(s, lambda, negative_class);
  const Real one_minus_u0sq = 1.0 - u0sq;

  const Real denom_A = a2 - (G + a2) * cs2;              // Branch A key denominator
  const Real denom_B = rhoh * a2 - s.bsq * G;            // Branch B key denominator
  const bool typeIIp_A = (std::abs(denom_A) < 1.0e-12 * (rhoh + s.bsq + s.pgas));
  const bool typeIIp_B = (std::abs(denom_B) < 1.0e-12 * (rhoh + s.bsq + s.pgas));

  const Real abs_bt = tangent.abs_bt;
  const Real *bt    = tangent.bt;                        // raw b_t^μ
  const Real g1     = tangent.g1;
  const Real g2     = tangent.g2;

  // Renormalized coefficient prefactors — Type II substitutions (Eqs. 1710-1747)
  //   coeff_bt2      = b_t²/(a²−(G+a²)c_s²) after substitution
  //   coeff_bt0      = b^0_t after substitution
  //   g1_over_bt,
  //   g2_over_bt     = g_{1,2} after substitution
  //                    (multiplied by g_i to preserve Type-II limit via
  //                     tangent.ghat_i which use paper Eq. 154 when |b_t| → 0)
  //
  //   At Type II (|b_t| → 0) in Branch A, paper's substitution `b^0_t/|b_t|`
  //   requires a finite limit (via f₁ = f₂ = 1/√2 and Antón Eq. 75), which is
  //   supplied by `tangent.bt_dir[0]`.  Similarly `g_i/|b_t|` uses paper
  //   Eq. 154 limit supplied by `tangent.ghat_{1,2}`.  Using raw
  //   `bt[0]/|b_t|` and `g_i/|b_t|` (which evaluate to 0/0 at Type II) would
  //   zero out entire L rows at strict Type II — a systematic bug.
  Real coeff_bt2, coeff_bt0, g1_over_bt, g2_over_bt;
  if (use_branch_A) {
    coeff_bt2  = typeIIp_A ? (Real)0.0 : (abs_bt / denom_A);
    coeff_bt0  = tangent.bt_dir[0];                       // b^0_t/|b_t| with Type-II limit
    g1_over_bt = tangent.ghat1;                           // g₁/|b_t| with Type-II limit
    g2_over_bt = tangent.ghat2;                           // g₂/|b_t| with Type-II limit
  } else {
    const Real G_Gpa2_cs2 = G * (G + a2) * cs2;
    coeff_bt2  = 1.0;
    coeff_bt0  = typeIIp_B ? (Real)0.0 : (bt[0] * G_Gpa2_cs2 / denom_B);
    g1_over_bt = typeIIp_B ? (Real)0.0 : (g1 * G_Gpa2_cs2 / denom_B);
    g2_over_bt = typeIIp_B ? (Real)0.0 : (g2 * G_Gpa2_cs2 / denom_B);
  }

  //=================================================================================
  //  Auxiliary quantity H_{m,±}  (paper Eq. 1608):
  //    H = (G + 2a²)·(b⁰ − (𝓑/a)·u⁰) − (b^x·λ − b⁰)
  //=================================================================================
  const Real H = (G + 2.0 * a2) * (b0 - B_over_a * u0) - (bx * lambda - b0);

  //=================================================================================
  //  C_{m,±}  (paper Eq. 1611)
  //    C = coeff_bt2 · { (c_s²−1)(G+a²)·a·u⁰·(u^x + b⁰B^x/Z)
  //                     + ((u⁰)² + b²/(ρh))·(λ·u^x − u⁰)·G }
  //        − coeff_bt0 · { 2(G+a²)·B^x·u⁰/(u⁰−λu^x) · (u⁰a + λ + b⁰𝓑/Z)
  //                       + ((u⁰)² + b²/(ρh))·G·(𝓑/a) }
  //        − g₂·coeff_g · { u⁰·H·(u⁰u^y + b⁰b^y/Z)
  //                         + (b^y((u⁰)² − 1) − u⁰u^y b⁰(2Z+b²)/Z)·(u⁰ − λu^x) }
  //        − g₁·coeff_g · { u⁰·H·(u⁰u^z + b⁰b^z/Z)
  //                         + (b^z((u⁰)² − 1) − u⁰u^z b⁰(2Z+b²)/Z)·(u⁰ − λu^x) }
  //=================================================================================
  const Real den_vel = u0 - lambda * ux;                 // u⁰ − λu^x
  if (std::abs(den_vel) < 1.0e-14) return false;
  const Real inv_den_vel = 1.0 / den_vel;
  const Real E_hydro_kappa = u0sq + s.bsq * inv_rhoh;    // (u⁰)² + b²/(ρh)
  const Real two_Z_plus_b2 = 2.0 * Z + s.bsq;

  const Real C_first =
        (cs2 - 1.0) * (G + a2) * a * u0 * (ux + b0 * Bx * inv_Z)
      + E_hydro_kappa * (lambda * ux - u0) * G;

  const Real C_second =
        2.0 * (G + a2) * Bx * u0 * inv_den_vel
            * (u0 * a + lambda + b0 * Bcal * inv_Z)
      + E_hydro_kappa * G * B_over_a;

  const Real C_third =
        u0 * H * (u0 * uy + b0 * by * inv_Z)
      + (by * (u0sq - 1.0) - u0 * uy * b0 * two_Z_plus_b2 * inv_Z) * den_vel;

  const Real C_fourth =
        u0 * H * (u0 * uz + b0 * bz * inv_Z)
      + (bz * (u0sq - 1.0) - u0 * uz * b0 * two_Z_plus_b2 * inv_Z) * den_vel;

  // NOTE: paper Eq. 1611 (Cm) writes "- g_2 {u^y-terms} - g_1 {u^z-terms}", which
  // is INCONSISTENT with the g_1↔y, g_2↔z convention used in every other §6.3
  // slot equation (Eqs. 1635, 1640, 1651, 1670, 1689) and with base header's
  // FillReducedMagnetosonicLeftEigenvectorSRMHD (which has row[1]/uy paired
  // with g1, row[2]/uz paired with g2).  Paper Cm has a typo.  We use the
  // consistent convention here (g1 with u^y-terms, g2 with u^z-terms).
  const Real Cm = coeff_bt2 * C_first
                - coeff_bt0 * C_second
                - g1_over_bt * C_third
                - g2_over_bt * C_fourth;

  //=================================================================================
  //  L_{m,±} components (paper Eqs. 1624-1706)
  //=================================================================================
  const Real two_minus_iu0sq = 2.0 - 1.0 / u0sq;

  //--- Slot D  (Eq. 1624):  L_{m,±,D} = C_{m,±} · ∂Z/∂D
  L_row[ID_R] = Cm * dZdU[ID_R];

  //--- Slot S^x  (Eq. 1628-1638)
  //   L_{m,±,S^x} = C·∂Z/∂S^x
  //     + coeff_bt2·{ (1−c_s²) a (G+a²)·((u⁰)² + (B^x)²/(ρh))
  //                   + (B² u^x − b⁰ B^x)/(ρh u⁰)·(λu^x − u⁰) G }
  //     + coeff_bt0·{ 2(G+a²)B^x/den_vel · ((1+au^x)u⁰ + B^x 𝓑/(ρh))
  //                   − (B² u^x − b⁰ B^x)/(ρh u⁰)·(𝓑/a) G }
  //     + g₁·coeff_g·{ u⁰·H·(u^x u^y + B^x b^y/(ρh u⁰))
  //                    − den_vel·(u^y (b^x u⁰ + B^x b²/(ρh)) − B^y u^x) }
  //     + g₂·coeff_g·{ u⁰·H·(u^x u^z + B^x b^z/(ρh u⁰))
  //                    − den_vel·(u^z (b^x u⁰ + B^x b²/(ρh)) − B^z u^x) }
  {
    const Real term1 =
          (1.0 - cs2) * a * (G + a2) * (u0sq + SQR(Bx) * inv_rhoh)
        + (B2 * ux - b0 * Bx) / (rhoh * u0) * (lambda * ux - u0) * G;
    const Real term2 =
          2.0 * (G + a2) * Bx * inv_den_vel
              * ((1.0 + a * ux) * u0 + Bx * Bcal * inv_rhoh)
        - (B2 * ux - b0 * Bx) / (rhoh * u0) * B_over_a * G;
    const Real term3 =
          u0 * H * (ux * uy + Bx * by / (rhoh * u0))
        - den_vel * (uy * (bx * u0 + Bx * s.bsq * inv_rhoh) - By * ux);
    const Real term4 =
          u0 * H * (ux * uz + Bx * bz / (rhoh * u0))
        - den_vel * (uz * (bx * u0 + Bx * s.bsq * inv_rhoh) - Bz * ux);

    L_row[ISN_R] = Cm * dZdU[ISN_R]
                 + coeff_bt2 * term1
                 + coeff_bt0 * term2
                 + g1_over_bt * term3
                 + g2_over_bt * term4;
  }

  //--- Slot S^y  (Eq. 1640-1649)
  {
    const Real term1 =
          (1.0 - cs2) * a * (G + a2) * Bx * By * inv_rhoh
        + (B2 * uy - b0 * By) / (rhoh * u0) * (lambda * ux - u0) * G;
    const Real term2 =
          2.0 * (G + a2) * Bx * inv_den_vel
              * (a * uy * u0 + By * Bcal * inv_rhoh)
        - (B2 * uy - b0 * By) / (rhoh * u0) * B_over_a * G;
    const Real term3 =
          u0 * H * (1.0 + SQR(uy) + By * by / (rhoh * u0))
        - den_vel * (uy * By * s.bsq * inv_rhoh + b0 * (1.0 + SQR(uy)));
    const Real term4 =
          u0 * H * (uy * uz + By * bz / (rhoh * u0))
        - den_vel * (uz * (by * u0 + By * s.bsq * inv_rhoh) - Bz * uy);

    L_row[IST1_R] = Cm * dZdU[IST1_R]
                  + coeff_bt2 * term1
                  + coeff_bt0 * term2
                  + g1_over_bt * term3
                  + g2_over_bt * term4;
  }

  //--- Slot S^z  (Eq. 1651-1660)
  {
    const Real term1 =
          (1.0 - cs2) * a * (G + a2) * Bx * Bz * inv_rhoh
        + (B2 * uz - b0 * Bz) / (rhoh * u0) * (lambda * ux - u0) * G;
    const Real term2 =
          2.0 * (G + a2) * Bx * inv_den_vel
              * (a * uz * u0 + Bz * Bcal * inv_rhoh)
        - (B2 * uz - b0 * Bz) / (rhoh * u0) * B_over_a * G;
    const Real term3 =
          u0 * H * (uz * uy + Bz * by / (rhoh * u0))
        - den_vel * (uy * (bz * u0 + Bz * s.bsq * inv_rhoh) - By * uz);
    const Real term4 =
          u0 * H * (1.0 + SQR(uz) + Bz * bz / (rhoh * u0))
        - den_vel * (uz * Bz * s.bsq * inv_rhoh + b0 * (1.0 + SQR(uz)));

    L_row[IST2_R] = Cm * dZdU[IST2_R]
                  + coeff_bt2 * term1
                  + coeff_bt0 * term2
                  + g1_over_bt * term3
                  + g2_over_bt * term4;
  }

  //--- Slot τ  (Eq. 1662-1666)
  //   L_{m,±,τ} = C·∂Z/∂τ
  //     − ((u⁰)² + B²/(ρh)) · G · { coeff_bt2·(λu^x − u⁰) − coeff_bt0 · (𝓑/a) }
  {
    const Real E_kappa_B = u0sq + B2 * inv_rhoh;
    L_row[ITAU_R] = Cm * dZdU[ITAU_R]
                  - E_kappa_B * G * ( coeff_bt2 * (lambda * ux - u0)
                                     - coeff_bt0 * B_over_a );
  }

  //--- Slot B^y  (Eq. 1670-1687)
  {
    const Real term1 =
          (1.0 - cs2) * a * (G + a2) * (Sy * Bx * inv_rhoh - 2.0 * By * ux * u0)
        + ( (-b0 * Sy + B2 * by) / (rhoh * u0)
            + (2.0 * u0sq - 1.0) * By ) * G * (lambda * ux - u0);
    const Real term2 =
          2.0 * Bx * (G + a2) * inv_den_vel
              * ( b0 * a * uy + Sy * Bcal * inv_rhoh
                  - 2.0 * u0 * By * (lambda + u0 * a) )
        + ( (b0 * Sy - B2 * by) / (rhoh * u0) - (2.0 * u0sq - 1.0) * By )
              * B_over_a * G;
    const Real term3 =
          H * ( b0 * (1.0 + SQR(uy)) + Sy * by * inv_rhoh
                - 2.0 * u0sq * By * uy )
        - den_vel * ( s.Emhd * u0                               // 𝓔·u⁰
                      + Sy * uy * (1.0 + s.bsq * inv_rhoh)
                      + uy * b0 * inv_u0 * (b0 * uy - By) )
        - 2.0 * By * (u0 * uy * b0 + one_minus_u0sq * inv_u0 * By);
    const Real term4 =
          H * ( (b0 * uy - 2.0 * u0sq * By) * uz + Sy * bz * inv_rhoh )
        - den_vel * ( Sy * uz * (1.0 + s.bsq * inv_rhoh)
                      + uy * b0 * inv_u0 * (b0 * uz - Bz) )
        - 2.0 * By * (u0 * uz * b0 + one_minus_u0sq * inv_u0 * Bz);

    L_row[IBT1_R] = Cm * dZdU[IBT1_R]
                  + coeff_bt2 * term1
                  + coeff_bt0 * term2
                  + g1_over_bt * term3
                  + g2_over_bt * term4;
  }

  //--- Slot B^z  (Eq. 1689-1706)
  {
    const Real term1 =
          (1.0 - cs2) * a * (G + a2) * (Sz * Bx * inv_rhoh - 2.0 * Bz * ux * u0)
        + ( (-b0 * Sz + B2 * bz) / (rhoh * u0)
            + (2.0 * u0sq - 1.0) * Bz ) * G * (lambda * ux - u0);
    const Real term2 =
          2.0 * Bx * (G + a2) * inv_den_vel
              * ( b0 * a * uz + Sz * Bcal * inv_rhoh
                  - 2.0 * u0 * Bz * (lambda + u0 * a) )
        + ( (b0 * Sz - B2 * bz) / (rhoh * u0) - (2.0 * u0sq - 1.0) * Bz )
              * B_over_a * G;
    const Real term3 =
          H * ( (b0 * uz - 2.0 * u0sq * Bz) * uy + Sz * by * inv_rhoh )
        - den_vel * ( Sz * uy * (1.0 + s.bsq * inv_rhoh)
                      + uz * b0 * inv_u0 * (b0 * uy - By) )
        - 2.0 * Bz * (u0 * uy * b0 + one_minus_u0sq * inv_u0 * By);
    const Real term4 =
          H * ( b0 * (1.0 + SQR(uz)) + Sz * bz * inv_rhoh
                - 2.0 * u0sq * Bz * uz )
        - den_vel * ( s.Emhd * u0
                      + Sz * uz * (1.0 + s.bsq * inv_rhoh)
                      + uz * b0 * inv_u0 * (b0 * uz - Bz) )
        - 2.0 * Bz * (u0 * uz * b0 + one_minus_u0sq * inv_u0 * Bz);

    L_row[IBT2_R] = Cm * dZdU[IBT2_R]
                  + coeff_bt2 * term1
                  + coeff_bt0 * term2
                  + g1_over_bt * term3
                  + g2_over_bt * term4;
  }

  // Reject non-finite entries.
  for (int k = 0; k < NRMHD; ++k) {
    if (!std::isfinite(L_row[k])) return false;
  }
  (void)Sx;  // Sx unused (S^x appears in paper only via ∂Z/∂S^x already in dZdU)
  return true;
}

//----------------------------------------------------------------------------------------
//! Top-level entry point for direct §6.3 L construction (all seven rows).
//! To be called from GetEigenVectorDirectSRMHD after R is built.
//!
//! Returns false if any per-row build fails (tangential-basis solve failure,
//! non-finite entries) OR row normalization can't proceed (L_i · R_i ≈ 0).
//! Alfvén and magnetosonic paths are stubs at this stage — they fall back to
//! L = R⁻¹ via Gauss-Jordan through the caller.  The entropic row (index 3)
//! is fully implemented and can be validated independently.
//----------------------------------------------------------------------------------------
inline bool GetDirectLeftEigenVectorSRMHD(const RMHDState &avg,
                                           const Real lambda[NRMHD],
                                           const Real (&R)[NRMHD][NRMHD],
                                           Real (&L)[NRMHD][NRMHD]) {
  //---- 1. Compute ∂Z/∂U^k once  ---------------------------------------------------
  Real dZdU[NRMHD] = {};
  if (!ComputePaperdZdUSRMHD(avg, dZdU)) return false;

  //---- 2. Branch classification (same as R) --------------------------------------
  bool use_branch_A[NRMHD] = {};
  GetRightMagnetosonicRenormBranchSRMHD(lambda, use_branch_A);

  //---- 3. Row 3: entropic (paper §6.3 Eqs. 1440-1460) ----------------------------
  BuildDirectLeftEntropicSRMHD(avg, dZdU, L[3]);

  //---- 4. Rows 1, 5: Alfvén ± (paper §6.3 Eqs. 1470-1598) ------------------------
  //  Same paper-sign-vs-slot-label subtlety as R side: paper's ± in Eq. (alfven)
  //  is a formula-sign choice, not a slot label.  At backward flow the max/min
  //  ordering of (λ_a,+, λ_a,-) SWAPS, so slot 5 (max) may correspond to paper
  //  "−".  Mirror the R-side Type-I-aware detection here so L rows are built
  //  with the same paper sign convention as their corresponding R columns.
  {
    const Real sqrt_E_local =
        std::sqrt(std::max((Real)0.0, avg.rhoh + avg.bsq));
    const Real d_p = avg.b0 + sqrt_E_local * avg.W;
    const Real d_m = avg.b0 - sqrt_E_local * avg.W;
    Real lambda_a_paper_plus, lambda_a_paper_minus;
    if (std::abs(d_p) > 1.0e-12) {
      lambda_a_paper_plus = (avg.b_n + sqrt_E_local * avg.util_n) / d_p;
    } else {
      lambda_a_paper_plus = avg.v_n;
    }
    if (std::abs(d_m) > 1.0e-12) {
      lambda_a_paper_minus = (avg.b_n - sqrt_E_local * avg.util_n) / d_m;
    } else {
      lambda_a_paper_minus = avg.v_n;
    }
    const bool type_I_active =
        (std::abs(lambda[1] - avg.v_n) < 1.0e-12)
     && (std::abs(lambda[5] - avg.v_n) < 1.0e-12);
    bool slot1_is_plus, slot5_is_plus;
    if (type_I_active) {
      slot1_is_plus = false;   // paper §3.3.1 default at Type I
      slot5_is_plus = true;
    } else {
      slot1_is_plus =
          (std::abs(lambda[1] - lambda_a_paper_plus)
           < std::abs(lambda[1] - lambda_a_paper_minus));
      slot5_is_plus =
          (std::abs(lambda[5] - lambda_a_paper_plus)
           < std::abs(lambda[5] - lambda_a_paper_minus));
    }

    AntonTangentialDataSRMHD tangent_alf_m;
    if (!GetAntonTangentialDataStrictSRMHD(avg, lambda[1], tangent_alf_m)) return false;
    if (!BuildDirectLeftAlfvenSRMHD(avg, tangent_alf_m, lambda[1],
                                     /*plus_branch=*/slot1_is_plus, dZdU, L[1])) return false;

    AntonTangentialDataSRMHD tangent_alf_p;
    if (!GetAntonTangentialDataStrictSRMHD(avg, lambda[5], tangent_alf_p)) return false;
    if (!BuildDirectLeftAlfvenSRMHD(avg, tangent_alf_p, lambda[5],
                                     /*plus_branch=*/slot5_is_plus, dZdU, L[5])) return false;
  }

  //---- 5. Rows 0, 2, 4, 6: magnetosonic (paper §6.3 Eqs. 1607-1750) --------------
  {
    const int rows[4]           = {0, 2, 4, 6};
    const bool neg_class[4]     = {true, true, false, false};
    for (int k = 0; k < 4; ++k) {
      const int r        = rows[k];
      const Real lam_wav = lambda[r];
      const bool branchA = use_branch_A[r];
      const bool nclass  = neg_class[k];

      AntonTangentialDataSRMHD tangent;
      if (!GetAntonTangentialDataStrictSRMHD(avg, lam_wav, tangent)) return false;
      if (!BuildDirectLeftMagnetosonicSRMHD(avg, tangent, lam_wav,
                                              branchA, nclass, dZdU, L[r])) return false;
    }
  }

  //---- 6. Apply T⁻¹ basis correction to every row --------------------------------
  //  Paper §6.3 writes L in Antón basis; T-transform on R put R in Athena
  //  τ = E − D basis; T⁻¹ on L rows brings L to the same basis so L·R = δ:
  //     L_athena[i][D] = L_anton[i][D] + L_anton[i][τ]
  for (int i = 0; i < NRMHD; ++i) {
    L[i][ID_R] += L[i][ITAU_R];
  }

  //---- 7. Reject non-finite entries ----------------------------------------------
  for (int i = 0; i < NRMHD; ++i) {
    for (int j = 0; j < NRMHD; ++j) {
      if (!std::isfinite(L[i][j])) return false;
    }
  }

  //---- 8. Biorthogonality correction: L ← (L·R)⁻¹ · L ---------------------------
  //  Paper §6.3 Eq. 1754: "L must be multiplied by normalization factors in order
  //  to fulfill the condition L_i · R_j = δ_{ij}."  The covariant path in the base
  //  header interprets this as a full matrix correction — because paper's raw L
  //  is not exactly R⁻¹, only close (biorthogonality residual up to ~1e-3 at
  //  moderate states in our tests; more at Type II / degeneracy).
  //
  //  Numerical benefit: P = L·R is close to identity regardless of cond(R), so
  //  inverting P is stable.  Then L_new = P⁻¹ · L is R⁻¹ but with off-diagonal
  //  precision much better than Gauss-Jordan directly on R at extreme states.
  //
  //  Fall back to plain row normalization if correction fails (P singular).
  if (!CorrectBiorthogonalitySRMHD(R, L)) {
    if (!NormalizeLeftRowsPaperSRMHD(L, R)) return false;
  }

  return true;
}

//----------------------------------------------------------------------------------------
//! Shared R build for both DIRECT_INVERSE and DIRECT_CONSERVED modes.
//! Builds R via Antón §5.2 direct conserved-variable formulas, applies the
//! T-transform to convert Antón E-basis → Athena τ = E − D basis, and
//! column-scales R (each column max-abs = 1) to condition it for downstream
//! Gauss-Jordan inversion or biorthogonality correction.
//!
//! Returns false on:
//!   • Non-finite R entries after §5.2 build
//!   • Tangential-basis solve failure at any Alfvén / magnetosonic wave
//!   • Magnetosonic column build failure
//!   • Zero or non-finite column max-abs (fully rank-deficient column)
//----------------------------------------------------------------------------------------
inline bool BuildDirectRightMatrixSRMHD(const RMHDState &avg,
                                         const Real lambda[NRMHD],
                                         Real (&R)[NRMHD][NRMHD],
                                         bool apply_column_scale = true) {
  // Zero-init R.
  for (int i = 0; i < NRMHD; ++i) {
    for (int j = 0; j < NRMHD; ++j) R[i][j] = 0.0;
  }

  //---- 1. Tangential basis + f₁, f₂ + |b_t| ---------------------------------------
  bool use_branch_A[NRMHD] = {};
  GetRightMagnetosonicRenormBranchSRMHD(lambda, use_branch_A);

  //---- 2. Column 3: entropic  -----------------------------------------------------
  {
    Real col[NRMHD];
    BuildDirectRightEntropicSRMHD(avg, col);
    for (int i = 0; i < NRMHD; ++i) R[i][3] = col[i];
  }

  //---- 3. Columns 1, 5: Alfvén ± ---------------------------------------------------
  //
  // The Alfvén eigenvector formulas (paper Eq. alfven.cons) are parametrized
  // by paper's ± convention, which chooses the sign in
  //     λ_a,± = (b^x ± √𝓔·u^x) / (b^0 ± √𝓔·W)
  // GetEigenValuesSRMHD assigns λ[1] = min(λ_a,+, λ_a,-) and
  // λ[5] = max(λ_a,+, λ_a,-).  At forward flow (u^x > 0 with b0 > 0),
  // λ_a,+ > λ_a,- so slot 5 = paper "+" and slot 1 = paper "−".  But at
  // backward flow (u^x < 0 with corresponding sign changes in b_n, b0),
  // the ordering FLIPS: slot 5 becomes paper "−" and slot 1 becomes paper "+".
  // If we hard-code plus_branch based on slot index we build the wrong
  // Alfvén eigenvector — Rayleigh quotient at S21 (S11 mirror) showed
  // Δλ = 1.54 on Alf+ column, physical residual 0.48.
  //
  // Fix: derive plus_branch by comparing λ[1], λ[5] against paper-formula
  // (λ_a,+, λ_a,−).  Purely a labelling correction — no formula change.
  {
    const Real sqrt_E_local =
        std::sqrt(std::max((Real)0.0, avg.rhoh + avg.bsq));
    const Real d_p = avg.b0 + sqrt_E_local * avg.W;
    const Real d_m = avg.b0 - sqrt_E_local * avg.W;
    Real lambda_a_paper_plus, lambda_a_paper_minus;
    if (std::abs(d_p) > 1.0e-12) {
      lambda_a_paper_plus = (avg.b_n + sqrt_E_local * avg.util_n) / d_p;
    } else {
      lambda_a_paper_plus = avg.v_n;
    }
    if (std::abs(d_m) > 1.0e-12) {
      lambda_a_paper_minus = (avg.b_n - sqrt_E_local * avg.util_n) / d_m;
    } else {
      lambda_a_paper_minus = avg.v_n;
    }
    // At Type I (paper §3.2, characterised by 𝓑=0 for the Alfvén wave; in the
    // lab frame this reduces to |Bn| → 0), all five middle waves collapse to
    // v_n and GetEigenValuesSRMHD detects this by forcing λ[1..5] = v_n.
    // Paper §3.3.1 shows the R_{a,+} and R_{a,-} formulas remain well-defined
    // and INDEPENDENT at Type I: the ± there is a formula-sign label for
    // eigenvector independence, not a physical eigenvalue label.  At Type I
    // the paper's default is slot 1 → R_{a,−}, slot 5 → R_{a,+} (arbitrary
    // but independent).
    //
    // Away from Type I, pick the paper sign (± in Eq. alfven) whose formula
    // best matches each slot's eigenvalue.  Slot assignment (min→1, max→5)
    // does NOT determine paper sign: at backward flow the paper formulas'
    // min/max swap, so slot 5 may correspond to paper "−".
    bool slot1_is_plus, slot5_is_plus;
    const bool type_I_active =
        (std::abs(lambda[1] - avg.v_n) < 1.0e-12)
     && (std::abs(lambda[5] - avg.v_n) < 1.0e-12);
    if (type_I_active) {
      slot1_is_plus = false;   // paper default: independent R_{a,−}, R_{a,+}
      slot5_is_plus = true;
    } else {
      slot1_is_plus =
          (std::abs(lambda[1] - lambda_a_paper_plus)
           < std::abs(lambda[1] - lambda_a_paper_minus));
      slot5_is_plus =
          (std::abs(lambda[5] - lambda_a_paper_plus)
           < std::abs(lambda[5] - lambda_a_paper_minus));
    }

    Real col[NRMHD];
    AntonTangentialDataSRMHD tangent_alf_m;
    if (!GetAntonTangentialDataStrictSRMHD(avg, lambda[1], tangent_alf_m)) return false;
    BuildDirectRightAlfvenSRMHD(avg, tangent_alf_m,
                                /*plus_branch=*/slot1_is_plus, col);
    for (int i = 0; i < NRMHD; ++i) R[i][1] = col[i];

    AntonTangentialDataSRMHD tangent_alf_p;
    if (!GetAntonTangentialDataStrictSRMHD(avg, lambda[5], tangent_alf_p)) return false;
    BuildDirectRightAlfvenSRMHD(avg, tangent_alf_p,
                                /*plus_branch=*/slot5_is_plus, col);
    for (int i = 0; i < NRMHD; ++i) R[i][5] = col[i];
  }

  //---- 4. Columns 0, 2, 4, 6: magnetosonic (fast∓, slow∓) --------------------------
  //  Wave index → (lambda index, negative_class, use_branch_A[k])
  //   k=0 → lambda[0] (fast⁻),  negative_class=true,  use_branch_A[0]
  //   k=1 → lambda[2] (slow⁻),  negative_class=true,  use_branch_A[2]
  //   k=2 → lambda[4] (slow⁺),  negative_class=false, use_branch_A[4]
  //   k=3 → lambda[6] (fast⁺),  negative_class=false, use_branch_A[6]
  {
    const int cols[4]           = {0, 2, 4, 6};
    const bool neg_class[4]     = {true, true, false, false};
    for (int k = 0; k < 4; ++k) {
      const int c        = cols[k];
      const Real lam_wav = lambda[c];
      const bool branchA = use_branch_A[c];
      const bool nclass  = neg_class[k];

      AntonTangentialDataSRMHD tangent;
      if (!GetAntonTangentialDataStrictSRMHD(avg, lam_wav, tangent)) return false;

      Real col[NRMHD];
      if (!BuildDirectRightMagnetosonicSRMHD(avg, tangent, lam_wav,
                                              branchA, nclass, col)) {
        return false;
      }
      for (int i = 0; i < NRMHD; ++i) R[i][c] = col[i];
    }
  }

  //---- 5. Reject non-finite R -----------------------------------------------------
  for (int i = 0; i < NRMHD; ++i) {
    for (int j = 0; j < NRMHD; ++j) {
      if (!std::isfinite(R[i][j])) return false;
    }
  }

  //---- 5b. E → τ = E − D basis transform (Antón writes in E-basis; Athena stores
  //  τ = E − D in u[ITAU_R]).  Apply T = I − e_τ · e_D^T to each column of R:
  //      R_athena[τ, k] = R_anton[E, k] − R_anton[D, k]
  //  This does not modify Antón's formulas — it is the change-of-basis matrix T
  //  applied AFTER paper's R is built, matching the E↔τ shuttle already performed
  //  at PackReducedState (line 655) and WriteReducedFlux (line 2233).  Without
  //  this, R columns span "Antón basis" wave directions, not Athena τ-basis
  //  eigenvectors — per-wave LF splitting in the characteristic reconstruction
  //  then uses λ_max[m] on a mislabelled projection, injecting spurious entropy
  //  into the τ slot over multiple cycles (visible in smooth rarefactions).
  for (int k = 0; k < NRMHD; ++k) {
    R[ITAU_R][k] -= R[ID_R][k];
  }

  //---- 5c. Column scaling — condition R for Gauss-Jordan inversion ---------------
  //  Antón §5.2 magnetosonic columns naturally scale as 1/(G·c_s²) (Branch B) and
  //  1/denom_A (Branch A), which explode in cold or near-degenerate plasma.  In
  //  practice R columns span ~6 orders of magnitude in a single matrix (e.g. cold
  //  rotor rarefaction: entropic col ~1e+0 while slow-magnetosonic cols ~1e+6).
  //  Gauss-Jordan on such R loses precision in the off-diagonal of L·R, which
  //  manifests as bin-4 saturation of the EFL_DEBUG lr_off histogram and drives
  //  spurious eig-decomp rejections in cold plasma (rotor tier2 fraction climbing
  //  to ~22% by cycle 80).
  //
  //  This step scales each column by 1/max|R[·,k]|, bringing every column to
  //  max-abs = 1.  Matches ScaleRightEigenvectorsSRMHD in the covariant path
  //  (base header line 2065).  Purely numerical — the reconstruction chain
  //  R·L·f = f is invariant under column scaling (Gauss-Jordan produces
  //  a correspondingly scaled L, and R·L is bit-identical whether pre-scaled
  //  or not).  Expected effect on cold-plasma states: cond(R) drops by 4-6
  //  orders, |L·R − I|_off migrates from bin 4 → bins 0–2, and cold-plasma
  //  eig-decomp rejections drop from ~22% to a few percent.
  if (apply_column_scale) {
    for (int c = 0; c < NRMHD; ++c) {
      Real scale = 0.0;
      for (int r = 0; r < NRMHD; ++r) {
        const Real absval = std::abs(R[r][c]);
        if (absval > scale) scale = absval;
      }
      if (!(scale > 0.0) || !std::isfinite(scale)) return false;
      const Real inv_scale = 1.0 / scale;
      for (int r = 0; r < NRMHD; ++r) R[r][c] *= inv_scale;
    }
  } else {
    // Diagnostic path — skip column scaling.  Still reject a fully-null column.
    for (int c = 0; c < NRMHD; ++c) {
      Real scale = 0.0;
      for (int r = 0; r < NRMHD; ++r) {
        const Real absval = std::abs(R[r][c]);
        if (absval > scale) scale = absval;
      }
      if (!(scale > 0.0) || !std::isfinite(scale)) return false;
    }
  }

  return true;
}

//----------------------------------------------------------------------------------------
//! DIRECT_INVERSE mode entry point.  Fast production path:
//!    R via §5.2 direct conserved-variable formulas + T-transform + column scaling
//!    L via numerical inverse (long-double Gauss-Jordan on the conditioned R)
//!
//! Cost: ~1500 ops/face.  Delivers machine-precision L·R = I at all 11 tested
//! states thanks to column-scaling.  Recommended default for production runs
//! using the direct route.
//----------------------------------------------------------------------------------------
inline bool GetEigenVectorDirectInverseSRMHD(const RMHDState &avg,
                                              const Real lambda[NRMHD],
                                              Real (&L)[NRMHD][NRMHD],
                                              Real (&R)[NRMHD][NRMHD]) {
  // Zero-init L (R is zeroed inside the helper).
  for (int i = 0; i < NRMHD; ++i) {
    for (int j = 0; j < NRMHD; ++j) L[i][j] = 0.0;
  }
  if (!BuildDirectRightMatrixSRMHD(avg, lambda, R)) return false;
  if (!InvertMatrixRMHD(R, L)) return false;
  return true;
}

//----------------------------------------------------------------------------------------
//! DIRECT_CONSERVED mode entry point.  Paper-native path:
//!    R via §5.2 direct conserved-variable formulas + T-transform + column scaling
//!    L via §6.3 direct conserved-variable formulas + T⁻¹ + biorthogonality
//!      correction (paper Eq. 1754)
//!
//! Cost: ~5900 ops/face (~4× slower than DIRECT_INVERSE).  Matches Antón §6.3
//! formula-by-formula — useful for paper-reproducibility verification.
//! Falls back to L = R⁻¹ Gauss-Jordan if the §6.3 build fails (extreme
//! degeneracy, non-finite intermediates).
//----------------------------------------------------------------------------------------
inline bool GetEigenVectorDirectSRMHD(const RMHDState &avg,
                                       const Real lambda[NRMHD],
                                       Real (&L)[NRMHD][NRMHD],
                                       Real (&R)[NRMHD][NRMHD]) {
  for (int i = 0; i < NRMHD; ++i) {
    for (int j = 0; j < NRMHD; ++j) L[i][j] = 0.0;
  }
  if (!BuildDirectRightMatrixSRMHD(avg, lambda, R)) return false;

  //---- L via §6.3 with biorthogonality correction, Gauss-Jordan fallback --------
  if (!GetDirectLeftEigenVectorSRMHD(avg, lambda, R, L)) {
    if (!InvertMatrixRMHD(R, L)) return false;
  }
  return true;
}

}  // namespace direct
}  // namespace characterisiticfields::rmhd

#endif  // HYDRO_CHARACTERISTIC_FIELDS_RMHD_DIRECT_HPP_
