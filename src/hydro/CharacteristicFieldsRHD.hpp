#ifndef HYDRO_CHARACTERISTIC_FIELDS_RHD_HPP_
#define HYDRO_CHARACTERISTIC_FIELDS_RHD_HPP_
//========================================================================================
// Athena++ astrophysical MHD code -- EFL extension
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file CharacteristicFieldsRHD.hpp
//! \brief High-order characteristic decomposition for special-relativistic
//!        hydrodynamics (SRHD), 5x5 Donat 1998 / Font (Living Rev. 2008) eigensystem.
//!
//! The 5-vector basis used internally is the permuted shifted-energy form
//!   q = (D, S_n, S_t1, S_t2, tau),    tau = E - D,
//! where (n, t1, t2) is the (normal, tangential1, tangential2) frame keyed off
//! ivx. Conversion to/from Athena's (D, S^1, S^2, S^3, E) cons/flux layout is
//! handled at pack/unpack boundaries (PackQRHD / WriteFluxRHD).
//!
//! Reference:
//!   Font (2008) Living Rev. Relativ. 11 (Section 9.3), citing
//!   Donat, Font, Ibanez, Marquina (1998) JCP 146, 58.

#include <algorithm>
#include <cmath>
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../mesh/mesh.hpp"
#include "../utils/utils.hpp"

namespace {

// Reuse the WENO5/WENO5Z/CS5 scalar reconstruction kernel and the recon-kind
// enum defined alongside the SRMHD characteristic header so that the same
// HO_RECON_KIND argument controls both the SRMHD and SRHD HO paths.  We do NOT
// re-include the SRMHD header here -- the kernel below is duplicated to keep
// this header self-contained when SRMHD is not compiled in.
enum HOReconKindRHD {
  HO_RECON_WENO5_RHD  = 0,
  HO_RECON_WENO5Z_RHD = 1,
  HO_RECON_CS5_RHD    = 2
};

inline Real ReconstructScalarHORHD(const Real char_flx[5], const int rec_kind) {
  constexpr Real optimw[3] = {1.0/10.0, 3.0/5.0, 3.0/10.0};
  constexpr Real epsl = 1.0e-42;  // small denominator floor (matches SRMHD)
  constexpr Real othreeotwo = 13.0/12.0;

  const Real fipt = char_flx[4];
  const Real fipo = char_flx[3];
  const Real fi   = char_flx[2];
  const Real fimo = char_flx[1];
  const Real fimt = char_flx[0];

  if (rec_kind == HO_RECON_CS5_RHD) {
    return (2.0 * fimt - 13.0 * fimo + 47.0 * fi + 27.0 * fipo - 3.0 * fipt) / 60.0;
  }

  Real a[3], b[3], w[3], fk[3];
  b[0] = othreeotwo * SQR(fimt - 2.0 * fimo + fi)
       + 0.25 * SQR(fimt - 4.0 * fimo + 3.0 * fi);
  b[1] = othreeotwo * SQR(fimo - 2.0 * fi + fipo)
       + 0.25 * SQR(fimo - fipo);
  b[2] = othreeotwo * SQR(fi - 2.0 * fipo + fipt)
       + 0.25 * SQR(3.0 * fi - 4.0 * fipo + fipt);

  if (rec_kind == HO_RECON_WENO5Z_RHD) {
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

// Number of waves in the SRHD system: (entropy, two transverse, two acoustic).
constexpr int NRHD = 5;

// Indices into the permuted (D, S_n, S_t1, S_t2, tau) state/flux 5-vector.
enum RHDIndex {
  ID_RHD   = 0,
  ISN_RHD  = 1,
  IST1_RHD = 2,
  IST2_RHD = 3,
  ITAU_RHD = 4
};

// (n, t1, t2) -> Athena (1, 2, 3) component indices keyed off ivx.
struct RHDDirMap {
  int in;   // IM index of the normal momentum component (IM1, IM2 or IM3)
  int it1;  // IM index of the first tangential momentum component
  int it2;  // IM index of the second tangential momentum component
  int iv_n;  // IVX/IVY/IVZ of normal velocity in primitive array
  int iv_t1; // IVX/IVY/IVZ of first tangential velocity
  int iv_t2; // IVX/IVY/IVZ of second tangential velocity
};

inline RHDDirMap GetRHDDirMap(const int ivx) {
  // ivx is one of IVX (=1), IVY (=2), IVZ (=3) per athena.hpp.
  // For each direction (n, t1, t2) is a cyclic permutation of (1, 2, 3).
  RHDDirMap map{};
  switch (ivx) {
    case IVX:
      map.in   = IM1; map.it1  = IM2; map.it2  = IM3;
      map.iv_n = IVX; map.iv_t1= IVY; map.iv_t2= IVZ;
      break;
    case IVY:
      map.in   = IM2; map.it1  = IM3; map.it2  = IM1;
      map.iv_n = IVY; map.iv_t1= IVZ; map.iv_t2= IVX;
      break;
    default:
      map.in   = IM3; map.it1  = IM1; map.it2  = IM2;
      map.iv_n = IVZ; map.iv_t1= IVX; map.iv_t2= IVY;
      break;
  }
  return map;
}

// Compact state struct for finalized primitives + thermodynamics.  All
// quantities are stored in the (n, t1, t2) frame defined by the current ivx.
struct RHDState {
  Real rho  = 0.0;   // rest-mass density
  Real pgas = 0.0;   // gas pressure
  Real vn   = 0.0;   // normal 3-velocity
  Real vt1  = 0.0;   // tangential1 3-velocity
  Real vt2  = 0.0;   // tangential2 3-velocity
  Real W    = 1.0;   // Lorentz factor 1/sqrt(1 - v^2)
  Real h    = 1.0;   // specific enthalpy h = 1 + gamma/(gamma-1) p/rho
  Real cs2  = 0.0;   // sound speed squared cs^2 = gamma p / (rho h)
  Real K    = 1.0;   // K = h (ideal gas / gamma-law); kept as a separate field
                     //   so the eigenvector formulas stay literal Donat 1998.
  Real v2   = 0.0;   // v^2 = vn^2 + vt1^2 + vt2^2 (cached)
};

// ----------------------------------------------------------------------------
// Pack the per-cell conservative state into the permuted shifted-energy
// 5-vector q = (D, S_n, S_t1, S_t2, tau = E - D).  Reads directly from Athena's
// `cons` array indexed (component, k, j, i).  No reduction is performed.
inline void PackQRHD(const AthenaArray<Real> &cons, const RHDDirMap &map,
                     const int k, const int j, const int i, Real q[NRHD]) {
  q[ID_RHD]   = cons(IDN, k, j, i);
  q[ISN_RHD]  = cons(map.in,  k, j, i);
  q[IST1_RHD] = cons(map.it1, k, j, i);
  q[IST2_RHD] = cons(map.it2, k, j, i);
  q[ITAU_RHD] = cons(IEN, k, j, i) - cons(IDN, k, j, i);  // tau = E - D
}

// Inverse of PackQRHD: write a flux 5-vector in the (D, S_n, S_t1, S_t2, tau)
// basis back into Athena's lab-frame flux array, restoring E from tau:
//   F_E = F_tau + F_D    (since tau = E - D and the operator is linear).
inline void WriteFluxRHD(const Real flx_tau[NRHD], const RHDDirMap &map,
                         const int k, const int j, const int i,
                         AthenaArray<Real> &flux) {
  flux(IDN,    k, j, i) = flx_tau[ID_RHD];
  flux(map.in, k, j, i) = flx_tau[ISN_RHD];
  flux(map.it1,k, j, i) = flx_tau[IST1_RHD];
  flux(map.it2,k, j, i) = flx_tau[IST2_RHD];
  flux(IEN,    k, j, i) = flx_tau[ITAU_RHD] + flx_tau[ID_RHD];  // E = tau + D
}

// ----------------------------------------------------------------------------
// FinalizeStateRHD: from primitive (rho, pgas, v^i) compute Lorentz factor,
// enthalpy, sound speed and K.  Athena's primitive `vel` slots store the spatial
// 4-velocity u^i = W v^i; we rebuild W from W = sqrt(1 + |u|^2).  This matches
// the convention used everywhere else in the SR hydro module (see
// adiabatic_hydro_sr.cpp).
inline void FinalizeStateRHD(RHDState &s, const Real gamma_adi) {
  // primitive `vel` slots already hold u^i (spatial part of 4-velocity)
  // when we copy them from prim(IVX/Y/Z, ...).  Caller supplies u^n, u^t1, u^t2
  // through s.vn, s.vt1, s.vt2 in that convention.
  const Real un = s.vn, ut1 = s.vt1, ut2 = s.vt2;
  const Real usq = un*un + ut1*ut1 + ut2*ut2;
  s.W = std::sqrt(1.0 + usq);
  const Real invW = 1.0 / s.W;
  s.vn  = un  * invW;
  s.vt1 = ut1 * invW;
  s.vt2 = ut2 * invW;
  s.v2  = s.vn*s.vn + s.vt1*s.vt1 + s.vt2*s.vt2;

  // gamma-law EOS:  h = 1 + gamma/(gamma-1) * p / rho;  K = h;
  // sound speed^2 (relativistic ideal gas): cs2 = gamma * p / (rho * h)
  const Real eps = s.pgas / (std::max(s.rho, (Real)TINY_NUMBER) * (gamma_adi - 1.0));
  s.h   = 1.0 + gamma_adi * eps;
  s.K   = s.h;
  s.cs2 = gamma_adi * s.pgas / (std::max(s.rho, (Real)TINY_NUMBER) * s.h);
}

// Build a finalized state from a single cell of the primitive array, with the
// (n, t1, t2) frame keyed off ivx.  The primitive `vel` slots are u^i = W v^i.
inline void GetStateCellRHD(const AthenaArray<Real> &prim, const RHDDirMap &map,
                            const int k, const int j, const int i,
                            const Real gamma_adi, RHDState &s) {
  s.rho  = prim(IDN, k, j, i);
  s.pgas = prim(IPR, k, j, i);
  s.vn   = prim(map.iv_n,  k, j, i);
  s.vt1  = prim(map.iv_t1, k, j, i);
  s.vt2  = prim(map.iv_t2, k, j, i);
  FinalizeStateRHD(s, gamma_adi);
}

// Simple arithmetic average of L and R primitive states at a face, then
// finalize.  This is the SR hydro analog of GetStateAvgSRMHD's primitive-mean
// strategy.  The averaging in u^i (rather than v^i) is used to avoid the
// nonlinear v->u inversion at the face.
inline void GetStateAvgRHD(const AthenaArray<Real> &prim, const RHDDirMap &map,
                           const int k, const int j, const int i, const int ivx,
                           const Real gamma_adi, RHDState &s) {
  // L is the cell on the negative-normal side of face i.
  int kL = k, jL = j, iL = i;
  switch (ivx) {
    case IVX: iL = i - 1; break;
    case IVY: jL = j - 1; break;
    default:  kL = k - 1; break;
  }
  // Average 3-velocities (not 4-velocities).  Athena stores u^i = W*v^i in the
  // primitive velocity slots, so convert to v^i first, average, then rebuild W.
  // This matches the working implementation in the reference repo and gives a
  // physically bounded average (|v_avg| < 1) even across strong shocks.
  const Real un_L  = prim(map.iv_n,  kL, jL, iL);
  const Real ut1_L = prim(map.iv_t1, kL, jL, iL);
  const Real ut2_L = prim(map.iv_t2, kL, jL, iL);
  const Real WL = std::sqrt(1.0 + un_L*un_L + ut1_L*ut1_L + ut2_L*ut2_L);
  const Real vn_L  = un_L  / WL;
  const Real vt1_L = ut1_L / WL;
  const Real vt2_L = ut2_L / WL;

  const Real un_R  = prim(map.iv_n,  k, j, i);
  const Real ut1_R = prim(map.iv_t1, k, j, i);
  const Real ut2_R = prim(map.iv_t2, k, j, i);
  const Real WR = std::sqrt(1.0 + un_R*un_R + ut1_R*ut1_R + ut2_R*ut2_R);
  const Real vn_R  = un_R  / WR;
  const Real vt1_R = ut1_R / WR;
  const Real vt2_R = ut2_R / WR;

  const Real rhoL = prim(IDN, kL, jL, iL);
  const Real rhoR = prim(IDN, k, j, i);
  const Real pL   = prim(IPR, kL, jL, iL);
  const Real pR   = prim(IPR, k, j, i);
  const Real gmo  = gamma_adi - 1.0;

  s.rho = 0.5 * (rhoL + rhoR);
  s.vn  = 0.5 * (vn_L + vn_R);
  s.vt1 = 0.5 * (vt1_L + vt1_R);
  s.vt2 = 0.5 * (vt2_L + vt2_R);
  s.v2  = s.vn*s.vn + s.vt1*s.vt1 + s.vt2*s.vt2;
  s.W   = 1.0 / std::sqrt(1.0 - s.v2);

  // Average specific internal energy ε (not p separately), then reconstruct
  // thermodynamics.  This gives a more physical enthalpy across strong
  // discontinuities than averaging p and ρ independently.
  const Real eps_L = pL / (gmo * std::max(rhoL, (Real)TINY_NUMBER));
  const Real eps_R = pR / (gmo * std::max(rhoR, (Real)TINY_NUMBER));
  const Real eps   = 0.5 * (eps_L + eps_R);
  s.pgas = gmo * s.rho * eps;
  s.h    = 1.0 + eps + s.pgas / std::max(s.rho, (Real)TINY_NUMBER);
  s.K    = s.h;
  s.cs2  = gamma_adi * s.pgas
         / (std::max(s.rho, (Real)TINY_NUMBER) * s.h);
}

// ----------------------------------------------------------------------------
// Eigenvalues -- closed form, Font (2008) eq. (lambdapm)/(lambda0).
//
//   lambda_pm = [ vn (1 - cs^2) +/- cs sqrt( (1 - v^2) [ 1 - vn^2
//                                                    - (v^2 - vn^2) cs^2 ] ) ]
//             / (1 - v^2 cs^2)
//   lambda_0  = vn   (algebraic multiplicity 3: entropy + 2 transverse modes)
//
// Returned in the order (lambda_-, lambda_0, lambda_0, lambda_0, lambda_+) so
// the indexing matches the eigenvector ordering below.
inline void GetEigenValuesRHD(const RHDState &s, Real lambda[NRHD]) {
  const Real cs2 = s.cs2;
  const Real vn  = s.vn;
  const Real v2  = s.v2;
  const Real vn2 = vn * vn;
  const Real one_minus_v2cs2 = 1.0 - v2 * cs2;
  // Discriminant of Donat's eigenvalue formula (always >= 0 for causal EOS
  // and |v| < 1; clamp against tiny negatives from roundoff).
  const Real disc = (1.0 - v2) * (1.0 - vn2 - (v2 - vn2) * cs2);
  const Real sdisc = std::sqrt(std::max(disc, (Real)0.0));
  const Real cs = std::sqrt(std::max(cs2, (Real)0.0));
  const Real inv_denom = 1.0 / one_minus_v2cs2;
  const Real lam_p = (vn * (1.0 - cs2) + cs * sdisc) * inv_denom;
  const Real lam_m = (vn * (1.0 - cs2) - cs * sdisc) * inv_denom;
  lambda[0] = lam_m;   // acoustic, slow
  lambda[1] = vn;      // entropy
  lambda[2] = vn;      // transverse 1
  lambda[3] = vn;      // transverse 2
  lambda[4] = lam_p;   // acoustic, fast
}

// ----------------------------------------------------------------------------
// Right and left eigenvectors -- closed form, Font (2008) eqs. r_{0,1}, r_{0,2},
// r_{0,3}, r_pm and l_{0,1}, l_{0,2}, l_{0,3}, l_mp (Donat et al. 1998).
//
// Storage convention:
//   R[component][wave]   so that  q_face = R . char  (matrix * column vector)
//   L[wave][component]   so that  char  = L . q
// with the wave ordering matching GetEigenValuesRHD:
//   wave 0 = acoustic (lambda_-), 1 = entropy (lambda_0,1),
//   wave 2 = transverse-1 (lambda_0,2), 3 = transverse-2 (lambda_0,3),
//   wave 4 = acoustic (lambda_+).
//
// Returns true on success, false if Delta is too close to zero (degenerate
// transverse-velocity branch -- indicates near-zero (1 - vn^2) or vacuum).
inline bool GetEigenVectorRHD(const RHDState &s, const Real lambda[NRHD],
                              Real L[NRHD][NRHD], Real R[NRHD][NRHD]) {
  const Real h = s.h;
  const Real W = s.W;
  const Real K = s.K;
  const Real vn = s.vn, vt1 = s.vt1, vt2 = s.vt2;
  const Real v2 = s.v2;
  const Real lam_m = lambda[0];
  const Real lam_p = lambda[4];
  const Real one_minus_vn2 = 1.0 - vn * vn;
  if (std::abs(one_minus_vn2) < TINY_NUMBER) return false;

  const Real Aplus  = one_minus_vn2 / (1.0 - vn * lam_p);
  const Real Aminus = one_minus_vn2 / (1.0 - vn * lam_m);

  const Real Delta = h*h*h * W * (K - 1.0) * one_minus_vn2
                   * (Aplus * lam_p - Aminus * lam_m);
  if (std::abs(Delta) < TINY_NUMBER) return false;
  if (std::abs(K - 1.0) < TINY_NUMBER) return false;  // l_{0,1} prefactor

  // ------------------------- right eigenvectors ---------------------------
  // r_{0,1}: entropy mode (Donat eq.)
  const Real K_over_hW = K / (h * W);
  R[ID_RHD  ][1] = K_over_hW;
  R[ISN_RHD ][1] = vn;
  R[IST1_RHD][1] = vt1;
  R[IST2_RHD][1] = vt2;
  R[ITAU_RHD][1] = 1.0 - K_over_hW;

  // r_{0,2}: transverse-1 mode
  const Real W2 = W * W;
  R[ID_RHD  ][2] = W * vt1;
  R[ISN_RHD ][2] = 2.0 * h * W2 * vn  * vt1;
  R[IST1_RHD][2] = h * (1.0 + 2.0 * W2 * vt1 * vt1);
  R[IST2_RHD][2] = 2.0 * h * W2 * vt1 * vt2;
  R[ITAU_RHD][2] = 2.0 * h * W2 * vt1 - W * vt1;

  // r_{0,3}: transverse-2 mode
  R[ID_RHD  ][3] = W * vt2;
  R[ISN_RHD ][3] = 2.0 * h * W2 * vn  * vt2;
  R[IST1_RHD][3] = 2.0 * h * W2 * vt1 * vt2;
  R[IST2_RHD][3] = h * (1.0 + 2.0 * W2 * vt2 * vt2);
  R[ITAU_RHD][3] = 2.0 * h * W2 * vt2 - W * vt2;

  // r_-, r_+: acoustic modes
  R[ID_RHD  ][0] = 1.0;
  R[ISN_RHD ][0] = h * W * Aminus * lam_m;
  R[IST1_RHD][0] = h * W * vt1;
  R[IST2_RHD][0] = h * W * vt2;
  R[ITAU_RHD][0] = h * W * Aminus - 1.0;

  R[ID_RHD  ][4] = 1.0;
  R[ISN_RHD ][4] = h * W * Aplus * lam_p;
  R[IST1_RHD][4] = h * W * vt1;
  R[IST2_RHD][4] = h * W * vt2;
  R[ITAU_RHD][4] = h * W * Aplus - 1.0;

  // ------------------------- left eigenvectors ----------------------------
  // l_{0,1}: entropy
  {
    const Real pref = W / (K - 1.0);
    L[1][ID_RHD  ] = pref * (h - W);
    L[1][ISN_RHD ] = pref * (W * vn);
    L[1][IST1_RHD] = pref * (W * vt1);
    L[1][IST2_RHD] = pref * (W * vt2);
    L[1][ITAU_RHD] = pref * (-W);
  }

  // l_{0,2}: transverse-1 (only depends on vn, vt1)
  {
    const Real pref = 1.0 / (h * one_minus_vn2);
    L[2][ID_RHD  ] = pref * (-vt1);
    L[2][ISN_RHD ] = pref * (vn * vt1);
    L[2][IST1_RHD] = pref * (one_minus_vn2);
    L[2][IST2_RHD] = 0.0;
    L[2][ITAU_RHD] = pref * (-vt1);
  }

  // l_{0,3}: transverse-2 (only depends on vn, vt2)
  {
    const Real pref = 1.0 / (h * one_minus_vn2);
    L[3][ID_RHD  ] = pref * (-vt2);
    L[3][ISN_RHD ] = pref * (vn * vt2);
    L[3][IST1_RHD] = 0.0;
    L[3][IST2_RHD] = pref * (one_minus_vn2);
    L[3][ITAU_RHD] = pref * (-vt2);
  }

  // l_- and l_+: acoustic (Donat l_mp).  The (+/-1) factor is associated with
  // the index pairing l_- <-> r_+ , l_+ <-> r_- (note the swap).  Following
  // Donat 1998 verbatim:
  //   l_-(...) carries an OVERALL +1 (paired with r_+, uses A_+ and lambda_+)
  //   l_+(...) carries an OVERALL -1 (paired with r_-, uses A_- and lambda_-)
  // We arranged the wave order as (acoustic-, entropy, t1, t2, acoustic+) for
  // R, so L's row 0 is the LEFT eigenvector of the lambda_- mode (= l_-) and
  // row 4 is l_+.  Per Donat l_- uses the A_+/lambda_+ block; l_+ uses A_-/lambda_-.
  //
  // To preserve biorthogonality L . R = I we therefore fill:
  //   L[0][.] from the (+1) version with A_+, lambda_+
  //   L[4][.] from the (-1) version with A_-, lambda_-
  const Real h2_over_Delta = (h * h) / Delta;
  const Real twoKm1 = 2.0 * K - 1.0;
  const Real v2_minus_vn2 = v2 - vn * vn;
  // wave 0 (acoustic-): use A_+, lambda_+, sign +1
  {
    const Real A = Aplus,  lam = lam_p, sgn = +1.0;
    const Real comm_tau = -vn - W*W * v2_minus_vn2 * twoKm1 * (vn - A * lam)
                       + K * A * lam;
    L[0][ID_RHD  ] = sgn * h2_over_Delta * (
        h * W * A * (vn - lam) + comm_tau );
    L[0][ISN_RHD ] = sgn * h2_over_Delta * (
        1.0 + W*W * v2_minus_vn2 * twoKm1 * (1.0 - A) - K * A );
    L[0][IST1_RHD] = sgn * h2_over_Delta * (
        W*W * vt1 * twoKm1 * A * (vn - lam) );
    L[0][IST2_RHD] = sgn * h2_over_Delta * (
        W*W * vt2 * twoKm1 * A * (vn - lam) );
    L[0][ITAU_RHD] = sgn * h2_over_Delta * comm_tau;
  }
  // wave 4 (acoustic+): use A_-, lambda_-, sign -1
  {
    const Real A = Aminus, lam = lam_m, sgn = -1.0;
    const Real comm_tau = -vn - W*W * v2_minus_vn2 * twoKm1 * (vn - A * lam)
                       + K * A * lam;
    L[4][ID_RHD  ] = sgn * h2_over_Delta * (
        h * W * A * (vn - lam) + comm_tau );
    L[4][ISN_RHD ] = sgn * h2_over_Delta * (
        1.0 + W*W * v2_minus_vn2 * twoKm1 * (1.0 - A) - K * A );
    L[4][IST1_RHD] = sgn * h2_over_Delta * (
        W*W * vt1 * twoKm1 * A * (vn - lam) );
    L[4][IST2_RHD] = sgn * h2_over_Delta * (
        W*W * vt2 * twoKm1 * A * (vn - lam) );
    L[4][ITAU_RHD] = sgn * h2_over_Delta * comm_tau;
  }

  return true;
}

// ----------------------------------------------------------------------------
// Eigensystem self-check: returns ||L . R - I||_inf.  For a correct biorthogonal
// eigensystem this should be at the level of round-off (~1e-14 in double for a
// well-conditioned state).  Caller decides what to do with the residual --
// typical use is to print it / abort when it exceeds a threshold during
// debugging, or wrap the call in #ifdef DEBUG_EIGENSYSTEM_RHD for production.
inline Real CheckEigenSystemRHD(const Real (&L)[NRHD][NRHD],
                                const Real (&R)[NRHD][NRHD]) {
  Real res = 0.0;
  for (int i = 0; i < NRHD; ++i) {
    for (int j = 0; j < NRHD; ++j) {
      Real acc = 0.0;
      for (int k = 0; k < NRHD; ++k) {
        acc += L[i][k] * R[k][j];
      }
      const Real expected = (i == j) ? 1.0 : 0.0;
      res = std::max(res, std::abs(acc - expected));
    }
  }
  return res;
}

// ----------------------------------------------------------------------------
// Cell-centered conserved variables in the permuted (D, S_n, S_t1, S_t2, tau)
// basis, computed directly FROM PRIMITIVES.  Using primitives (rather than the
// evolved cons array) ensures exact consistency with GetFluxesCellRHD, which is
// critical for the Lax-Friedrichs flux split f^± = (f ± λ·q)/2.  If q came
// from the evolved cons and f from prim, floor clamping or con2prim round-trip
// errors would make them inconsistent → oscillatory reconstruction.
inline void GetConsCellRHD(const RHDState &s, Real q[NRHD]) {
  const Real D     = s.rho * s.W;
  const Real rhohW2 = s.rho * s.h * s.W * s.W;
  q[ID_RHD  ] = D;
  q[ISN_RHD ] = rhohW2 * s.vn;
  q[IST1_RHD] = rhohW2 * s.vt1;
  q[IST2_RHD] = rhohW2 * s.vt2;
  q[ITAU_RHD] = rhohW2 - s.pgas - D;  // tau = E - D = (rho h W^2 - p) - D
}

// ----------------------------------------------------------------------------
// Cell-centered fluxes in the permuted (D, S_n, S_t1, S_t2, tau) basis.
//
//   F^n_D    = D vn
//   F^n_S_n  = S_n vn + p
//   F^n_S_t1 = S_t1 vn
//   F^n_S_t2 = S_t2 vn
//   F^n_tau  = (S_n - D vn) = (rho h W^2 - rho W) vn = rho W (h W - 1) vn
//   (recovered from F^n_E - F^n_D, where F^n_E = S^n)
inline void GetFluxesCellRHD(const RHDState &s, Real flx[NRHD]) {
  const Real D     = s.rho * s.W;
  const Real rhohW2 = s.rho * s.h * s.W * s.W;
  flx[ID_RHD  ] = D * s.vn;
  flx[ISN_RHD ] = rhohW2 * s.vn * s.vn + s.pgas;
  flx[IST1_RHD] = rhohW2 * s.vn * s.vt1;
  flx[IST2_RHD] = rhohW2 * s.vn * s.vt2;
  flx[ITAU_RHD] = (rhohW2 * s.vn) - D * s.vn;   // S_n - F_D
}

// ----------------------------------------------------------------------------
// Build the 6-cell stencil (cells i-3..i+2 for the face at i in the ivx
// direction) of conservative state, flux and per-cell eigenvalues, all in the
// permuted shifted-energy basis.  Used by ReconCharFieldsStencilRHD below.
inline void BuildFaceCompatibleStencilDataRHD(
    const AthenaArray<Real> &prim,
    const RHDDirMap &map, const int k, const int j, const int i, const int ivx,
    const Real gamma_adi,
    Real cons_stencil[6][NRHD], Real flx_stencil[6][NRHD],
    Real lambda_stencil[6][NRHD]) {
  // Offsets relative to face index i: stencil cells are (i-3, i-2, i-1, i, i+1, i+2).
  // For ivx = IVY/IVZ, the offset applies to j or k respectively.
  // Both cons_stencil and flx_stencil are computed FROM PRIMITIVES to guarantee
  // mutual consistency in the LF split (see GetConsCellRHD comment).
  for (int s = 0; s < 6; ++s) {
    const int off = s - 3;
    int kk = k, jj = j, ii = i;
    switch (ivx) {
      case IVX: ii = i + off; break;
      case IVY: jj = j + off; break;
      default:  kk = k + off; break;
    }
    RHDState st;
    GetStateCellRHD(prim, map, kk, jj, ii, gamma_adi, st);
    GetConsCellRHD(st, cons_stencil[s]);
    GetFluxesCellRHD(st, flx_stencil[s]);
    GetEigenValuesRHD(st, lambda_stencil[s]);
  }
}

#if GENERAL_RELATIVITY
// ----------------------------------------------------------------------------
// GR-tetrad pipeline for SR-hydro EFL (Antón 2006 §3.4 step (i)+(ii)).
// Mirrors the SRMHD GRMHD path: transform 6-cell stencil to local Minkowski
// at face (k,j,i) via Coordinates::StencilPrimToLocal[1|2|3], then run the
// unchanged SR-hydro pipeline on tetrad-frame primitives. By the equivalence
// principle, the SR eigensystem and physical flux are correct in the local
// frame.  After ReconFluxRHD writes the tetrad-frame face flux, the caller
// invokes FluxToGlobal[1|2|3] for the inverse transform back to global coords.
// ----------------------------------------------------------------------------

// Build SR RHDState from a single tetrad-frame stencil entry.  After
// StencilPrimToLocal, velocity slots in stencil_entry are direction-permuted
// to match the RHDDirMap convention: map.iv_n is the face-normal, map.iv_t1/t2
// the two tangentials.
inline void BuildRHDStateFromStencilEntry(const Real stencil_entry[NWAVE],
                                          const RHDDirMap &map,
                                          const Real gamma_adi, RHDState &s) {
  s.rho  = stencil_entry[IDN];
  s.pgas = stencil_entry[IPR];
  // Velocities written by Coordinates::StencilPrimToLocal[1|2|3] are tetrad-
  // frame u^(i) (spatial 4-velocity components in the local Minkowski frame),
  // direction-permuted into IVX/IVY/IVZ slots.  RHDDirMap reads the right
  // component for each ivx, regardless of the permutation.
  s.vn  = stencil_entry[map.iv_n];
  s.vt1 = stencil_entry[map.iv_t1];
  s.vt2 = stencil_entry[map.iv_t2];
  // FinalizeStateRHD expects u^i in s.{vn,vt1,vt2} and converts to v^i + W.
  FinalizeStateRHD(s, gamma_adi);
}

// GR-tetrad version of BuildFaceCompatibleStencilDataRHD.  Performs the
// stencil-wide tetrad transform via Coordinates::StencilPrimToLocal[1|2|3],
// then runs the SR-hydro per-cell pipeline on tetrad-frame primitives.  Also
// produces the face avg state (from L=stencil[2], R=stencil[3]) and fills
// out_tetrad_cons(*, i) for FluxToGlobal — matches the GRMHD pattern.
//
// Note: SR-hydro does not use magnetic fields, but the StencilPrimToLocal API
// always takes (bn_face, bcc) parameters.  In hydro builds these are guarded
// out by `#if MAGNETIC_FIELDS_ENABLED` inside the coord implementation, so we
// pass dummy values (`prim` reused as a placeholder for `bcc`).
inline void BuildFaceCompatibleStencilDataTetradGRRHD(
    MeshBlock *pmb,
    const int k, const int j, const int i, const int ivx,
    const AthenaArray<Real> &prim,
    const Real gamma_adi,
    Real cons_stencil[6][NRHD], Real flx_stencil[6][NRHD],
    Real lambda_stencil[6][NRHD],
    RHDState &avg_state, AthenaArray<Real> &out_tetrad_cons) {
  Real stencil_prim[6][NWAVE];
  Real stencil_bbx[6];          // unused for hydro
  const Real bn_face_dummy = 0.0;

  // (i) Stencil-wide tetrad transform — single dispatch for all 6 cells.
  switch (ivx) {
    case IVX:
      pmb->pcoord->StencilPrimToLocal1(k, j, i, bn_face_dummy, prim, prim,
                                       stencil_prim, stencil_bbx);
      break;
    case IVY:
      pmb->pcoord->StencilPrimToLocal2(k, j, i, bn_face_dummy, prim, prim,
                                       stencil_prim, stencil_bbx);
      break;
    default:
      pmb->pcoord->StencilPrimToLocal3(k, j, i, bn_face_dummy, prim, prim,
                                       stencil_prim, stencil_bbx);
      break;
  }

  const RHDDirMap map = GetRHDDirMap(ivx);

  // (ii) Per-cell SR-hydro pipeline on tetrad-frame primitives.
  for (int s = 0; s < 6; ++s) {
    RHDState st;
    BuildRHDStateFromStencilEntry(stencil_prim[s], map, gamma_adi, st);
    GetConsCellRHD(st, cons_stencil[s]);
    GetFluxesCellRHD(st, flx_stencil[s]);
    GetEigenValuesRHD(st, lambda_stencil[s]);
  }

  // (iii) Tetrad-frame avg state from L (s=2) and R (s=3) stencil entries.
  Real avg_prim[NWAVE];
  for (int n = 0; n < NWAVE; ++n) {
    avg_prim[n] = 0.5 * (stencil_prim[2][n] + stencil_prim[3][n]);
  }
  BuildRHDStateFromStencilEntry(avg_prim, map, gamma_adi, avg_state);

  // (iv) Populate tetrad_cons(*, i) for FluxToGlobal.
  // FluxToGlobal[1|2|3] for hydro expects Athena's NHYDRO conserved layout:
  //   cons(IDN)         = D
  //   cons(IEN)         = rhohW^2 - p  = T^00 (SR total energy)
  //   cons(map.in)      = S_n          (face-normal momentum)
  //   cons(map.it1/t2)  = S_t1, S_t2   (tangential momenta, direction-permuted)
  const Real D = avg_state.rho * avg_state.W;
  const Real rhohW2 = avg_state.rho * avg_state.h * avg_state.W * avg_state.W;
  out_tetrad_cons(IDN, i)      = D;
  out_tetrad_cons(IEN, i)      = rhohW2 - avg_state.pgas;
  out_tetrad_cons(map.in,  i)  = rhohW2 * avg_state.vn;
  out_tetrad_cons(map.it1, i)  = rhohW2 * avg_state.vt1;
  out_tetrad_cons(map.it2, i)  = rhohW2 * avg_state.vt2;
}
#endif  // GENERAL_RELATIVITY

// Maximal |lambda| over the 6-cell stencil for each wave -- the LF splitting
// speed used by the characteristic-space Lax-Friedrichs reconstruction.
inline void GetMaximalWaveSpeedStencilRHD(const Real lambda_stencil[6][NRHD],
                                           Real lambda_max[NRHD],
                                           const bool use_global) {
  if (use_global) {
    // Global max: use the single largest wave speed across ALL waves and ALL
    // stencil cells for every component.  Prevents zero LF dissipation when
    // per-wave eigenvalues vanish (e.g., entropy mode at vn=0 in a shock tube).
    Real global_max = 0.0;
    for (int n = 0; n < NRHD; ++n) {
      for (int s = 0; s < 6; ++s) {
        global_max = std::max(global_max, std::abs(lambda_stencil[s][n]));
      }
    }
    global_max = std::min(global_max, (Real)1.0);
    for (int n = 0; n < NRHD; ++n) lambda_max[n] = global_max;
  } else {
    // Local (per-wave) max: each characteristic component gets its own
    // maximum wave speed.  Standard approach; more accurate for smooth flows
    // but can leave entropy/transverse waves with zero dissipation.
    for (int n = 0; n < NRHD; ++n) {
      Real lam_max = 0.0;
      for (int s = 0; s < 6; ++s) {
        lam_max = std::max(lam_max, std::abs(lambda_stencil[s][n]));
      }
      lambda_max[n] = std::min(lam_max, (Real)1.0);
    }
  }
}

// ----------------------------------------------------------------------------
// Characteristic-field reconstruction at a face.  For each wave m, build the
// LF-split characteristic flux from the 6-cell stencil and apply WENO5/WENO5Z/CS5
// to the +/- branches.  Returns char_flx[m] = upwind characteristic flux at the
// face.
inline void ReconCharFieldsStencilRHD(
    const Real flx_stencil[6][NRHD], const Real cons_stencil[6][NRHD],
    const Real lambda_max[NRHD], const Real L[NRHD][NRHD],
    Real char_flx[NRHD], const int rec_kind) {
  // Project conservatives and fluxes onto the characteristic basis at the
  // averaged face state: char_q[s][m] = sum_l L[m][l] q_stencil[s][l].
  Real char_q[6][NRHD] = {};
  Real char_f[6][NRHD] = {};
  for (int s = 0; s < 6; ++s) {
    for (int m = 0; m < NRHD; ++m) {
      Real q_acc = 0.0, f_acc = 0.0;
      for (int l = 0; l < NRHD; ++l) {
        q_acc += L[m][l] * cons_stencil[s][l];
        f_acc += L[m][l] * flx_stencil[s][l];
      }
      char_q[s][m] = q_acc;
      char_f[s][m] = f_acc;
    }
  }
  // LF flux splitting in characteristic space, then high-order reconstruction
  // of the upwind side.
  for (int m = 0; m < NRHD; ++m) {
    const Real lam = lambda_max[m];
    Real fp[5], fm[5];
    for (int s = 0; s < 6; ++s) {
      const Real fps = 0.5 * (char_f[s][m] + lam * char_q[s][m]);  // f^+
      const Real fns = 0.5 * (char_f[s][m] - lam * char_q[s][m]);  // f^-
      if (s < 5) fp[s]   = fps;       // stencil cells i-3..i+1 for +flux
      if (s > 0) fm[s-1] = fns;       // stencil cells i-2..i+2 for -flux (mirrored)
    }
    // f^+ at face uses cells (i-3, i-2, i-1, i, i+1) reading L->R
    const Real fhat_p = ReconstructScalarHORHD(fp, rec_kind);
    // f^- at face uses cells (i+2, i+1, i, i-1, i-2) reading R->L; reverse fm
    Real fm_rev[5] = { fm[4], fm[3], fm[2], fm[1], fm[0] };
    const Real fhat_m = ReconstructScalarHORHD(fm_rev, rec_kind);
    char_flx[m] = fhat_p + fhat_m;
  }
}

// Back-projection: physical flux at face = R . char_flux.
inline void ReconFluxRHD(const Real char_flx[NRHD], const Real R[NRHD][NRHD],
                         Real flx_tau[NRHD]) {
  for (int n = 0; n < NRHD; ++n) {
    Real acc = 0.0;
    for (int m = 0; m < NRHD; ++m) {
      acc += R[n][m] * char_flx[m];
    }
    flx_tau[n] = acc;
  }
}

}  // anonymous namespace

#endif  // HYDRO_CHARACTERISTIC_FIELDS_RHD_HPP_
