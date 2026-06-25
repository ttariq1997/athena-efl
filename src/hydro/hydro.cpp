//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file hydro.cpp
//! \brief implementation of functions in class Hydro

// C headers

// C++ headers
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../defs.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../mesh/mesh.hpp"
#include "../reconstruct/reconstruction.hpp"
#include "hydro.hpp"
#include "CharacteristicFieldsRMHD.hpp"
#include "hydro_diffusion/hydro_diffusion.hpp"
#include "srcterms/hydro_srcterms.hpp"

//! constructor, initializes data structures and parameters

Hydro::Hydro(MeshBlock *pmb, ParameterInput *pin) :
    pmy_block(pmb), u(NHYDRO, pmb->ncells3, pmb->ncells2, pmb->ncells1),
    w(NHYDRO, pmb->ncells3, pmb->ncells2, pmb->ncells1),
    u1(NHYDRO, pmb->ncells3, pmb->ncells2, pmb->ncells1),
    w1(NHYDRO, pmb->ncells3, pmb->ncells2, pmb->ncells1),
    dvn(pmb->ncells1), dvt(pmb->ncells1),
    // C++11: nested brace-init-list in Hydro member initializer list = aggregate init. of
    // flux[3] array --> direct list init. of each array element --> direct init. via
    // constructor overload resolution of non-aggregate class type AthenaArray<Real>
    flux{ {NHYDRO, pmb->ncells3, pmb->ncells2, pmb->ncells1+1},
          {NHYDRO, pmb->ncells3, pmb->ncells2+1, pmb->ncells1,
           (pmb->pmy_mesh->f2 ? AthenaArray<Real>::DataStatus::allocated :
            AthenaArray<Real>::DataStatus::empty)},
          {NHYDRO, pmb->ncells3+1, pmb->ncells2, pmb->ncells1,
           (pmb->pmy_mesh->f3 ? AthenaArray<Real>::DataStatus::allocated :
            AthenaArray<Real>::DataStatus::empty)}
    },
    coarse_cons_(NHYDRO, pmb->ncc3, pmb->ncc2, pmb->ncc1,
                 (pmb->pmy_mesh->multilevel ? AthenaArray<Real>::DataStatus::allocated :
                  AthenaArray<Real>::DataStatus::empty)),
    coarse_prim_(NHYDRO, pmb->ncc3, pmb->ncc2, pmb->ncc1,
                 (pmb->pmy_mesh->multilevel ? AthenaArray<Real>::DataStatus::allocated :
                  AthenaArray<Real>::DataStatus::empty)),
    hbvar(pmb, &u, &coarse_cons_, flux, HydroBoundaryQuantity::cons),
    hsrc(this, pin),
    hdif(this, pin) {
  int nc1 = pmb->ncells1, nc2 = pmb->ncells2, nc3 = pmb->ncells3;
  Mesh *pm = pmy_block->pmy_mesh;

  pmb->RegisterMeshBlockData(u);

  const std::string ho_recon_name =
      pin->GetOrAddString("time", "xorder_HO", "weno5z");
  if (ho_recon_name == "weno5") {
    ho_recon_ = characterisiticfields::rmhd::HO_RECON_WENO5;
  } else if (ho_recon_name == "weno5z") {
    ho_recon_ = characterisiticfields::rmhd::HO_RECON_WENO5Z;
  } else if (ho_recon_name == "cs5") {
    ho_recon_ = characterisiticfields::rmhd::HO_RECON_CS5;
  } else {
    std::stringstream msg;
    msg << "### FATAL ERROR in Hydro constructor" << std::endl
        << "Unsupported time/xorder_HO='" << ho_recon_name
        << "'. Valid choices are weno5, weno5z, and cs5." << std::endl;
    ATHENA_ERROR(msg);
  }

  // HO eigensystem B² floor: faces whose avg-state b² < this skip the
  // eigenvector solve and fall into the existing failure path (matches
  // the σ < 1e4 high-magnetisation cap, but for the B → 0 limit where
  // the Antón eigenvectors become ill-conditioned).  Default 0.0 (off);
  // 1e-10 catches FM-IC field-free skin cells without affecting any
  // physically meaningful B-loaded face.  Not gated on EFL — applies to
  // all HO Rusanov runs (pure HO and EFL both).
  ho_b2_min_eig_ = pin->GetOrAddReal("hydro", "ho_b2_min_eig", 0.0);
  ho_bn_min_eig_ = pin->GetOrAddReal("hydro", "ho_bn_min_eig", 0.0);
  // Master on/off toggles.  When false, the corresponding threshold check
  // is bypassed entirely (HO runs regardless of avg.bsq or Bn²).  Default
  // true preserves legacy behaviour.
  ho_b2_gate_enable_ = pin->GetOrAddBoolean("hydro", "ho_b2_gate_enable", true);
  ho_bn_gate_enable_ = pin->GetOrAddBoolean("hydro", "ho_bn_gate_enable", true);


#if EFL_ENABLED
  efl_enabled = pin->GetOrAddBoolean("hydro", "efl_enable", false);
  efl_cmax_ = pin->GetOrAddReal("hydro", "efl_cmax", 1.0);
  efl_cE_ = pin->GetOrAddReal("hydro", "efl_cE", 1.0);
  efl_theta_skip_lo_ = pin->GetOrAddReal("hydro", "efl_theta_skip_lo", 1.0e-6);
  efl_buffer_cycles_ = pin->GetOrAddInteger("hydro", "efl_buffer_cycles", 0);
  efl_atm_threshold_ = pin->GetOrAddReal("hydro", "efl_atm_threshold", 0.0);
  // dfloor is read from input here, not via pmb->peos: the EOS object is
  // constructed AFTER Hydro (meshblock.cpp:196 vs :228), so pmb->peos is
  // nullptr at this point.  Reading from pin gives the same value the EOS
  // will see when it constructs, since both pull from the same parsed input.
  // Default mirrors EquationOfState::EquationOfState (sqrt(1024 * float_min)).
  const Real dfloor_local = pin->GetOrAddReal(
      "hydro", "dfloor",
      std::sqrt(1024.0 * std::numeric_limits<float>::min()));
  efl_rho_atm_th_ = efl_atm_threshold_ * dfloor_local;
  ho_global_lf_speed_ = pin->GetOrAddBoolean("hydro", "ho_global_lf_speed", false);
  // Neighbor-aware atm mask: 7-cell-stencil extended LO protection.
  // When enabled, SetAtmMask produces a graded mask (0=body, 0.5=skin,
  // 1.0=deep atm), and the entropy-residual loop forces residual = cmax
  // at any cell whose 7-cell stencil (self + 6 face neighbors) has mask
  // > 0.4 — propagating to θ = 0 at adjacent faces via BuildFaceLimiter.
  efl_atm_mask_enable_ = pin->GetOrAddBoolean("hydro", "efl_atm_mask_enable", false);
  // Hybrid HO/LO scheme (replacement for EFL entropy-residual mechanism)
  hybrid_enable_     = pin->GetOrAddBoolean("hydro", "hybrid_enable", false);
  hybrid_rho_cutoff_ = pin->GetOrAddReal("hydro", "hybrid_rho_cutoff", 0.0);
  // Face-θ clamp: any face whose entropy-residual sensor returns
  // θ ≤ efl_theta_clamp is forced to pure LO.  Default 0.0 (disabled);
  // FM-torus paper recipe uses 0.5 to suppress half-blended cliffs at
  // atm-mask binary transitions.
  efl_theta_clamp_ = pin->GetOrAddReal("hydro", "efl_theta_clamp", 0.0);
  if (efl_enabled) {
    if (!RELATIVISTIC_DYNAMICS) {
      std::stringstream msg;
      msg << "### FATAL ERROR in Hydro constructor" << std::endl
          << "EFL is currently implemented only for SR/GR (with -s or -g)."
          << std::endl;
      ATHENA_ERROR(msg);
    }
    if (std::strcmp(RIEMANN_SOLVER, "rusanov") == 0) {
      std::stringstream msg;
      msg << "### FATAL ERROR in Hydro constructor" << std::endl
          << "EFL requires a low-order base solver; do not combine it with "
          << "RIEMANN_SOLVER=rusanov." << std::endl;
      ATHENA_ERROR(msg);
    }
    if (pmb->precon->xorder == 4) {
      std::stringstream msg;
      msg << "### FATAL ERROR in Hydro constructor" << std::endl
          << "SR-only EFL is not yet wired to Athena's native fourth-order face "
          << "flux correction path." << std::endl;
      ATHENA_ERROR(msg);
    }
    if (NGHOST < 4) {
      std::stringstream msg;
      msg << "### FATAL ERROR in Hydro constructor" << std::endl
          << "SR-only EFL requires NGHOST >= 4 for the entropy-residual stencil."
          << std::endl;
      ATHENA_ERROR(msg);
    }
  }
#endif

  // Allocate optional memory primitive/conserved variable registers for time-integrator
  if (pmb->precon->xorder == 4) {
    // fourth-order hydro cell-centered approximations
    u_cc.NewAthenaArray(NHYDRO, nc3, nc2, nc1);
    w_cc.NewAthenaArray(NHYDRO, nc3, nc2, nc1);
  }

  // If user-requested time integrator is type 3S*, allocate additional memory registers
  std::string integrator = pin->GetOrAddString("time", "integrator", "vl2");
  if (integrator == "ssprk5_4" || STS_ENABLED) {
    // future extension may add "int nregister" to Hydro class
    u2.NewAthenaArray(NHYDRO, nc3, nc2, nc1);
  }

  // If STS RKL2, allocate additional memory registers
  if (STS_ENABLED) {
    std::string sts_integrator = pin->GetOrAddString("time", "sts_integrator", "rkl2");
    if (sts_integrator == "rkl2") {
      u0.NewAthenaArray(NHYDRO, nc3, nc2, nc1);
      fl_div.NewAthenaArray(NHYDRO, nc3, nc2, nc1);
    }
  }

  // "Enroll" in S/AMR by adding to vector of tuples of pointers in MeshRefinement class
  if (pm->multilevel) {
    refinement_idx = pmy_block->pmr->AddToRefinement(&u, &coarse_cons_);
  }

  // enroll HydroBoundaryVariable object
  hbvar.bvar_index = pmb->pbval->bvars.size();
  pmb->pbval->bvars.push_back(&hbvar);
  pmb->pbval->bvars_main_int.push_back(&hbvar);
  if (STS_ENABLED) {
    if (hdif.hydro_diffusion_defined) {
      pmb->pbval->bvars_sts.push_back(&hbvar);
    }
  }

  // Allocate memory for scratch arrays
  dt1_.NewAthenaArray(nc1);
  dt2_.NewAthenaArray(nc1);
  dt3_.NewAthenaArray(nc1);
  dxw_.NewAthenaArray(nc1);
  wl_.NewAthenaArray(NWAVE, nc1);
  wr_.NewAthenaArray(NWAVE, nc1);
  wlb_.NewAthenaArray(NWAVE, nc1);
  x1face_area_.NewAthenaArray(nc1+1);
  if (pm->f2) {
    x2face_area_.NewAthenaArray(nc1);
    x2face_area_p1_.NewAthenaArray(nc1);
  }
  if (pm->f3) {
    x3face_area_.NewAthenaArray(nc1);
    x3face_area_p1_.NewAthenaArray(nc1);
  }
  cell_volume_.NewAthenaArray(nc1);
  dflx_.NewAthenaArray(NHYDRO, nc1);
  if (MAGNETIC_FIELDS_ENABLED && RELATIVISTIC_DYNAMICS) { // only used in (SR/GR)MHD
    bb_normal_.NewAthenaArray(nc1);
  }
  if (RELATIVISTIC_DYNAMICS && std::strcmp(RIEMANN_SOLVER, "hlld") == 0) {
    // only used in (SR/GR)MHD with HLLD
    lambdas_p_l_.NewAthenaArray(nc1);
    lambdas_m_l_.NewAthenaArray(nc1);
    lambdas_p_r_.NewAthenaArray(nc1);
    lambdas_m_r_.NewAthenaArray(nc1);
  }
  if (GENERAL_RELATIVITY) { // only used in GR
    g_.NewAthenaArray(NMETRIC, nc1);
    gi_.NewAthenaArray(NMETRIC, nc1);
    cons_.NewAthenaArray(NWAVE, nc1);
  }

  // fourth-order hydro integration scheme
  if (pmb->precon->xorder == 4) {
    // 4D scratch arrays
    wl3d_.NewAthenaArray(NWAVE, nc3, nc2, nc1);
    wr3d_.NewAthenaArray(NWAVE, nc3, nc2, nc1);
    scr1_nkji_.NewAthenaArray(NHYDRO, nc3, nc2, nc1);
    scr2_nkji_.NewAthenaArray(NHYDRO, nc3, nc2, nc1);
    // 1D scratch arrays
    laplacian_l_fc_.NewAthenaArray(nc1);
    laplacian_r_fc_.NewAthenaArray(nc1);
  }

  // Persistent scratch arrays for the SRMHD HO Rusanov flux (reduced state +
  // per-cell flux + per-cell eigenvalues + face dxw).  The SR-hydro HO path
  // operates directly on cons / prim and does not use these.
#if MAGNETIC_FIELDS_ENABLED
  constexpr int NRMHD_LOCAL = 7;  // must match characterisiticfields::rmhd::NRMHD
  ho_u_red_.NewAthenaArray(NRMHD_LOCAL, nc3, nc2, nc1);
  ho_f_red_.NewAthenaArray(NRMHD_LOCAL, nc3, nc2, nc1);
  ho_lambda_.NewAthenaArray(NRMHD_LOCAL, nc3, nc2, nc1);
  ho_dxw_face_.NewAthenaArray(nc1 + 1);
#endif

#if EFL_ENABLED
  efl_limiter_x1.NewAthenaArray(nc3, nc2, nc1+1);
  if (pm->f2) efl_limiter_x2.NewAthenaArray(nc3, nc2+1, nc1);
  if (pm->f3) efl_limiter_x3.NewAthenaArray(nc3+1, nc2, nc1);

  entropy_residual.NewAthenaArray(nc3, nc2, nc1);
  entropy_curr_.NewAthenaArray(nc3, nc2, nc1);
  entropy_prev1_.NewAthenaArray(nc3, nc2, nc1);
  entropy_prev2_.NewAthenaArray(nc3, nc2, nc1);
  entropy_prev3_.NewAthenaArray(nc3, nc2, nc1);
  // Cell-centered atm mask (allocated regardless of efl_atm_mask_enable_ so
  // outputs.cpp and CombineFluxesDir can reference it unconditionally; it
  // simply stays at ZeroClear when the knob is off).
  atm_mask_.NewAthenaArray(nc3, nc2, nc1);

  // HO flux scratch — production-critical for the blender.  EMF arrays
  // (ho_e*_x*f_) only allocated in MHD mode.
  ho_x1flux_.NewAthenaArray(NHYDRO, nc3, nc2, nc1+1);
  if (pm->f2) ho_x2flux_.NewAthenaArray(NHYDRO, nc3, nc2+1, nc1);
  if (pm->f3) ho_x3flux_.NewAthenaArray(NHYDRO, nc3+1, nc2, nc1);
#if EFL_DEBUG
  // LO snapshot arrays — same shape as HO; populated in CalculateFluxes.
  // Diagnostic only; never read by the blender.
  lo_x1flux_.NewAthenaArray(NHYDRO, nc3, nc2, nc1+1);
  if (pm->f2) lo_x2flux_.NewAthenaArray(NHYDRO, nc3, nc2+1, nc1);
  if (pm->f3) lo_x3flux_.NewAthenaArray(NHYDRO, nc3+1, nc2, nc1);
#endif
#if MAGNETIC_FIELDS_ENABLED
  ho_e3_x1f_.NewAthenaArray(nc3, nc2, nc1+1);
  ho_e2_x1f_.NewAthenaArray(nc3, nc2, nc1+1);
#if EFL_DEBUG
  lo_e3_x1f_.NewAthenaArray(nc3, nc2, nc1+1);
  lo_e2_x1f_.NewAthenaArray(nc3, nc2, nc1+1);
#endif
  if (pm->f2) {
    ho_e1_x2f_.NewAthenaArray(nc3, nc2+1, nc1);
    ho_e3_x2f_.NewAthenaArray(nc3, nc2+1, nc1);
#if EFL_DEBUG
    lo_e1_x2f_.NewAthenaArray(nc3, nc2+1, nc1);
    lo_e3_x2f_.NewAthenaArray(nc3, nc2+1, nc1);
#endif
  }
  if (pm->f3) {
    ho_e2_x3f_.NewAthenaArray(nc3+1, nc2, nc1);
    ho_e1_x3f_.NewAthenaArray(nc3+1, nc2, nc1);
#if EFL_DEBUG
    lo_e2_x3f_.NewAthenaArray(nc3+1, nc2, nc1);
    lo_e1_x3f_.NewAthenaArray(nc3+1, nc2, nc1);
#endif
  }
#endif

  efl_limiter_x1.ZeroClear();
  if (pm->f2) efl_limiter_x2.ZeroClear();
  if (pm->f3) efl_limiter_x3.ZeroClear();
  entropy_residual.ZeroClear();
  entropy_curr_.ZeroClear();
  entropy_prev1_.ZeroClear();
  entropy_prev2_.ZeroClear();
  entropy_prev3_.ZeroClear();
  atm_mask_.ZeroClear();
  ho_x1flux_.ZeroClear();
  if (pm->f2) ho_x2flux_.ZeroClear();
  if (pm->f3) ho_x3flux_.ZeroClear();
#if EFL_DEBUG
  lo_x1flux_.ZeroClear();
  if (pm->f2) lo_x2flux_.ZeroClear();
  if (pm->f3) lo_x3flux_.ZeroClear();
#endif
#if MAGNETIC_FIELDS_ENABLED
  ho_e3_x1f_.ZeroClear();
  ho_e2_x1f_.ZeroClear();
  if (pm->f2) { ho_e1_x2f_.ZeroClear(); ho_e3_x2f_.ZeroClear(); }
  if (pm->f3) { ho_e2_x3f_.ZeroClear(); ho_e1_x3f_.ZeroClear(); }
#if EFL_DEBUG
  lo_e3_x1f_.ZeroClear();
  lo_e2_x1f_.ZeroClear();
  if (pm->f2) { lo_e1_x2f_.ZeroClear(); lo_e3_x2f_.ZeroClear(); }
  if (pm->f3) { lo_e2_x3f_.ZeroClear(); lo_e1_x3f_.ZeroClear(); }
#endif
#endif
#endif  // EFL_ENABLED

  UserTimeStep_ = pmb->pmy_mesh->UserTimeStep_;
}

//----------------------------------------------------------------------------------------
//! \fn Real Hydro::GetWeightForCT(Real dflx, Real rhol, Real rhor, Real dx, Real dt)
//! \brief Calculate the weighting factor for the constrained transport method

Real Hydro::GetWeightForCT(Real dflx, Real rhol, Real rhor, Real dx, Real dt) {
  Real v_over_c = (1024.0)* dt * dflx / (dx * (rhol + rhor));
  Real tmp_min = std::min(static_cast<Real>(0.5), v_over_c);
  return 0.5 + std::max(static_cast<Real>(-0.5), tmp_min);
}

#if EFL_ENABLED
namespace {

// Lagrange-derivative weights of polynomial through (x[k], _) at x[0].
// w[i] = L_i'(x[0]).  Used for the backward-biased time derivative ∂s/∂t
// (npts = 1..4 → 0..3rd-order).
inline void DerivativeWeightsAtCurrent(const Real *x, int npts, Real *w) {
  for (int i = 0; i < npts; ++i) {
    Real accum = 0.0;
    for (int m = 0; m < npts; ++m) {
      if (m == i) continue;
      Real term = 1.0 / (x[i] - x[m]);
      for (int n = 0; n < npts; ++n) {
        if (n == i || n == m) continue;
        term *= (x[0] - x[n]) / (x[i] - x[n]);
      }
      accum += term;
    }
    w[i] = accum;
  }
}

// Derivative of the Lagrange polynomial through (xn[k], f[k]) for k=0..N-1,
// evaluated at x_eval (which may or may not coincide with a node).  Robust
// formula: L_k'(x_eval) = Σ_{m≠k} 1/(x_k-x_m) · ∏_{n≠k,m} (x_eval-x_n)/(x_k-x_n).
// On uniform grid with N=6, skip-center stencil, this collapses to the
// textbook (-1, 9, -45, 45, -9, 1)/(60·h) sixth-order centered formula.
inline Real LagrangeDeriv(const Real *f, const Real *xn, int N, Real x_eval) {
  Real result = 0.0;
  for (int k = 0; k < N; ++k) {
    Real Lkp = 0.0;
    for (int m = 0; m < N; ++m) {
      if (m == k) continue;
      Real term = 1.0 / (xn[k] - xn[m]);
      for (int n = 0; n < N; ++n) {
        if (n == k || n == m) continue;
        term *= (x_eval - xn[n]) / (xn[k] - xn[n]);
      }
      Lkp += term;
    }
    result += f[k] * Lkp;
  }
  return result;
}

// Standard 6-point centered finite-difference coefficients (-1, 9, -45, 45,
// -9, 1)/60 for a uniform grid with spacing dx.  Used as the fast path when
// block_size.x{n}rat == 1.0 — saves ~18× the flops of the full Lagrange form.
inline Real Centered6Uniform(Real fm3, Real fm2, Real fm1,
                             Real fp1, Real fp2, Real fp3, Real dx) {
  return (-fm3 + 9.0*fm2 - 45.0*fm1 + 45.0*fp1 - 9.0*fp2 + fp3) / (60.0 * dx);
}

// Grid-aware 6-point centered derivative ∂a/∂x¹ at cell (k, j, i).
// Skip-center stencil [i-3..i+3]\{i} when both sides have room.  Boundary
// cells where ±3 reach exceeds [lo, hi] use a 6-cell biased window via
// Lagrange (still 5th-order on non-uniform grids).  Stencil width matches
// WENO5Z's reach so the sensor sees every cell the HO reconstruction reads.
//
// `is_unif`/`dx_unif` precomputed at function-entry: when the block grid
// is uniform AND the cell has full ±3 stencil, we use the fast textbook
// coefficients; otherwise fall through to the general Lagrange form.
inline Real DerivX1(const AthenaArray<Real> &a, int k, int j, int i,
                    const AthenaArray<Real> &x1v, int lo, int hi,
                    bool is_unif, Real dx_unif) {
  if (is_unif && i - 3 >= lo && i + 3 <= hi) {
    return Centered6Uniform(a(k,j,i-3), a(k,j,i-2), a(k,j,i-1),
                            a(k,j,i+1), a(k,j,i+2), a(k,j,i+3), dx_unif);
  }
  Real f[6], xn[6];
  if (i - 3 >= lo && i + 3 <= hi) {
    f[0]=a(k,j,i-3); f[1]=a(k,j,i-2); f[2]=a(k,j,i-1);
    f[3]=a(k,j,i+1); f[4]=a(k,j,i+2); f[5]=a(k,j,i+3);
    xn[0]=x1v(i-3); xn[1]=x1v(i-2); xn[2]=x1v(i-1);
    xn[3]=x1v(i+1); xn[4]=x1v(i+2); xn[5]=x1v(i+3);
  } else {
    int s0 = std::max(lo, i - 3);
    if (s0 + 5 > hi) s0 = hi - 5;
    for (int m = 0; m < 6; ++m) { f[m] = a(k,j,s0+m); xn[m] = x1v(s0+m); }
  }
  return LagrangeDeriv(f, xn, 6, x1v(i));
}

inline Real DerivX2(const AthenaArray<Real> &a, int k, int j, int i,
                    const AthenaArray<Real> &x2v, int lo, int hi,
                    bool is_unif, Real dx_unif) {
  if (is_unif && j - 3 >= lo && j + 3 <= hi) {
    return Centered6Uniform(a(k,j-3,i), a(k,j-2,i), a(k,j-1,i),
                            a(k,j+1,i), a(k,j+2,i), a(k,j+3,i), dx_unif);
  }
  Real f[6], xn[6];
  if (j - 3 >= lo && j + 3 <= hi) {
    f[0]=a(k,j-3,i); f[1]=a(k,j-2,i); f[2]=a(k,j-1,i);
    f[3]=a(k,j+1,i); f[4]=a(k,j+2,i); f[5]=a(k,j+3,i);
    xn[0]=x2v(j-3); xn[1]=x2v(j-2); xn[2]=x2v(j-1);
    xn[3]=x2v(j+1); xn[4]=x2v(j+2); xn[5]=x2v(j+3);
  } else {
    int s0 = std::max(lo, j - 3);
    if (s0 + 5 > hi) s0 = hi - 5;
    for (int m = 0; m < 6; ++m) { f[m] = a(k,s0+m,i); xn[m] = x2v(s0+m); }
  }
  return LagrangeDeriv(f, xn, 6, x2v(j));
}

inline Real DerivX3(const AthenaArray<Real> &a, int k, int j, int i,
                    const AthenaArray<Real> &x3v, int lo, int hi,
                    bool is_unif, Real dx_unif) {
  if (is_unif && k - 3 >= lo && k + 3 <= hi) {
    return Centered6Uniform(a(k-3,j,i), a(k-2,j,i), a(k-1,j,i),
                            a(k+1,j,i), a(k+2,j,i), a(k+3,j,i), dx_unif);
  }
  Real f[6], xn[6];
  if (k - 3 >= lo && k + 3 <= hi) {
    f[0]=a(k-3,j,i); f[1]=a(k-2,j,i); f[2]=a(k-1,j,i);
    f[3]=a(k+1,j,i); f[4]=a(k+2,j,i); f[5]=a(k+3,j,i);
    xn[0]=x3v(k-3); xn[1]=x3v(k-2); xn[2]=x3v(k-1);
    xn[3]=x3v(k+1); xn[4]=x3v(k+2); xn[5]=x3v(k+3);
  } else {
    int s0 = std::max(lo, k - 3);
    if (s0 + 5 > hi) s0 = hi - 5;
    for (int m = 0; m < 6; ++m) { f[m] = a(s0+m,j,i); xn[m] = x3v(s0+m); }
  }
  return LagrangeDeriv(f, xn, 6, x3v(k));
}

}  // namespace

void Hydro::CalculateEntropy(const AthenaArray<Real> &prim, AthenaArray<Real> &ent) {
  MeshBlock *pmb = pmy_block;
  const Real gamma = pmb->peos->GetGamma();
  const int nc1 = pmb->ncells1;
  const int nc2 = pmb->ncells2;
  const int nc3 = pmb->ncells3;

  for (int k = 0; k < nc3; ++k) {
    for (int j = 0; j < nc2; ++j) {
      for (int i = 0; i < nc1; ++i) {
        const Real rho = std::max(prim(IDN, k, j, i), pmb->peos->GetDensityFloor());
        const Real pgas = std::max(prim(IPR, k, j, i), pmb->peos->GetPressureFloor());
        ent(k, j, i) = std::log(pgas / std::pow(rho, gamma));
      }
    }
  }
}

void Hydro::CalculateEntropyResidual(const AthenaArray<Real> &prim,
                                     const AthenaArray<Real> &ent) {
  MeshBlock *pmb = pmy_block;
  Mesh *pm = pmb->pmy_mesh;

  // Reset all sensor outputs.  During the buffer cycles, the limiter arrays
  // remain zero — the flux blender then uses pure LO until the entropy
  // history is long enough for a reliable time derivative.
  entropy_residual.ZeroClear();
  efl_limiter_x1.ZeroClear();
  if (pm->f2) efl_limiter_x2.ZeroClear();
  if (pm->f3) efl_limiter_x3.ZeroClear();
  if (pm->ncycle < efl_buffer_cycles_) return;

  // Loop range: face limiter averages cells {i-1, i} across each face, so we
  // need residuals 1 ghost layer wide in each active direction.
  const int is = pmb->is - 1;
  const int ie = pmb->ie + 1;
  const int js = pm->f2 ? pmb->js - 1 : pmb->js;
  const int je = pm->f2 ? pmb->je + 1 : pmb->je;
  const int ks = pm->f3 ? pmb->ks - 1 : pmb->ks;
  const int ke = pm->f3 ? pmb->ke + 1 : pmb->ke;

  // Bounds for the spatial-derivative stencil (full array indices).  At
  // cells where the centered ±3 reach exceeds these, DerivX{1,2,3} biases
  // the 6-cell window inward.
  const int i_lo = 0, i_hi = pmb->ncells1 - 1;
  const int j_lo = 0, j_hi = pmb->ncells2 - 1;
  const int k_lo = 0, k_hi = pmb->ncells3 - 1;

  // Detect uniform grid per direction (block_size.x{n}rat == 1.0).  When
  // uniform, DerivX{n} uses the textbook (-1,9,-45,45,-9,1)/(60·dx) form
  // instead of full Lagrange — same result, ~18× cheaper.
  const bool is_unif_x1 = (pmb->block_size.x1rat == 1.0);
  const Real dx_unif_x1 = is_unif_x1 ? pmb->pcoord->dx1f(pmb->is) : 0.0;
  const bool is_unif_x2 = pm->f2 && (pmb->block_size.x2rat == 1.0);
  const Real dx_unif_x2 = is_unif_x2 ? pmb->pcoord->dx2f(pmb->js) : 0.0;
  const bool is_unif_x3 = pm->f3 && (pmb->block_size.x3rat == 1.0);
  const Real dx_unif_x3 = is_unif_x3 ? pmb->pcoord->dx3f(pmb->ks) : 0.0;

#if GENERAL_RELATIVITY
  AthenaArray<Real> g(NMETRIC, pmb->ncells1);
  AthenaArray<Real> gi(NMETRIC, pmb->ncells1);
#endif

  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je; ++j) {
      // Clamp (k, j) to the active range only for queries that may not be
      // defined at θ-axis ghosts (metric / face widths).  The cell-centered
      // primitives are valid throughout [is, ie] for any (k, j).
      const int kc = std::min(std::max(k, pmb->ks), pmb->ke);
      const int jc = std::min(std::max(j, pmb->js), pmb->je);
      pmb->pcoord->CenterWidth1(kc, jc, is, ie, dxw_);
#if GENERAL_RELATIVITY
      pmb->pcoord->CellMetric(kc, jc, is, ie, g, gi);
#endif
      for (int i = is; i <= ie; ++i) {
        // Extended atm protection (early-out before any Ds/Dt compute):
        // if this cell or any of its 6 face neighbors is in atm
        // (atm_mask_ > 0.4 — catches skin 0.5 and deep atm 1.0), force
        // residual = efl_cmax_ so the downstream face limiter produces θ=0
        // at adjacent faces.  Skips ~250 flops + sqrt per atm-region cell.
        if (efl_atm_mask_enable_) {
          bool any_atm = (atm_mask_(k, j, i) > 0.4)
                      || (i > i_lo && atm_mask_(k, j, i-1) > 0.4)
                      || (i < i_hi && atm_mask_(k, j, i+1) > 0.4)
                      || (j > j_lo && atm_mask_(k, j-1, i) > 0.4)
                      || (j < j_hi && atm_mask_(k, j+1, i) > 0.4)
                      || (k > k_lo && atm_mask_(k-1, j, i) > 0.4)
                      || (k < k_hi && atm_mask_(k+1, j, i) > 0.4);
          if (any_atm) {
            entropy_residual(k, j, i) = efl_cmax_;
            continue;
          }
        }

        const Real ux = prim(IVX, k, j, i);
        const Real uy = prim(IVY, k, j, i);
        const Real uz = prim(IVZ, k, j, i);

        // Residual (entropy advection along coord-time streamlines):
        //   R = ∂_t s + (α v^i − β^i) ∂_i s
        // ADM 3+1 ingredients (generic, any spacetime):
        //   W   = sqrt(1 + γ_ij util^i util^j)        Lorentz factor
        //   v^i = util^i / W                          Eulerian 3-velocity
        //   α   = sqrt(−1 / g^{00})                   lapse
        //   β^i = α² g^{0i}                           shift  (standard ADM)
        // So the transport velocity is dx^i/dt = α v^i − β^i.
        // SR limit: α=1, β^i=0, γ_ij=δ_ij  ⇒  R = ∂_t s + v^i ∂_i s.
        Real W, vtx, vty, vtz;
#if GENERAL_RELATIVITY
        {
          const Real util_sq = g(I11,i)*SQR(ux) + g(I22,i)*SQR(uy) + g(I33,i)*SQR(uz)
                             + 2.0*(g(I12,i)*ux*uy + g(I13,i)*ux*uz + g(I23,i)*uy*uz);
          W = std::sqrt(1.0 + util_sq);
          const Real vx = ux/W, vy = uy/W, vz = uz/W;
          const Real alpha  = std::sqrt(-1.0 / gi(I00, i));
          const Real alpha2 = alpha * alpha;
          const Real betax  = alpha2 * gi(I01, i);
          const Real betay  = alpha2 * gi(I02, i);
          const Real betaz  = alpha2 * gi(I03, i);
          vtx = alpha * vx - betax;
          vty = alpha * vy - betay;
          vtz = alpha * vz - betaz;
        }
#else
        W = std::sqrt(1.0 + SQR(ux) + SQR(uy) + SQR(uz));
        vtx = ux/W; vty = uy/W; vtz = uz/W;
#endif

        // Spatial gradient: 6-point centered.  Uniform grid → fast
        // textbook coefficients; non-uniform or boundary-biased cells →
        // grid-aware Lagrange.  Stencil width matches WENO5Z's reach.
        const Real dsdx = DerivX1(ent, k, j, i, pmb->pcoord->x1v,
                                  i_lo, i_hi, is_unif_x1, dx_unif_x1);
        const Real dsdy = pm->f2 ? DerivX2(ent, k, j, i, pmb->pcoord->x2v,
                                           j_lo, j_hi, is_unif_x2, dx_unif_x2) : 0.0;
        const Real dsdz = pm->f3 ? DerivX3(ent, k, j, i, pmb->pcoord->x3v,
                                           k_lo, k_hi, is_unif_x3, dx_unif_x3) : 0.0;

        // Time derivative: backward-biased Lagrange through up to 4 levels
        // — 1st-order at ncycle=1, 2nd at ncycle=2, 3rd at ncycle≥3.
        Real dts = 0.0;
        if (pm->ncycle >= 1) {
          const int npts = std::min(4, pm->ncycle + 1);
          Real times[4]   = {efl_time_curr_, efl_time_prev1_,
                             efl_time_prev2_, efl_time_prev3_};
          Real weights[4] = {0.0, 0.0, 0.0, 0.0};
          DerivativeWeightsAtCurrent(times, npts, weights);
          dts = weights[0] * ent(k, j, i);
          if (npts > 1) dts += weights[1] * entropy_prev1_(k, j, i);
          if (npts > 2) dts += weights[2] * entropy_prev2_(k, j, i);
          if (npts > 3) dts += weights[3] * entropy_prev3_(k, j, i);
        }

        // Material derivative ∂_t s + v^i ∂_i s.  At cycle 0 there is no
        // time history; fall back to a dimensionless spatial-jump sensor
        // so a stationary discontinuity at startup (e.g. cylindrical blast
        // IC) isn't mistakenly marked smooth.
        Real residual = dts + vtx*dsdx + vty*dsdy + vtz*dsdz;
        if (pm->ncycle == 0) {
          const Real sx = std::abs(dsdx) * dxw_(i);
          const Real sy = pm->f2 ? std::abs(dsdy) * pmb->pcoord->dx2f(jc) : 0.0;
          const Real sz = pm->f3 ? std::abs(dsdz) * pmb->pcoord->dx3f(kc) : 0.0;
          residual = std::max({sx, sy, sz});
        }
        entropy_residual(k, j, i) = std::min(efl_cmax_, efl_cE_ * std::abs(residual));
      }
    }
  }
}

// Build the face-θ limiter for direction Dir (1, 2, or 3).
//
// Reads cell-centered entropy_residual, averages the two cells straddling
// each face along the face-normal axis, maps ER ∈ [0, 1] to θ = clip(1−ER),
// and applies the half-blend clamp (efl_theta_clamp).  Writes efl_limiter_x{Dir}.
//
// Loop bounds:
//   * Face-normal axis runs ±1 (i.e. [is, ie+1] for x1, etc.) — every face.
//   * Transverse axes extend by 1 ghost when active, matching the reach of
//     the downstream CombineFluxesDir blender that reads the limiter.
template <int Dir>
void Hydro::BuildFaceLimiter() {
  static_assert(Dir >= 1 && Dir <= 3, "BuildFaceLimiter Dir must be 1, 2, or 3");
  MeshBlock *pmb = pmy_block;
  Mesh *pm = pmb->pmy_mesh;

  if (Dir == 2 && !pm->f2) return;
  if (Dir == 3 && !pm->f3) return;

  AthenaArray<Real> &limiter = (Dir == 1) ? efl_limiter_x1
                             : (Dir == 2) ? efl_limiter_x2
                                          : efl_limiter_x3;

  // Per-direction transverse halo: ±1 in any active non-normal axis (so
  // CombineFluxesDir's reach can read the limiter at neighbour rows).
  const int is = pmb->is - ((Dir != 1) ? 1 : 0);
  const int ie = pmb->ie + ((Dir != 1) ? 1 : 0);
  const int js = pmb->js - ((Dir != 2 && pm->f2) ? 1 : 0);
  const int je = pmb->je + ((Dir != 2 && pm->f2) ? 1 : 0);
  const int ks = pmb->ks - ((Dir != 3 && pm->f3) ? 1 : 0);
  const int ke = pmb->ke + ((Dir != 3 && pm->f3) ? 1 : 0);

  // Extend by 1 in the face-normal direction (face count = cell count + 1).
  const int iu = ie + ((Dir == 1) ? 1 : 0);
  const int ju = je + ((Dir == 2) ? 1 : 0);
  const int ku = ke + ((Dir == 3) ? 1 : 0);

  for (int k = ks; k <= ku; ++k) {
    for (int j = js; j <= ju; ++j) {
      for (int i = is; i <= iu; ++i) {
        // 2-cell average across the face (left + right of the face plane).
        const int km = (Dir == 3) ? k - 1 : k;
        const int jm = (Dir == 2) ? j - 1 : j;
        const int im = (Dir == 1) ? i - 1 : i;
        const Real er_face = 0.5 * (entropy_residual(km, jm, im)
                                  + entropy_residual(k,  j,  i));
        const Real theta_raw = std::max((Real)0.0, std::min((Real)1.0, 1.0 - er_face));
        // Optional half-blend clamp (knob efl_theta_clamp, default 0.0 = off).
        // Setting it > 0 forces θ → 0 at faces where the sensor is only
        // partially confident, preventing wrong-sign HO contributions from
        // dominating in the half-blended regime at atm-mask transitions.
        limiter(k, j, i) = (theta_raw <= efl_theta_clamp_) ? (Real)0.0 : theta_raw;
      }
    }
  }
}

// Explicit instantiations for the three face directions.
template void Hydro::BuildFaceLimiter<1>();
template void Hydro::BuildFaceLimiter<2>();
template void Hydro::BuildFaceLimiter<3>();

void Hydro::UpdateEFL() {
  // efl_enabled is the APPLY gate (CalculateFluxes), not the compute gate.
  // When the compile flag EFL_ENABLED is set, UpdateEFL ALWAYS runs so that
  // entropy_residual / atm_mask_ / efl_limiter_x{1,2,3} stay live even with
  // efl_enable=false — that path is the "diagnostic LO" mode (HO + blend
  // skipped in CalculateFluxes via do_efl=false, but EFL fields keep
  // updating so they can be dumped and inspected against pure-LO dynamics).
  Mesh *pm = pmy_block->pmy_mesh;
  efl_time_curr_ = pmy_block->pmy_mesh->time;
  // Build the cell-centered atm mask first; it's consumed by both the HO
  // Riemann fast-skip and CombineFluxesDir's gate later this cycle.
  if (efl_atm_mask_enable_) {
    SetAtmMask(w);
  }
  CalculateEntropy(w, entropy_curr_);
  CalculateEntropyResidual(w, entropy_curr_);
  if (pm->ncycle < efl_buffer_cycles_) return;
  BuildFaceLimiter<1>();
  BuildFaceLimiter<2>();
  BuildFaceLimiter<3>();
}

// Graded atm mask with hard-coded pts = 1 (immediate face neighbors only):
//   atm_mask_(k,j,i) = 0.0   if prim(IDN) > rho_atm_th         (body)
//   atm_mask_(k,j,i) = 0.5   if prim(IDN) ≤ rho_atm_th AND any of the 6 face
//                            neighbors is body                  (skin)
//   atm_mask_(k,j,i) = 1.0   if prim(IDN) ≤ rho_atm_th AND all 6 face
//                            neighbors are also atm-density     (deep atm)
//
// The graded values cooperate with the 7-cell-stencil check in
// CalculateEntropyResidual (threshold > 0.4) to force entropy_residual =
// efl_cmax_ at any cell whose stencil touches atm — extending the LO zone
// by one body cell layer beyond the legacy face-min density gate.
//
// Loop covers ALL cells (ghosts included) since downstream face gates and
// the entropy-residual stencil reference ghost rows.  Neighbor lookups
// clamp to [0, ncells-1] so polar / radial domain edges are handled
// safely without false-positive body neighbors past the boundary.
void Hydro::SetAtmMask(const AthenaArray<Real> &prim) {
  MeshBlock *pmb = pmy_block;
  const int iu = pmb->ncells1 - 1;
  const int ju = pmb->ncells2 - 1;
  const int ku = pmb->ncells3 - 1;
  const Real rho_atm_th = efl_rho_atm_th_;

  for (int k = 0; k <= ku; ++k) {
    for (int j = 0; j <= ju; ++j) {
      for (int i = 0; i <= iu; ++i) {
        int atm;
        if (prim(IDN, k, j, i) > rho_atm_th) {
          atm = 0;                                  // body
        } else {
          atm = 1;                                  // at atm density (skin candidate)
          bool body_neighbor = false;
          if (i > 0  && prim(IDN, k, j, i-1) > rho_atm_th) body_neighbor = true;
          else if (i < iu && prim(IDN, k, j, i+1) > rho_atm_th) body_neighbor = true;
          else if (j > 0  && prim(IDN, k, j-1, i) > rho_atm_th) body_neighbor = true;
          else if (j < ju && prim(IDN, k, j+1, i) > rho_atm_th) body_neighbor = true;
          else if (k > 0  && prim(IDN, k-1, j, i) > rho_atm_th) body_neighbor = true;
          else if (k < ku && prim(IDN, k+1, j, i) > rho_atm_th) body_neighbor = true;
          if (!body_neighbor) atm = 2;              // deep atm (no body face-neighbor)
        }
        atm_mask_(k, j, i) = static_cast<Real>(atm) * 0.5;   // 0 / 0.5 / 1.0
      }
    }
  }
}

void Hydro::AdvanceEntropyHistory() {
  // Same rationale as UpdateEFL: shift the entropy history every cycle so
  // diagnostic-LO mode (efl_enable=false in an -efl build) still carries a
  // consistent Ds/Dt stencil and writes valid restart state.
  MeshBlock *pmb = pmy_block;
  const int nc1 = pmb->ncells1;
  const int nc2 = pmb->ncells2;
  const int nc3 = pmb->ncells3;

  for (int k = 0; k < nc3; ++k) {
    for (int j = 0; j < nc2; ++j) {
      for (int i = 0; i < nc1; ++i) {
        entropy_prev3_(k, j, i) = entropy_prev2_(k, j, i);
        entropy_prev2_(k, j, i) = entropy_prev1_(k, j, i);
        entropy_prev1_(k, j, i) = entropy_curr_(k, j, i);
      }
    }
  }

  efl_time_prev3_ = efl_time_prev2_;
  efl_time_prev2_ = efl_time_prev1_;
  efl_time_prev1_ = efl_time_curr_;
}

// Restart helpers — see hydro.hpp for rationale.  Layout (per MeshBlock):
//   entropy_prev1_   nc3*nc2*nc1*sizeof(Real)
//   entropy_prev2_   nc3*nc2*nc1*sizeof(Real)
//   entropy_prev3_   nc3*nc2*nc1*sizeof(Real)
//   efl_time_prev1_  sizeof(Real)
//   efl_time_prev2_  sizeof(Real)
//   efl_time_prev3_  sizeof(Real)
std::size_t Hydro::EflRestartSize() const {
  return entropy_prev1_.GetSizeInBytes()
       + entropy_prev2_.GetSizeInBytes()
       + entropy_prev3_.GetSizeInBytes()
       + 3 * sizeof(Real);
}

std::size_t Hydro::EflRestartPack(char *dst) const {
  char *p = dst;
  std::memcpy(p, entropy_prev1_.data(), entropy_prev1_.GetSizeInBytes());
  p += entropy_prev1_.GetSizeInBytes();
  std::memcpy(p, entropy_prev2_.data(), entropy_prev2_.GetSizeInBytes());
  p += entropy_prev2_.GetSizeInBytes();
  std::memcpy(p, entropy_prev3_.data(), entropy_prev3_.GetSizeInBytes());
  p += entropy_prev3_.GetSizeInBytes();
  std::memcpy(p, &efl_time_prev1_, sizeof(Real)); p += sizeof(Real);
  std::memcpy(p, &efl_time_prev2_, sizeof(Real)); p += sizeof(Real);
  std::memcpy(p, &efl_time_prev3_, sizeof(Real)); p += sizeof(Real);
  return static_cast<std::size_t>(p - dst);
}

std::size_t Hydro::EflRestartUnpack(const char *src) {
  const char *p = src;
  std::memcpy(entropy_prev1_.data(), p, entropy_prev1_.GetSizeInBytes());
  p += entropy_prev1_.GetSizeInBytes();
  std::memcpy(entropy_prev2_.data(), p, entropy_prev2_.GetSizeInBytes());
  p += entropy_prev2_.GetSizeInBytes();
  std::memcpy(entropy_prev3_.data(), p, entropy_prev3_.GetSizeInBytes());
  p += entropy_prev3_.GetSizeInBytes();
  std::memcpy(&efl_time_prev1_, p, sizeof(Real)); p += sizeof(Real);
  std::memcpy(&efl_time_prev2_, p, sizeof(Real)); p += sizeof(Real);
  std::memcpy(&efl_time_prev3_, p, sizeof(Real)); p += sizeof(Real);
  return static_cast<std::size_t>(p - src);
}
#endif
