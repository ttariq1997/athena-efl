//========================================================================================
// Athena++ astrophysical MHD code -- EFL extension
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file rusanov_hydro_rel.cpp
//! \brief Special-relativistic hydrodynamics high-order Rusanov / Lax-Friedrichs
//!        solver in the shifted-energy basis (D, S_n, S_t1, S_t2, tau).
//!
//! Mirrors src/hydro/rsolvers/mhd/rusanov_mhd_rel.cpp but for pure SR hydro:
//!   * 5-component system (no magnetic field, no EMF outputs).
//!   * 5x5 closed-form Donat 1998 / Font (2008) eigensystem.
//!   * Same per-(k,j) row driver pattern: precompute per-cell state, then
//!     per-face eigensystem + 6-cell stencil + characteristic recon + back-project.
//!
//! NOTE (Phase 1 -- HO only): this file currently provides only the HO entry
//! points that get called when configure --flux=rusanov is selected.  EFL wiring
//! (per-face theta-skip, LO-fallback marker, blend) is NOT yet hooked up; that
//! comes in Phase 2 once the HO scheme is validated standalone.

// C headers

// C++ headers
#include <algorithm>  // max(), min()
#include <cmath>      // abs(), isfinite(), sqrt(), NAN
#include <cstring>    // strcmp
#include <iostream>
#include <limits>
#include <sstream>

// Athena++ headers
#include "../../../athena.hpp"
#include "../../../athena_arrays.hpp"
#include "../../../coordinates/coordinates.hpp"
#include "../../../eos/eos.hpp"
#include "../../../mesh/mesh.hpp"
#include "../../hydro.hpp"
#include "../../CharacteristicFieldsRHD.hpp"

// Optional runtime self-check of the eigensystem at every face.  Enable by
// configuring with -DDEBUG_EIGENSYSTEM_RHD or by adding the macro at compile
// time; disabled by default (cost is ~125 mul-adds per face).
// #define DEBUG_EIGENSYSTEM_RHD

namespace {

// Mark a face's flux as invalid (all components = NaN).  Used by the EFL hook
// in Phase 2 to flag failed eigensystem solves; in Phase 1 a hard-fail aborts.
void MarkInvalidFaceRHD(const int k, const int j, const int i,
                        AthenaArray<Real> &flux) {
  const Real nan = std::numeric_limits<Real>::quiet_NaN();
  for (int n = 0; n < NHYDRO; ++n) {
    flux(n, k, j, i) = nan;
  }
}

const char *DirLabelRHD(const int ivx) {
  switch (ivx) {
    case IVX: return "x1";
    case IVY: return "x2";
    default:  return "x3";
  }
}

// ----------------------------------------------------------------------------
// Per-direction face-loop driver.  For each face in [il..iu] x [jl..ju] x [kl..ku]
// (with the row index along ivx):
//   1. Pack the 6-cell stencil of (cons_tau, flux_tau, lambda) for that face.
//   2. Build face-averaged primitive state -> eigenvalues + L/R eigenvectors.
//   3. LF-split + WENO5/WENO5Z/CS5 in characteristic space.
//   4. Back-project: F_face_tau = R . char_flux.
//   5. Convert tau -> E and write to the lab-frame flux array.
//
// `allow_local_invalid` mirrors the SRMHD path: when true (EFL mode), failed
// eigensystem faces are NaN-marked for the LO fallback to take over; when false
// (standalone HO), failure is fatal.
void RusanovFluxDirRHD(Hydro *ph,
                       AthenaArray<Real> &prim,
                       AthenaArray<Real> &flux_dir,
                       const int ivx,
                       const int kl, const int ku,
                       const int jl, const int ju,
                       const int il, const int iu,
                       const bool allow_local_invalid,
                       const int ho_recon_kind) {
  MeshBlock *pmb = ph->pmy_block;
  const Real gamma_adi = pmb->peos->GetGamma();
  const RHDDirMap map = GetRHDDirMap(ivx);

  flux_dir.ZeroClear();

#if GENERAL_RELATIVITY
  // Tetrad-frame conserved at each face — input to FluxToGlobal[1|2|3].
  // Populated inside BuildFaceCompatibleStencilDataTetradGRRHD per face.
  AthenaArray<Real> tetrad_cons(NHYDRO, pmb->ncells1);
  // Empty placeholder for hydro: bbx, ey, ez are unused inside FluxToGlobal
  // (guarded by `if (MAGNETIC_FIELDS_ENABLED)`); same pattern as hlle_rel.cpp.
  AthenaArray<Real> empty{};
#endif

  for (int k = kl; k <= ku; ++k) {
    for (int j = jl; j <= ju; ++j) {
      for (int i = il; i <= iu; ++i) {
        // -------- (1) build avg state + 6-cell stencil --------------------
        RHDState avg_state;
        Real cons_stencil[6][NRHD] = {};
        Real flx_stencil[6][NRHD] = {};
        Real lambda_stencil[6][NRHD] = {};

#if GENERAL_RELATIVITY
        // GR path: tetrad-transform the 6-cell stencil at face (k,j,i),
        // then build cons/flux/eigvals + avg state on tetrad-frame primitives.
        // Also populates tetrad_cons(*, i) for FluxToGlobal step (iii) below.
        BuildFaceCompatibleStencilDataTetradGRRHD(
            pmb, k, j, i, ivx, prim, gamma_adi,
            cons_stencil, flx_stencil, lambda_stencil,
            avg_state, tetrad_cons);
#else
        // SR path: avg state from L/R cells, stencil from cell-centered prims.
        GetStateAvgRHD(prim, map, k, j, i, ivx, gamma_adi, avg_state);
        BuildFaceCompatibleStencilDataRHD(prim, map, k, j, i, ivx,
                                          gamma_adi,
                                          cons_stencil, flx_stencil,
                                          lambda_stencil);
#endif

        // -------- (2) eigenvalues + eigenvectors at avg state -------------
        // The eigensystem is in tetrad frame for GR (correct by equivalence
        // principle) and in lab frame for SR (lab=tetrad).
        Real lambda_avg[NRHD] = {};
        GetEigenValuesRHD(avg_state, lambda_avg);

        Real L_eig[NRHD][NRHD] = {};
        Real R_eig[NRHD][NRHD] = {};
        const bool eig_ok = GetEigenVectorRHD(avg_state, lambda_avg,
                                              L_eig, R_eig);

        if (!eig_ok) {
          if (allow_local_invalid) {
            MarkInvalidFaceRHD(k, j, i, flux_dir);
            continue;
          }
          std::stringstream msg;
          msg << "### FATAL ERROR in RusanovFluxDirRHD" << std::endl
              << "Eigensystem failure for " << DirLabelRHD(ivx) << " interface "
              << "(i=" << i << ", j=" << j << ", k=" << k << ")" << std::endl
              << "rho=" << avg_state.rho << " p=" << avg_state.pgas
              << " W=" << avg_state.W << " h=" << avg_state.h
              << " cs2=" << avg_state.cs2 << std::endl;
          ATHENA_ERROR(msg);
        }

#ifdef DEBUG_EIGENSYSTEM_RHD
        {
          const Real biorth_res = CheckEigenSystemRHD(L_eig, R_eig);
          if (biorth_res > 1.0e-10) {
            std::stringstream dbg;
            dbg << "[DEBUG_EIGENSYSTEM_RHD] " << DirLabelRHD(ivx)
                << " face (i=" << i << ", j=" << j << ", k=" << k
                << "): ||L*R - I||_inf = " << biorth_res
                << "  rho=" << avg_state.rho << " p=" << avg_state.pgas
                << " W=" << avg_state.W << " v_n=" << avg_state.vn
                << " cs2=" << avg_state.cs2 << std::endl;
            std::cerr << dbg.str();
          }
        }
#endif

        // -------- (3) LF split + WENO5/WENO5Z/CS5 in characteristic space ---
        Real lambda_max[NRHD] = {};
        GetMaximalWaveSpeedStencilRHD(lambda_stencil, lambda_max,
                                      ph->ho_global_lf_speed_);

        Real char_flx[NRHD] = {};
        ReconCharFieldsStencilRHD(flx_stencil, cons_stencil, lambda_max, L_eig,
                                  char_flx, ho_recon_kind);

        // -------- (4) back-project to (D, S_n, S_t1, S_t2, tau) -----------
        Real flx_tau[NRHD] = {};
        ReconFluxRHD(char_flx, R_eig, flx_tau);

        // -------- (5) write tetrad/lab flux to lab-frame flux array -------
        WriteFluxRHD(flx_tau, map, k, j, i, flux_dir);

#if GENERAL_RELATIVITY
        // -------- (6) GR step (iii): transform tetrad-frame flux back to
        //              global GR coords via the face metric.
        switch (ivx) {
          case IVX:
            pmb->pcoord->FluxToGlobal1(k, j, i, i, tetrad_cons, empty,
                                       flux_dir, empty, empty);
            break;
          case IVY:
            pmb->pcoord->FluxToGlobal2(k, j, i, i, tetrad_cons, empty,
                                       flux_dir, empty, empty);
            break;
          default:
            pmb->pcoord->FluxToGlobal3(k, j, i, i, tetrad_cons, empty,
                                       flux_dir, empty, empty);
            break;
        }
#endif

        // sanity: detect NaN/Inf flux candidates
        bool flux_ok = true;
        for (int n = 0; n < NHYDRO; ++n) {
          flux_ok = flux_ok && std::isfinite(flux_dir(n, k, j, i));
        }
        if (!flux_ok) {
          if (allow_local_invalid) {
            MarkInvalidFaceRHD(k, j, i, flux_dir);
            continue;
          }
          std::stringstream msg;
          msg << "### FATAL ERROR in RusanovFluxDirRHD" << std::endl
              << "Non-finite HO SRHD flux candidate for "
              << DirLabelRHD(ivx) << " interface "
              << "(i=" << i << ", j=" << j << ", k=" << k << ")" << std::endl
              << "time=" << pmb->pmy_mesh->time << std::endl;
          ATHENA_ERROR(msg);
        }
      }
    }
  }
}

}  // namespace

// ============================================================================
// Hydro::RusanovFlux entry points (single-direction and 3-direction overloads).
// These are called directly from Hydro::CalculateFluxes when --flux=rusanov is
// selected (mirrors the SRMHD-side dispatch in src/hydro/calculate_fluxes.cpp).
// ============================================================================

// 1D / single-direction overload (used when nx2 == nx3 == 1).
void Hydro::RusanovFlux(AthenaArray<Real> &prim,
                        AthenaArray<Real> &x1flux) {
  MeshBlock *pmb = pmy_block;
#if EFL_DEBUG
  if (ho_counter_cycle_ != pmb->pmy_mesh->ncycle) {
    ho_counter_cycle_ = pmb->pmy_mesh->ncycle;
    ho_hybridized_ = 0;
    ho_pure_ho_ = 0;
  }
#endif
  const bool allow_local_invalid =
#if EFL_ENABLED
      efl_enabled;
#else
      false;
#endif
  RusanovFluxDirRHD(this, prim, x1flux, IVX,
                    pmb->ks, pmb->ke, pmb->js, pmb->je,
                    pmb->is, pmb->ie + 1,
                    allow_local_invalid, ho_recon_);
}

void Hydro::RusanovFlux(AthenaArray<Real> &prim,
                        AthenaArray<Real> &x1flux,
                        AthenaArray<Real> &x2flux,
                        AthenaArray<Real> &x3flux) {
  MeshBlock *pmb = pmy_block;
#if EFL_DEBUG
  if (ho_counter_cycle_ != pmb->pmy_mesh->ncycle) {
    ho_counter_cycle_ = pmb->pmy_mesh->ncycle;
    ho_hybridized_ = 0;
    ho_pure_ho_ = 0;
  }
#endif
  const bool allow_local_invalid =
#if EFL_ENABLED
      efl_enabled;
#else
      false;
#endif

  // x1
  int x1_jl = pmb->js, x1_ju = pmb->je;
  int x1_kl = pmb->ks, x1_ku = pmb->ke;
  if (pmb->pmy_mesh->f2) {
    x1_jl = pmb->js - 1;
    x1_ju = pmb->je + 1;
    if (pmb->pmy_mesh->f3) {
      x1_kl = pmb->ks - 1;
      x1_ku = pmb->ke + 1;
    }
  }
  RusanovFluxDirRHD(this, prim, x1flux, IVX,
                    x1_kl, x1_ku, x1_jl, x1_ju,
                    pmb->is, pmb->ie + 1,
                    allow_local_invalid, ho_recon_);

  // x2
  if (pmb->pmy_mesh->f2) {
    const int x2_kl = pmb->pmy_mesh->f3 ? pmb->ks - 1 : pmb->ks;
    const int x2_ku = pmb->pmy_mesh->f3 ? pmb->ke + 1 : pmb->ke;
    RusanovFluxDirRHD(this, prim, x2flux, IVY,
                      x2_kl, x2_ku, pmb->js, pmb->je + 1,
                      pmb->is - 1, pmb->ie + 1,
                      allow_local_invalid, ho_recon_);
  }

  // x3
  if (pmb->pmy_mesh->f3) {
    RusanovFluxDirRHD(this, prim, x3flux, IVZ,
                      pmb->ks, pmb->ke + 1, pmb->js - 1, pmb->je + 1,
                      pmb->is - 1, pmb->ie + 1,
                      allow_local_invalid, ho_recon_);
  }
}

// ============================================================================
// Stub: the SRHD HO Rusanov solver is driven from Hydro::CalculateFluxes (not
// from a per-row line solver), so the per-row Hydro::RiemannSolver entry must
// never be reached when --flux=rusanov is selected.  Match the SRMHD pattern.
// ============================================================================
#if RSOLVER_IS_RUSANOV && !EFL_ENABLED
void Hydro::RiemannSolver(const int k, const int j, const int il, const int iu,
                          const int ivx,
                          AthenaArray<Real> &wl, AthenaArray<Real> &wr,
                          AthenaArray<Real> &flx, const AthenaArray<Real> &dxw) {
  std::stringstream msg;
  msg << "### FATAL ERROR in Hydro::RiemannSolver" << std::endl
      << "The high-order SRHD Rusanov solver is driven from "
      << "Hydro::CalculateFluxes() and should not enter the low-order "
      << "line-solver interface." << std::endl;
  ATHENA_ERROR(msg);
}
#endif
