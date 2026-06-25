//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file rusanov_mhd_rel.cpp
//! \brief SRMHD high-order Rusanov flux path in the shifted-energy basis.

#include <cmath>
#include <limits>
#include <sstream>

#include "../../hydro.hpp"
#include "../../CharacteristicFieldsRMHD.hpp"
#include "../../../athena.hpp"
#include "../../../athena_arrays.hpp"
#include "../../../coordinates/coordinates.hpp"
#include "../../../field/field.hpp"
#include "../../../mesh/mesh.hpp"

using namespace characterisiticfields::rmhd;

namespace {

const char *DirLabel(const int ivx) {
  switch (ivx) {
    case IVX:
      return "x1";
    case IVY:
      return "x2";
    default:
      return "x3";
  }
}

void MarkInvalidFace(const int k, const int j, const int i,
                     AthenaArray<Real> &flux_dir,
                     AthenaArray<Real> &emf_t1,
                     AthenaArray<Real> &emf_t2) {
  const Real nan = std::numeric_limits<Real>::quiet_NaN();
  for (int n = 0; n < NHYDRO; ++n) {
    flux_dir(n, k, j, i) = nan;
  }
  emf_t1(k, j, i) = nan;
  emf_t2(k, j, i) = nan;
}

// REMOVED: BuildLocalReducedRusanovFallbackFace function
// This fallback mechanism is no longer needed as we want to identify
// and fix root causes rather than mask them with fallbacks.
// If eigensystem fails or produces non-finite fluxes, the code will
// now error immediately to help debug the underlying issues.

void RusanovFluxDir(Hydro *ph,
                    AthenaArray<Real> &prim,
                    AthenaArray<Real> &cons,
                    const AthenaArray<Real> &bn_face,
                    AthenaArray<Real> &bcc,
                    AthenaArray<Real> &flux_dir,
                    AthenaArray<Real> &emf_t1,
                    AthenaArray<Real> &emf_t2,
                    AthenaArray<Real> *wct_dir,
                    const int ivx,
                    const int kl, const int ku,
                    const int jl, const int ju,
                    const int il, const int iu,
                    const bool allow_local_invalid,
                    const int ho_recon_kind) {
  MeshBlock *pmb = ph->pmy_block;
  const bool track_wct = (wct_dir != nullptr) && !allow_local_invalid;
  flux_dir.ZeroClear();
  emf_t1.ZeroClear();
  emf_t2.ZeroClear();
  if (track_wct) {
    wct_dir->ZeroClear();
  }

  // Scratch arrays for reduced state, flux, eigenvalues (per-cell precompute).
  // These are passed by reference from the multi-direction RusanovFlux wrapper
  // to avoid heap allocation per direction (3× per RK stage).
  // NOTE: In the GR tetrad pipeline these three arrays are NOT used — the
  // stencil builder reads prim/bcc directly and the LO fallback uses tetrad-
  // frame stencil data. We keep the Hydro member scratch around only for the
  // SR path. Skipping the SR-style precompute in GR saves an eigenvalue-quartic
  // solve per cell × 3 directions × N_RK substages per timestep.
  AthenaArray<Real> &u_red = ph->ho_u_red_;
  AthenaArray<Real> &f_red = ph->ho_f_red_;
  AthenaArray<Real> &lambda = ph->ho_lambda_;
  AthenaArray<Real> &dxw_face = ph->ho_dxw_face_;

#if GENERAL_RELATIVITY
  // Tetrad-frame scratch for FluxToGlobal: tetrad_cons (NWAVE×ncells1) and
  // tetrad_bbx (ncells1) carry the avg-state tetrad-frame conserved + face
  // normal B at each face along this sweep, populated inside
  // BuildFaceCompatibleStencilDataTetradGRMHD and consumed by FluxToGlobal.
  // No L/R prim scratch needed — the stencil tetrad transform produces
  // everything needed for both stencil cells AND the avg state.
  AthenaArray<Real> tetrad_bbx(pmb->ncells1);
  AthenaArray<Real> tetrad_cons(NWAVE, pmb->ncells1);
#else
  // SR precompute: u_red (reduced cell-averaged conserved, E→τ reduction and
  // direction-mapping done here), f_red (cell-centered reduced flux built
  // entirely from cell-center prim + bcc — same convention as SRHD), and
  // lambda (per-cell eigenvalues). All three are consumed by the HO stencil
  // builder via direct array reads.
  for (int k = 0; k < pmb->ncells3; ++k) {
    for (int j = 0; j < pmb->ncells2; ++j) {
      GetReducedStateSRMHD(k, j, ivx, 0, pmb->ncells1 - 1, cons, bcc, u_red);
      GetFluxesSRMHD(k, j, ivx, 0, pmb->ncells1 - 1, prim, bcc, u_red, f_red);
      GetEigenValuesCellsSRMHD(pmb, k, j, ivx, 0, pmb->ncells1 - 1, prim, bcc,
                               u_red, lambda);
    }
  }
#endif

  for (int k = kl; k <= ku; ++k) {
    for (int j = jl; j <= ju; ++j) {
      switch (ivx) {
        case IVX:
          pmb->pcoord->CenterWidth1(k, j, il, iu, dxw_face);
          break;
        case IVY:
          pmb->pcoord->CenterWidth2(k, j, il, iu, dxw_face);
          break;
        default:
          pmb->pcoord->CenterWidth3(k, j, il, iu, dxw_face);
          break;
      }
      // EFL HO short-circuit: if the face limiter θ is below threshold,
      // the blended flux is dominated by the LO contribution and the HO
      // work would be multiplied by ~0. Skip entirely — flux_dir/emf/wct
      // are already ZeroClear'd at function entry, and CombineFluxesDirMHD
      // treats a finite-zero HO flux as valid (blend → (1-θ)·LO ≈ LO).
#if EFL_ENABLED
      const bool efl_skip_enabled = ph->efl_enabled && allow_local_invalid;
      const Real theta_lo = ph->efl_theta_skip_lo_;
      const AthenaArray<Real> *efl_lim = nullptr;
      if (efl_skip_enabled) {
        switch (ivx) {
          case IVX: efl_lim = &ph->efl_limiter_x1; break;
          case IVY: efl_lim = &ph->efl_limiter_x2; break;
          default:  efl_lim = &ph->efl_limiter_x3; break;
        }
      }
#if GENERAL_RELATIVITY
      // GRMHD atm fast-skip:  CombineFluxesDirMHD's atm gate fires the
      // face to LO.  Mirror that gate here and short-circuit — flux/EMF
      // stay at ZeroClear so the downstream gate sees the kept LO flux,
      // identical to the no-skip path but without wasted HO compute.
      // Two modes (must match CombineFluxesDirMHD):
      //   efl_atm_mask_enable_=true → cell-mask: skip if either neighbor
      //                                cell has atm_mask_=1 (deep atm).
      //   else if efl_rho_atm_th_>0 → legacy: skip if min(ρ_L,ρ_R)≤thresh.
      const bool atm_mask_enabled = efl_skip_enabled
                                    && ph->efl_atm_mask_enable_;
      const bool atm_check_enabled = efl_skip_enabled
                                     && !ph->efl_atm_mask_enable_
                                     && (ph->efl_rho_atm_th_ > 0.0);
      const Real rho_atm_th = ph->efl_rho_atm_th_;
#endif
#endif
      for (int i = il; i <= iu; ++i) {
#if EFL_ENABLED
        if (efl_skip_enabled && (*efl_lim)(k, j, i) <= theta_lo) {
          continue;  // outputs already zero; HO contribution negligible
        }
#if GENERAL_RELATIVITY
        if (atm_mask_enabled) {
          bool either_atm = false;
          switch (ivx) {
            case IVX: either_atm = (ph->atm_mask_(k, j, i-1) > 0.5)
                                || (ph->atm_mask_(k, j, i)   > 0.5); break;
            case IVY: either_atm = (ph->atm_mask_(k, j-1, i) > 0.5)
                                || (ph->atm_mask_(k, j, i)   > 0.5); break;
            default:  either_atm = (ph->atm_mask_(k-1, j, i) > 0.5)
                                || (ph->atm_mask_(k, j, i)   > 0.5); break;
          }
          if (either_atm) {
            continue;  // mask says deep atm; HO work would be wasted
          }
        } else if (atm_check_enabled) {
          Real rho_low;
          switch (ivx) {
            case IVX: rho_low = std::min(prim(IDN, k, j, i-1),
                                          prim(IDN, k, j, i)); break;
            case IVY: rho_low = std::min(prim(IDN, k, j-1, i),
                                          prim(IDN, k, j, i)); break;
            default:  rho_low = std::min(prim(IDN, k-1, j, i),
                                          prim(IDN, k, j, i)); break;
          }
          if (rho_low <= rho_atm_th) {
            continue;  // atm gate downstream forces LO; HO work would be wasted
          }
        }
#endif
#endif
        RMHDState avg_state{};
        Real lambda_avg[NRMHD] = {};
        Real L_eig[NRMHD][NRMHD] = {};
        Real R_eig[NRMHD][NRMHD] = {};
        Real lambda_max[NRMHD] = {};
        Real char_flx[NRMHD] = {};
        Real rflx_face[NRMHD] = {};
        Real cons_stencil[6][NRMHD] = {};
        Real flx_stencil[6][NRMHD] = {};
        Real lambda_stencil[6][NRMHD] = {};

#if GENERAL_RELATIVITY
        // Antón 2006 tetrad-frame pipeline (§3.4, arXiv:astro-ph/0506063):
        // The stencil-aware tetrad transform handles ALL transformation work
        // for this face: produces per-cell tetrad cons/flux/eigval stencils
        // for the HO reconstruction, AND the tetrad avg state (built from
        // L=stencil[2] and R=stencil[3] post-transform — no redundant L/R
        // PrimToLocal call), AND fills tetrad_cons(*, i) + tetrad_bbx(i) for
        // FluxToGlobal consumption after reconstruction completes.
        BuildFaceCompatibleStencilDataTetradGRMHD(
            pmb, k, j, i, ivx, prim, bn_face, bcc,
            cons_stencil, flx_stencil, lambda_stencil,
            avg_state, tetrad_cons, tetrad_bbx);
#else
        GetStateAvgSRMHD(pmb, k, j, i, ivx, prim, bn_face, bcc, u_red, avg_state);
#endif

        // Skip eigensystem for three regimes where the Antón decomposition is
        // ill-conditioned at the avg state.  Per-cell stencil quantities
        // (U, F, λ) stay finite at B → 0 / Bn → 0 — only the eigenvector
        // matrix L, R becomes rank-deficient — so the gate intercepts at the
        // eigsys step and leaves the upstream stencil pipeline untouched.
        //   (1) σ ≥ 1e4  — extremely magnetically dominated; magnetosonic
        //                  speeds collapse onto the Alfvén speed.
        //   (2) b² ≤ ho_b2_min_eig — B → 0 limit; Alfvén and slow speeds
        //                  collapse, eigenvector inversion amplifies FP noise.
        //                  Catches "no field at all" cells (atm, field nulls).
        //                  Default ho_b2_min_eig = 0 (disabled).  Master
        //                  toggle ho_b2_gate_enable (default true) bypasses
        //                  this check entirely when false.
        //   (3) Bn² ≤ ho_bn_min_eig — Antón Type-I degeneracy; 5 of 7
        //                  eigenvalues collapse to v_n, cond(R) ~ 1/Bn².
        //                  Catches faces where total b is healthy but the
        //                  face-normal component vanishes — the canonical
        //                  failure mode for purely poloidal FM-torus ICs in
        //                  the x3 (φ) sweep (B^φ_lab = 0 → tetrad Bn small).
        //                  Literature precedent: Mignone+ 2009 HLLD; Mattia
        //                  & Mignone 2022 HLLC→HLL fallback.  Default 0
        //                  (disabled); recommended 1e-6 in production.
        //                  Master toggle ho_bn_gate_enable (default true)
        //                  bypasses this check entirely when false.
        // In all three cases eig_ok stays false → failure path (MarkInvalidFace
        // in EFL mode → external LLF; tetrad LO Rusanov in non-EFL GR;
        // FATAL in non-EFL SR — all preserved unchanged).
        const Real sigma = (avg_state.rho > 0.0)
            ? avg_state.bsq / (2.0 * avg_state.rho) : 1.0e30;
        bool eig_ok = false;

        const bool b2_gate_pass =
            !ph->ho_b2_gate_enable_ || (avg_state.bsq > ph->ho_b2_min_eig_);
        const bool bn_gate_pass =
            !ph->ho_bn_gate_enable_ || (SQR(avg_state.Bn) > ph->ho_bn_min_eig_);

        if (sigma < 1.0e4 && b2_gate_pass && bn_gate_pass) {
          // In tetrad frame (GR) or lab frame (SR) the eigensystem is identical
          // — avg_state is already in SR form in both cases.
          GetEigenValuesSRMHD(avg_state, lambda_avg);
#if EFL_DEBUG
          ++ph->ho_eig_calls_;
#endif
          eig_ok = GetEigenVectorSRMHD(avg_state, lambda_avg, L_eig, R_eig);
        }

        // Track statistics and handle eigensystem failure
        if (eig_ok) {
          // eigensystem succeeded — proceed to characteristic reconstruction
#if EFL_DEBUG
          // Diagnostic: bin |L·R - I| diagonal & off-diagonal errors so we
          // see the actual conditioning distribution of the renormalized
          // eigensystem on this run.  Pure diagnostic — never gates anything.
          {
            const PaperProductSummarySRMHD summary =
                SummarizeProductPaperSRMHD(L_eig, R_eig);
            auto bin_idx = [](Real err) -> int {
              if (!std::isfinite(err)) return 4;
              if (err < 1.0e-12) return 0;
              if (err < 1.0e-10) return 1;
              if (err < 1.0e-8)  return 2;
              if (err < 1.0e-6)  return 3;
              return 4;
            };
            ++ph->lr_diag_bins_[bin_idx(summary.max_diag_err)];
            ++ph->lr_off_bins_ [bin_idx(summary.max_offdiag)];
          }
#endif
        } else {
          // Complete eigensystem failure — use low-order Rusanov fallback
#if EFL_DEBUG
          ++ph->ho_hard_fail_;
#endif
          // In EFL mode, mark face invalid for external solver
          if (allow_local_invalid) {
            MarkInvalidFace(k, j, i, flux_dir, emf_t1, emf_t2);
            continue;
          }

          Real rflx_fb[NRMHD] = {};
#if GENERAL_RELATIVITY
          // LO fallback: tetrad-frame Rusanov using the already-built
          // stencil L/R (indices 2 and 3) = cells i-1 and i in sweep dir.
          // Near the horizon in relativistic flow, cell-centered U_L/U_R
          // differ by large factors (per-cell W); this over-dissipates but
          // is stable. Activates rarely (only when eigenvector solver
          // numerics fail on extreme states).
          Real lam_max_fb = 0.0;
          for (int m = 0; m < NRMHD; ++m) {
            lam_max_fb = std::max(lam_max_fb, std::abs(lambda_stencil[2][m]));
            lam_max_fb = std::max(lam_max_fb, std::abs(lambda_stencil[3][m]));
          }
          lam_max_fb = std::min(lam_max_fb, (Real)1.0);
          for (int n = 0; n < NRMHD; ++n) {
            rflx_fb[n] = 0.5 * (flx_stencil[2][n] + flx_stencil[3][n])
                       - 0.5 * lam_max_fb
                             * (cons_stencil[3][n] - cons_stencil[2][n]);
          }
#else
          // SR LO Rusanov fallback disabled — the EFL scheme's MarkInvalidFace
          // path (allow_local_invalid branch above) is how we handle eigen-
          // system failures now. If a user runs non-EFL SR and hits eigensystem
          // failure, error out loudly rather than silently fall back.
          {
            std::stringstream msg;
            msg << "### FATAL ERROR in Hydro::RusanovFlux (SR)" << std::endl
                << "HO eigensystem failed (sigma>=1e4 or solver fail) but the "
                << "SR LO Rusanov fallback has been disabled. Run with EFL "
                << "enabled (hydro/efl_enable=true) so failed faces fall back "
                << "to the external LLF solver, or re-enable the LO fallback."
                << std::endl;
            ATHENA_ERROR(msg);
          }
#endif

          // Write through the proper reduced-to-Athena mapping functions.
          // This correctly handles the tau↔E shift and EMF sign conventions.
          WriteReducedFlux(rflx_fb, k, j, i, ivx, flux_dir);
          WriteReducedEMF(rflx_fb, k, j, i, emf_t1, emf_t2);

#if GENERAL_RELATIVITY
          // Antón 2006 §3.4 step (iii): transform tetrad-frame flux back to
          // global coordinates via the face metric — same pipeline as the HO
          // success path.
          CallFluxToGlobalSingle(pmb, k, j, i, ivx, tetrad_cons, tetrad_bbx,
                                 flux_dir, emf_t1, emf_t2);
#endif

          continue; // Exit after fallback - don't run characteristic reconstruction
        }

        // Only run characteristic reconstruction if eigensystem was successful.
        // In GR, the tetrad-frame stencil was already built before the sigma
        // check (so the LO fallback could share it); in SR we build it here.
#if !GENERAL_RELATIVITY
        BuildFaceCompatibleStencilDataSRMHD(
            k, j, i, ivx, u_red, f_red, lambda,
            cons_stencil, flx_stencil, lambda_stencil);
#endif
        GetMaximalWaveSpeedStencilSRMHD(lambda_stencil, lambda_max);
        ReconCharFieldsStencilSRMHD(flx_stencil, cons_stencil, lambda_max,
                                    L_eig, char_flx, ho_recon_kind);
        ReconFluxRMHD(k, j, i, ivx, char_flx, R_eig, flux_dir, emf_t1, emf_t2);

#if GENERAL_RELATIVITY
        // Antón 2006 §3.4 step (iii): transform tetrad-frame HO flux back to
        // global coordinates via the face metric.
        CallFluxToGlobalSingle(pmb, k, j, i, ivx, tetrad_cons, tetrad_bbx,
                               flux_dir, emf_t1, emf_t2);
#endif

        bool flux_ok = std::isfinite(emf_t1(k, j, i)) && std::isfinite(emf_t2(k, j, i));
        for (int n = 0; n < NHYDRO; ++n) {
          flux_ok = flux_ok && std::isfinite(flux_dir(n, k, j, i));
        }
        if (!flux_ok) {
          // Non-finite flux detected - no fallback, error immediately
          if (allow_local_invalid) {
            MarkInvalidFace(k, j, i, flux_dir, emf_t1, emf_t2);
            continue;
          }
          // No fallback - error immediately to identify root cause
          std::stringstream msg;
          msg << "### FATAL ERROR in Hydro::RusanovFlux" << std::endl
              << "Non-finite HO SRMHD flux candidate for "
              << DirLabel(ivx) << " interface "
              << "(i=" << i << ", j=" << j << ", k=" << k << ")" << std::endl
              << "time=" << pmb->pmy_mesh->time << std::endl;
          ATHENA_ERROR(msg);
        }

        if (track_wct) {
          Real rho_l = 0.0;
          Real rho_r = 0.0;
          switch (ivx) {
            case IVX:
              rho_l = prim(IDN, k, j, i - 1);
              rho_r = prim(IDN, k, j, i);
              break;
            case IVY:
              rho_l = prim(IDN, k, j - 1, i);
              rho_r = prim(IDN, k, j, i);
              break;
            default:
              rho_l = prim(IDN, k - 1, j, i);
              rho_r = prim(IDN, k, j, i);
              break;
          }
          (*wct_dir)(k, j, i) =
              ph->GetWeightForCT(flux_dir(IDN, k, j, i), rho_l, rho_r,
                                 dxw_face(i), pmb->pmy_mesh->dt);
        }
      }
    }
  }
}

}  // namespace

#if RSOLVER_IS_RUSANOV && !EFL_ENABLED
void Hydro::RiemannSolver(const int k, const int j, const int il, const int iu,
                          const int ivx, const AthenaArray<Real> &bx,
                          AthenaArray<Real> &wl, AthenaArray<Real> &wr,
                          AthenaArray<Real> &flx,
                          AthenaArray<Real> &ey, AthenaArray<Real> &ez,
                          AthenaArray<Real> &wct, const AthenaArray<Real> &dxw) {
  std::stringstream msg;
  msg << "### FATAL ERROR in Hydro::RiemannSolver" << std::endl
      << "The high-order SRMHD Rusanov solver is driven from Hydro::CalculateFluxes()"
      << " and should not enter the low-order line-solver interface." << std::endl;
  ATHENA_ERROR(msg);
}
#endif

void Hydro::RusanovFlux(AthenaArray<Real> &prim,
                        AthenaArray<Real> &cons,
                        FaceField &b,
                        AthenaArray<Real> &bcc,
                        AthenaArray<Real> &x1flux,
                        AthenaArray<Real> &e3_x1f,
                        AthenaArray<Real> &e2_x1f) {
  MeshBlock *pmb = pmy_block;

#if EFL_DEBUG
  if (ho_counter_cycle_ != pmb->pmy_mesh->ncycle) {
    ho_counter_cycle_ = pmb->pmy_mesh->ncycle;
    ho_eig_calls_ = 0;
    ho_hard_fail_ = 0;
    ho_hybridized_ = 0;
    ho_pure_ho_ = 0;
    for (int b = 0; b < 5; ++b) {
      lr_diag_bins_[b] = 0;
      lr_off_bins_[b]  = 0;
    }
  }
#endif

  const bool allow_local_invalid =
#if EFL_ENABLED
      efl_enabled;
#else
      false;
#endif
  RusanovFluxDir(this, prim, cons, b.x1f, bcc, x1flux, e3_x1f, e2_x1f,
                 nullptr, IVX, pmb->ks, pmb->ke, pmb->js, pmb->je, pmb->is, pmb->ie + 1,
                 allow_local_invalid,
                 ho_recon_);
}

void Hydro::RusanovFlux(AthenaArray<Real> &prim,
                        AthenaArray<Real> &cons,
                        FaceField &b,
                        AthenaArray<Real> &bcc,
                        AthenaArray<Real> &x1flux,
                        AthenaArray<Real> &e3_x1f,
                        AthenaArray<Real> &e2_x1f,
                        AthenaArray<Real> &w_x1f,
                        AthenaArray<Real> &x2flux,
                        AthenaArray<Real> &e1_x2f,
                        AthenaArray<Real> &e3_x2f,
                        AthenaArray<Real> &w_x2f,
                        AthenaArray<Real> &x3flux,
                        AthenaArray<Real> &e2_x3f,
                        AthenaArray<Real> &e1_x3f,
                        AthenaArray<Real> &w_x3f) {
  MeshBlock *pmb = pmy_block;

#if EFL_DEBUG
  if (ho_counter_cycle_ != pmb->pmy_mesh->ncycle) {
    ho_counter_cycle_ = pmb->pmy_mesh->ncycle;
    ho_eig_calls_ = 0;
    ho_hard_fail_ = 0;
    ho_hybridized_ = 0;
    ho_pure_ho_ = 0;
    for (int b = 0; b < 5; ++b) {
      lr_diag_bins_[b] = 0;
      lr_off_bins_[b]  = 0;
    }
  }
#endif

  const bool allow_local_invalid =
#if EFL_ENABLED
      efl_enabled;
#else
      false;
#endif

  int x1_jl = pmb->js;
  int x1_ju = pmb->je;
  int x1_kl = pmb->ks;
  int x1_ku = pmb->ke;
  if (pmb->pmy_mesh->f2) {
    x1_jl = pmb->js - 1;
    x1_ju = pmb->je + 1;
    if (pmb->pmy_mesh->f3) {
      x1_kl = pmb->ks - 1;
      x1_ku = pmb->ke + 1;
    }
  }

  RusanovFluxDir(this, prim, cons, b.x1f, bcc, x1flux, e3_x1f, e2_x1f,
                 &w_x1f, IVX, x1_kl, x1_ku, x1_jl, x1_ju,
                 pmb->is, pmb->ie + 1,
                 allow_local_invalid,
                 ho_recon_);

  if (pmb->pmy_mesh->f2) {
    const int x2_kl = pmb->pmy_mesh->f3 ? pmb->ks - 1 : pmb->ks;
    const int x2_ku = pmb->pmy_mesh->f3 ? pmb->ke + 1 : pmb->ke;
    RusanovFluxDir(this, prim, cons, b.x2f, bcc, x2flux, e1_x2f, e3_x2f,
                   &w_x2f, IVY, x2_kl, x2_ku, pmb->js, pmb->je + 1,
                   pmb->is - 1, pmb->ie + 1,
                   allow_local_invalid,
                   ho_recon_);
  }

  if (pmb->pmy_mesh->f3) {
    RusanovFluxDir(this, prim, cons, b.x3f, bcc, x3flux, e2_x3f, e1_x3f,
                   &w_x3f, IVZ, pmb->ks, pmb->ke + 1, pmb->js - 1, pmb->je + 1,
                   pmb->is - 1, pmb->ie + 1,
                   allow_local_invalid,
                   ho_recon_);
  }
}
