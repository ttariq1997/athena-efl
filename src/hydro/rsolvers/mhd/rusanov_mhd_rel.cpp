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
#include "../../CharacteristicFieldsRMHDDirect.hpp"    // Antón §5 direct-conserved route
#include "../../../athena.hpp"
#include "../../../athena_arrays.hpp"
#include "../../../coordinates/coordinates.hpp"
#include "../../../field/field.hpp"
#include "../../../mesh/mesh.hpp"
#include "../../../reconstruct/reconstruction.hpp"  // for Reconstruction::cs5_wp_s*_i / cs5_use_nonuniform_i (non-uniform CS5)

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
        // Stencil-wide b² gate: HO recon uses 6 cells along ivx; if ANY
        // has cell-centered B² ≤ ho_b2_stencil_min, the CS5/WENO smoothness
        // assumption breaks for B and the HO output can poison the EFL
        // blend.  Lab-frame B² is a conservative proxy (B²=0 ⇒ b²=0).
        // Default 0.0 → check is skipped (bit-identical to legacy).
        // Stricter than the face-avg b² gate (which checks only L+R).
        // EFL-only: in non-EFL mode (allow_local_invalid=false) there is
        // no external LO solver to defer to, so the gate is skipped.
        if (ph->ho_b2_stencil_min_ > 0.0 && allow_local_invalid) {
          const Real b2_stencil_min = ph->ho_b2_stencil_min_;
          bool stencil_field_ok = true;
          for (int s = -3; s <= 2; ++s) {
            int kk = k, jj = j, ii = i;
            switch (ivx) {
              case IVX: ii = i + s; break;
              case IVY: jj = j + s; break;
              default:  kk = k + s; break;
            }
            const Real Bsq = SQR(bcc(IB1, kk, jj, ii))
                           + SQR(bcc(IB2, kk, jj, ii))
                           + SQR(bcc(IB3, kk, jj, ii));
            if (Bsq <= b2_stencil_min) {
              stencil_field_ok = false;
              break;
            }
          }
          if (!stencil_field_ok) {
            MarkInvalidFace(k, j, i, flux_dir, emf_t1, emf_t2);
            continue;
          }
        }

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

        // HO reconstruction mode routing:
        //   COMPONENTWISE — skip Anton eigsys entirely; go straight to the
        //                   componentwise Tier-2 flux (Guercilena+17 §2.2 Eq. 7)
        //                   below.  Saves ~40% of per-face HO cost by not
        //                   computing the eigenvector matrices.
        //   AUTO          — try Anton characteristic; on eig_ok=false fall
        //                   through to componentwise Tier-2.  In auto mode the
        //                   Anton L·R tolerance in GetLeftEigenVectorSRMHD is
        //                   already tightened to 1e-6 (via the strict-fallback
        //                   flag propagated at Hydro init), so eig_ok=false
        //                   faithfully reports faces where Tier-1 would produce
        //                   a bad L → route them to Tier-2 instead of LO.
        //   CHARACTERISTIC — legacy path, bit-identical: run eigsys, on
        //                   eig_ok=false take the existing LO/MarkInvalid/FATAL
        //                   fallback.
        const int recon_mode = ph->ho_recon_mode_;
        const bool skip_eigsys = (recon_mode == HO_MODE_COMPONENTWISE);

        if (!skip_eigsys && sigma < 1.0e4 && b2_gate_pass && bn_gate_pass) {
          // In tetrad frame (GR) or lab frame (SR) the eigensystem is identical
          // — avg_state is already in SR form in both cases.
          GetEigenValuesSRMHD(avg_state, lambda_avg);
#if EFL_DEBUG
          ++ph->ho_eig_calls_;
#endif
          // Route the eigenvector build based on recon_mode:
          //   DIRECT_INVERSE   — Antón §5.2 R + T-transform + column scaling
          //                      + L = R⁻¹ via Gauss-Jordan (fast production path)
          //   DIRECT_CONSERVED — Antón §5.2 R + §6.3 direct-conserved L +
          //                      biorthogonality correction (paper-native, ~4×
          //                      slower; useful for paper-reproducibility runs)
          //   other modes      — covariant-first route from base header.
          if (recon_mode == HO_MODE_DIRECT_INVERSE) {
            eig_ok = direct::GetEigenVectorDirectInverseSRMHD(
                avg_state, lambda_avg, L_eig, R_eig);
          } else if (recon_mode == HO_MODE_DIRECT_CONSERVED) {
            eig_ok = direct::GetEigenVectorDirectSRMHD(
                avg_state, lambda_avg, L_eig, R_eig);
          } else {
            eig_ok = GetEigenVectorSRMHD(avg_state, lambda_avg, L_eig, R_eig);
          }
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
        } else if (recon_mode == HO_MODE_CHARACTERISTIC) {
          // eig_ok=false in CHARACTERISTIC mode — legacy LO fallback path
          // (bit-identical to pre-2026-07 behavior).
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
          // failure in CHARACTERISTIC mode, error out loudly rather than
          // silently fall back. (Users hitting this should switch to
          // ho_recon_mode=auto so componentwise Tier-2 catches it instead.)
          {
            std::stringstream msg;
            msg << "### FATAL ERROR in Hydro::RusanovFlux (SR)" << std::endl
                << "HO eigensystem failed (sigma>=1e4 or solver fail) with "
                << "ho_recon_mode=characteristic. Switch to ho_recon_mode=auto "
                << "(default) so the componentwise Tier-2 path handles this "
                << "face, or enable EFL so MarkInvalidFace routes to LLF."
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
        } else {
          // eig_ok=false OR skip_eigsys — AUTO / COMPONENTWISE Tier-2 path.
          // Guercilena+17 §2.2 Eq. 7 componentwise LF split: no eigsys, no
          // characteristic projection.  For AUTO mode this catches Anton
          // ill-conditioning without the LO diffusion penalty.
#if EFL_DEBUG
          if ((recon_mode == HO_MODE_AUTO
               || recon_mode == HO_MODE_DIRECT_CONSERVED
               || recon_mode == HO_MODE_DIRECT_INVERSE)
              && !skip_eigsys) {
            ++ph->ho_hard_fail_;
            ++ph->ho_tier1_reject_;
          }
#endif
#if !GENERAL_RELATIVITY
          // SR path: the eigsys success branch would have built the stencil
          // AFTER eig_ok — but Tier-2 also needs it, so build it here.
          // GR path pre-built the stencil during the tetrad transform (above).
          BuildFaceCompatibleStencilDataSRMHD(
              k, j, i, ivx, u_red, f_red, lambda,
              cons_stencil, flx_stencil, lambda_stencil);
#endif
          GetMaximalWaveSpeedStencilSRMHD(lambda_stencil, lambda_max);

          // Componentwise LF uses a SCALAR κ = max over all 7 wave speeds
          // across the stencil (vs. per-wave λ_max in the characteristic path).
          // Slightly more diffusive but guaranteed finite for finite input.
          Real amax_scalar = 0.0;
          for (int m = 0; m < NRMHD; ++m) {
            if (lambda_max[m] > amax_scalar) amax_scalar = lambda_max[m];
          }

          // Non-uniform CS5 weight fetch (identical logic to Tier-1 block).
          const Real *cs5_wp_face_t2 = nullptr;
          const Real *cs5_wm_face_t2 = nullptr;
          Real cs5_wp_buf_t2[5], cs5_wm_buf_t2[5];
          if (ivx == IVX && pmb->precon->cs5_use_nonuniform_i
              && ho_recon_kind == HO_RECON_CS5) {
            cs5_wp_buf_t2[0] = pmb->precon->cs5_wp_s0_i(i);
            cs5_wp_buf_t2[1] = pmb->precon->cs5_wp_s1_i(i);
            cs5_wp_buf_t2[2] = pmb->precon->cs5_wp_s2_i(i);
            cs5_wp_buf_t2[3] = pmb->precon->cs5_wp_s3_i(i);
            cs5_wp_buf_t2[4] = pmb->precon->cs5_wp_s4_i(i);
            cs5_wm_buf_t2[0] = pmb->precon->cs5_wm_s0_i(i);
            cs5_wm_buf_t2[1] = pmb->precon->cs5_wm_s1_i(i);
            cs5_wm_buf_t2[2] = pmb->precon->cs5_wm_s2_i(i);
            cs5_wm_buf_t2[3] = pmb->precon->cs5_wm_s3_i(i);
            cs5_wm_buf_t2[4] = pmb->precon->cs5_wm_s4_i(i);
            cs5_wp_face_t2 = cs5_wp_buf_t2;
            cs5_wm_face_t2 = cs5_wm_buf_t2;
          }

          ReconComponentwiseFluxStencilSRMHD(
              flx_stencil, cons_stencil, amax_scalar, rflx_face,
              ho_recon_kind, cs5_wp_face_t2, cs5_wm_face_t2);

          WriteReducedFlux(rflx_face, k, j, i, ivx, flux_dir);
          WriteReducedEMF (rflx_face, k, j, i, emf_t1, emf_t2);

#if GENERAL_RELATIVITY
          CallFluxToGlobalSingle(pmb, k, j, i, ivx, tetrad_cons, tetrad_bbx,
                                 flux_dir, emf_t1, emf_t2);
#endif

          // Tier-2 is guaranteed finite for finite input; still guard against
          // upstream NaN in the stencil (e.g. rare user-BC edge cases).
          bool t2_flux_ok = std::isfinite(emf_t1(k, j, i))
                         && std::isfinite(emf_t2(k, j, i));
          for (int n = 0; n < NHYDRO; ++n) {
            t2_flux_ok = t2_flux_ok && std::isfinite(flux_dir(n, k, j, i));
          }
          if (!t2_flux_ok) {
            if (allow_local_invalid) {
              MarkInvalidFace(k, j, i, flux_dir, emf_t1, emf_t2);
              continue;
            }
            std::stringstream msg;
            msg << "### FATAL ERROR in Hydro::RusanovFlux (Tier-2)" << std::endl
                << "Componentwise LF produced non-finite flux at "
                << DirLabel(ivx) << " interface (i=" << i << ", j=" << j
                << ", k=" << k << ")" << std::endl
                << "time=" << pmb->pmy_mesh->time << std::endl;
            ATHENA_ERROR(msg);
          }

#if EFL_DEBUG
          ++ph->ho_tier2_calls_;
#endif

          // Tier-2 CT weight tracking (same as Tier-1 block below).
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
          continue;  // Tier-2 done for this face; skip the Tier-1 recon below.
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

        // Non-uniform CS5 path: when active and reconstructing in the radial
        // direction, fetch per-face Mignone-2014 Vandermonde weights from the
        // Reconstruction class.  Otherwise pass nullptr → uniform textbook
        // (2,-13,47,27,-3)/60 inside ReconstructScalarHO.
        const Real *cs5_wp_face = nullptr;
        const Real *cs5_wm_face = nullptr;
        Real cs5_wp_buf[5], cs5_wm_buf[5];
        if (ivx == IVX && pmb->precon->cs5_use_nonuniform_i
            && ho_recon_kind == HO_RECON_CS5) {
          cs5_wp_buf[0] = pmb->precon->cs5_wp_s0_i(i);
          cs5_wp_buf[1] = pmb->precon->cs5_wp_s1_i(i);
          cs5_wp_buf[2] = pmb->precon->cs5_wp_s2_i(i);
          cs5_wp_buf[3] = pmb->precon->cs5_wp_s3_i(i);
          cs5_wp_buf[4] = pmb->precon->cs5_wp_s4_i(i);
          cs5_wm_buf[0] = pmb->precon->cs5_wm_s0_i(i);
          cs5_wm_buf[1] = pmb->precon->cs5_wm_s1_i(i);
          cs5_wm_buf[2] = pmb->precon->cs5_wm_s2_i(i);
          cs5_wm_buf[3] = pmb->precon->cs5_wm_s3_i(i);
          cs5_wm_buf[4] = pmb->precon->cs5_wm_s4_i(i);
          cs5_wp_face = cs5_wp_buf;
          cs5_wm_face = cs5_wm_buf;
        }
        ReconCharFieldsStencilSRMHD(flx_stencil, cons_stencil, lambda_max,
                                    L_eig, char_flx, ho_recon_kind,
                                    cs5_wp_face, cs5_wm_face);
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
#if EFL_DEBUG
        // Tier-1 (characteristic) success — reached only when eig_ok and the
        // finite check passed.  The Tier-2 (componentwise) branch increments
        // ho_tier2_calls_ in its own success block above.
        ++ph->ho_tier1_calls_;
#endif

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
    ho_tier1_calls_ = 0;
    ho_tier2_calls_ = 0;
    ho_tier1_reject_ = 0;
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
    ho_tier1_calls_ = 0;
    ho_tier2_calls_ = 0;
    ho_tier1_reject_ = 0;
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
