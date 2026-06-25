#ifndef HYDRO_HYDRO_HPP_
#define HYDRO_HYDRO_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file hydro.hpp
//! \brief definitions for Hydro class

// C headers

// C++ headers
#include <cstdint>
#include <string>
#include <vector>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../bvals/cc/hydro/bvals_hydro.hpp"
#include "hydro_diffusion/hydro_diffusion.hpp"
#include "srcterms/hydro_srcterms.hpp"

class MeshBlock;
class ParameterInput;

// TODO(felker): consider adding a struct FaceFlux w/ overloaded ctor in athena.hpp, or:
// using FaceFlux = AthenaArray<Real>[3];

//! \class Hydro
//! \brief hydro data and functions

class Hydro {
  friend class Field;
  friend class EquationOfState;
 public:
  Hydro(MeshBlock *pmb, ParameterInput *pin);

  // data
  // TODO(KGF): make this private, if possible
  MeshBlock* pmy_block;    // ptr to MeshBlock containing this Hydro

  // conserved and primitive variables
  AthenaArray<Real> u, w;        // time-integrator memory register #1
  AthenaArray<Real> u1, w1;      // time-integrator memory register #2
  AthenaArray<Real> u2;          // time-integrator memory register #3
  AthenaArray<Real> u0, fl_div; // rkl2 STS memory registers;
  // for the HL3D2 solver
  AthenaArray<Real> dvn, dvt;
  // (no more than MAX_NREGISTER allowed)

  AthenaArray<Real> flux[3];  // face-averaged flux vector

  // Persistent scratch arrays for HO Rusanov flux — allocated once, reused.
  // Populated by RusanovFluxDir's precompute loop (u_red, f_red, lambda per
  // cell in sweep-direction order), then consumed by the HO stencil builder
  // via direct array reads.
  AthenaArray<Real> ho_u_red_, ho_f_red_, ho_lambda_;
  AthenaArray<Real> ho_dxw_face_;

  // Hard B² minimum for HO eigensystem.  Faces whose tetrad/SR-frame
  // avg-state b² falls below this skip eigenvector construction → fall
  // into the existing failure path (GR: tetrad LO Rusanov fallback;
  // SR+EFL: MarkInvalidFace → external LLF; SR-only: FATAL).  Mirrors
  // the σ < 1e4 high-magnetisation cap; here we guard the opposite limit
  // where the Antón eigenvectors become ill-conditioned as B → 0.
  // Default 0.0 → disabled (current behaviour bit-identical).
  Real ho_b2_min_eig_{0.0};

  // Master enable for the b² gate.  When false the threshold check is
  // bypassed entirely (HO runs regardless of avg.bsq).  Default true
  // preserves legacy behaviour: with the threshold default = 0.0 the
  // gate trips only at exactly-zero bsq, which is the historical
  // safety net.  Use false to allow CS5 to fire even on truly zero-B
  // faces (useful for diagnosing whether HO/LO transitions at the
  // field-cutoff drive entropy accumulation).
  bool ho_b2_gate_enable_{true};

  // Hard Bn² minimum for HO eigensystem (face-normal magnetic field
  // squared in the tetrad/SR frame at the avg state).  Companion to
  // ho_b2_min_eig_: that catches B → 0 (zero TOTAL field); this catches
  // Bn → 0 (zero FACE-NORMAL field, healthy tangential b).  Triggers
  // the Antón Type-I degeneracy where 5 eigenvalues collapse onto v_n
  // (entropy + 2 Alfvén + 2 slow), making cond(R) ~ 1/Bn² and L,R
  // ill-conditioned even though per-cell U, F, λ stay finite.
  // Literature: Mignone, Ugliano & Bodo 2009 MNRAS 393, 1141 self-
  // diagnosed HLLC-MB failing the 3D rotor test from this; Mattia &
  // Mignone 2022 fall back HLLC → HLL at degenerate states.  Our gate
  // is the GR/SR-EFL analogue: failed faces use LLF (EFL) or tetrad
  // LO Rusanov (non-EFL GR).  Default 0.0 → disabled.  Recommended
  // value: 1e-6 (i.e. |Bn| > 1e-3, comfortably above type_i_tol).
  Real ho_bn_min_eig_{0.0};

  // Master enable for the Bn² gate.  See ho_b2_gate_enable_ for
  // semantics — same on/off pattern, applied to the Bn² threshold.
  bool ho_bn_gate_enable_{true};

#if EFL_ENABLED
  AthenaArray<Real> efl_limiter_x1;  // x1 face limiter used by SR-only EFL
  AthenaArray<Real> efl_limiter_x2;
  AthenaArray<Real> efl_limiter_x3;
  AthenaArray<Real> entropy_residual;  // cell-centered entropy residual
  bool efl_enabled{false};
  Real efl_theta_skip_lo_{1.0e-6};        // skip HO per-face work if θ ≤ this
  bool ho_global_lf_speed_{false};         // use global max wave speed in LF split
#if EFL_DEBUG
  // Per-cycle hybrid/pure-HO face counters (written from CombineFluxesDir*,
  // reset at HO entry).  ho_hybridized_ = faces with 0 < theta < 1;
  // ho_pure_ho_ = faces with theta ~ 1.  Diagnostic only.
  std::int64_t ho_hybridized_{0};
  std::int64_t ho_pure_ho_{0};
  // Per-cycle histograms of max|L·R - I| diag/off-diag errors per HO face.
  // Buckets: [0] <1e-12  [1] <1e-10  [2] <1e-8  [3] <1e-6  [4] >=1e-6 or NaN.
  // Populated in rusanov_mhd_rel.cpp after a successful eigsys.
  std::int64_t lr_diag_bins_[5]{};
  std::int64_t lr_off_bins_[5]{};
#endif
  // Atmosphere-aware LO fallback at the flux blender.  When > 0, any face
  // whose min(rho_L, rho_R) <= efl_rho_atm_th_ is forced to LO regardless
  // of theta.  Mirrors gr-athena++'s rho_atm_th = efl_th * dfloor logic in
  // calculate_fluxes.cpp; needed for stable TOV / NS evolutions where the
  // sharp surface would otherwise drive HO into unphysical states.
  // Read from CombineFluxesDirHydro (free function), so must be public.
  Real efl_atm_threshold_{0.0};   // factor on dfloor; 0 disables
  Real efl_rho_atm_th_{0.0};      // = efl_atm_threshold_ * dfloor (cached)
  // ----------------------------------------------------------------------
  // Neighbor-aware ATM MASK (extended LO protection at the disk surface).
  // When efl_atm_mask_enable_ = true, a cell-centered graded mask is
  // computed once per cycle (in UpdateEFL) using the immediate face
  // neighbors only (hard-coded pts = 1):
  //     atm_mask_(k,j,i) = 0.0   if prim(IDN) > rho_atm_th  (body)
  //     atm_mask_(k,j,i) = 0.5   if prim(IDN) ≤ rho_atm_th AND any of
  //                              the 6 face neighbors is body (skin)
  //     atm_mask_(k,j,i) = 1.0   if prim(IDN) ≤ rho_atm_th AND no body
  //                              face neighbor (deep atm)
  // The CalculateEntropyResidual loop short-circuits any cell whose 7-cell
  // stencil (self + 6 face neighbors) contains mask > 0.4: residual is
  // forced to efl_cmax_ and the per-cell Ds/Dt work is skipped.  Downstream
  // BuildFaceLimiter then produces θ = 0 at faces inside or adjacent to
  // the atm region — extending LO protection by one body cell layer.
  AthenaArray<Real> atm_mask_;     // cell-centered graded mask (allocated when EFL on)
  bool efl_atm_mask_enable_{false}; // master toggle for the extended atm protection
  // ----------------------------------------------------------------------
  // Hybrid HO/LO scheme — fully separate from the EFL entropy-residual
  // mechanism above.  When hybrid_enable_ = true:
  //   * EFL machinery (entropy residual, face limiter, atm threshold) is
  //     BYPASSED completely.
  //   * Per face: theta = 1 (pure HO) if min(rho_L, rho_R) >= rho_cutoff,
  //                theta = 0 (pure LO) otherwise.
  //   * The HO flux is still computed (via the same RusanovFlux path),
  //     and CombineFluxesDirHydro performs the binary blend.
  // Hydro path only for now.  Default off.
  bool hybrid_enable_{false};
  Real hybrid_rho_cutoff_{0.0};   // hard absolute-density threshold
  // Face-θ clamp threshold (knob efl_theta_clamp).  Any face whose
  // entropy-residual sensor returns θ ≤ this threshold is forced to pure
  // LO.  Default 0.0 → disabled.
  Real efl_theta_clamp_{0.0};
#endif

  // storage for SMR/AMR
  // TODO(KGF): remove trailing underscore or revert to private:
  AthenaArray<Real> coarse_cons_, coarse_prim_;
  int refinement_idx{-1};

  // fourth-order intermediate quantities
  AthenaArray<Real> u_cc, w_cc;      // cell-centered approximations

  HydroBoundaryVariable hbvar;
  HydroSourceTerms hsrc;
  HydroDiffusion hdif;

  // functions
  void NewBlockTimeStep();    // computes new timestep on a MeshBlock
  void AddFluxDivergence(const Real wght, AthenaArray<Real> &u_out);
  void AddFluxDivergence_STS(const Real wght, int stage,
                             AthenaArray<Real> &u_out,
                             AthenaArray<Real> &fl_div_out,
                             std::vector<int> idx_subset);
  void CalculateFluxes(AthenaArray<Real> &w, FaceField &b,
                       AthenaArray<Real> &bcc, const int order);
  void CalculateFluxes_STS();
#if EFL_ENABLED
  void UpdateEFL();
  void AdvanceEntropyHistory();
  void SetAtmMask(const AthenaArray<Real> &prim);
  // Restart support for EFL: the entropy time-history (3 previous snapshots
  // + their times) cannot be re-derived from current primitives — without it
  // the Lagrange Ds/Dt collapses to 1 point and the sensor stays at 0 until
  // 3 cycles have accumulated, which silently drops EFL protection right
  // after a restart.  Everything else (entropy_curr_, entropy_residual,
  // atm_mask_, face limiters) rebuilds in the first post-restart UpdateEFL.
  std::size_t EflRestartSize() const;
  std::size_t EflRestartPack(char *dst) const;
  std::size_t EflRestartUnpack(const char *src);
#endif
#if MAGNETIC_FIELDS_ENABLED
  void RusanovFlux(AthenaArray<Real> &prim, AthenaArray<Real> &cons,
                   FaceField &b, AthenaArray<Real> &bcc,
                   AthenaArray<Real> &x1flux,
                   AthenaArray<Real> &e3_x1f,
                   AthenaArray<Real> &e2_x1f);
  void RusanovFlux(AthenaArray<Real> &prim, AthenaArray<Real> &cons,
                   FaceField &b, AthenaArray<Real> &bcc,
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
                   AthenaArray<Real> &w_x3f);
#else  // SR-hydro HO Rusanov: no FaceField, no EMF, no cons (stencil from prim)
  void RusanovFlux(AthenaArray<Real> &prim,
                   AthenaArray<Real> &x1flux);
  void RusanovFlux(AthenaArray<Real> &prim,
                   AthenaArray<Real> &x1flux,
                   AthenaArray<Real> &x2flux,
                   AthenaArray<Real> &x3flux);
#endif
#if !MAGNETIC_FIELDS_ENABLED  // Hydro:
  void RiemannSolver(
      const int k, const int j, const int il, const int iu,
      const int ivx,
      AthenaArray<Real> &wl, AthenaArray<Real> &wr, AthenaArray<Real> &flx,
      const AthenaArray<Real> &dxw);
#else  // MHD:
  void RiemannSolver(
      const int k, const int j, const int il, const int iu,
      const int ivx, const AthenaArray<Real> &bx,
      AthenaArray<Real> &wl, AthenaArray<Real> &wr, AthenaArray<Real> &flx,
      AthenaArray<Real> &ey, AthenaArray<Real> &ez,
      AthenaArray<Real> &wct, const AthenaArray<Real> &dxw);
#endif
  void CalculateVelocityDifferences(const int k, const int j, const int il, const int iu,
    const int ivx, AthenaArray<Real> &dvn, AthenaArray<Real> &dvt);
  Real GetWeightForCT(Real dflx, Real rhol, Real rhor, Real dx, Real dt);

#if EFL_DEBUG
  // Per-cycle diagnostic accessors (EFL_DEBUG only).  Summed across
  // meshblocks (and across MPI ranks via MPI_Reduce) in main.cpp, then
  // printed as a CSV row prefixed by "efl_debug,".
  std::int64_t GetHOEigCallsCount()    const { return ho_eig_calls_; }
  std::int64_t GetHOHardFailCount()    const { return ho_hard_fail_; }
  std::int64_t GetHOHybridizedCount()  const { return ho_hybridized_; }
  std::int64_t GetHOPureHOCount()      const { return ho_pure_ho_; }
  std::int64_t GetLRDiagBin(int b)     const {
    return (b >= 0 && b < 5) ? lr_diag_bins_[b] : 0;
  }
  std::int64_t GetLROffBin(int b)      const {
    return (b >= 0 && b < 5) ? lr_off_bins_[b]  : 0;
  }
#endif

 private:
  AthenaArray<Real> dt1_, dt2_, dt3_;  // scratch arrays used in NewTimeStep
  // scratch space used to compute fluxes
  AthenaArray<Real> dxw_;
  AthenaArray<Real> x1face_area_, x2face_area_, x3face_area_;
  AthenaArray<Real> x2face_area_p1_, x3face_area_p1_;
  AthenaArray<Real> cell_volume_;
  // 2D
  AthenaArray<Real> wl_, wr_, wlb_;
  AthenaArray<Real> dflx_;
  AthenaArray<Real> bb_normal_;    // normal magnetic field, for (SR/GR)MHD
  AthenaArray<Real> lambdas_p_l_;  // most positive wavespeeds in left state
  AthenaArray<Real> lambdas_m_l_;  // most negative wavespeeds in left state
  AthenaArray<Real> lambdas_p_r_;  // most positive wavespeeds in right state
  AthenaArray<Real> lambdas_m_r_;  // most negative wavespeeds in right state
  // 2D GR
  AthenaArray<Real> g_, gi_;       // metric and inverse, for timesteps and some rsolvers
  AthenaArray<Real> cons_;         // conserved state, for some GR Riemann solvers

  // fourth-order hydro
  // 4D scratch arrays
  AthenaArray<Real> scr1_nkji_, scr2_nkji_;
  AthenaArray<Real> wl3d_, wr3d_;
  // 1D scratch arrays
  AthenaArray<Real> laplacian_l_fc_, laplacian_r_fc_;

  int ho_recon_{0};

#if EFL_DEBUG
  // High-order eigensystem counters (per-cycle, diagnostic only).
  std::int64_t ho_eig_calls_{0};
  std::int64_t ho_hard_fail_{0};
  int ho_counter_cycle_{-1};
#endif

#if EFL_ENABLED
 public:
  // HO flux/EMF arrays — production-critical for the EFL blender.
  AthenaArray<Real> ho_x1flux_, ho_e3_x1f_, ho_e2_x1f_;
  AthenaArray<Real> ho_x2flux_, ho_e1_x2f_, ho_e3_x2f_;
  AthenaArray<Real> ho_x3flux_, ho_e2_x3f_, ho_e1_x3f_;
#if EFL_DEBUG
  // LO snapshot arrays — copied from `flux[Xi]` and field EMFs immediately
  // after the LO Riemann solve, before CombineFluxesDir overwrites them with
  // the HO+LO blend.  Pure diagnostic (HDF5 dump only); the blender never
  // reads from these.
  AthenaArray<Real> lo_x1flux_, lo_e3_x1f_, lo_e2_x1f_;
  AthenaArray<Real> lo_x2flux_, lo_e1_x2f_, lo_e3_x2f_;
  AthenaArray<Real> lo_x3flux_, lo_e2_x3f_, lo_e1_x3f_;
#endif
 private:
  AthenaArray<Real> entropy_curr_, entropy_prev1_, entropy_prev2_, entropy_prev3_;
  Real efl_cmax_{1.0};
  Real efl_cE_{1.0};
  Real efl_time_curr_{0.0};
  Real efl_time_prev1_{0.0};
  Real efl_time_prev2_{0.0};
  Real efl_time_prev3_{0.0};
  int efl_buffer_cycles_{0};

  void CalculateEntropy(const AthenaArray<Real> &prim, AthenaArray<Real> &ent);
  void CalculateEntropyResidual(const AthenaArray<Real> &prim,
                                const AthenaArray<Real> &ent);
  // Build the face-θ limiter array from cell-centered entropy_residual.  Dir
  // selects the face-normal direction (1=x1, 2=x2, 3=x3) — picks which
  // efl_limiter_xN array to write to, sets up direction-appropriate loop
  // bounds, and indexes the stencil along the face-normal axis.
  // Explicitly instantiated in hydro.cpp for Dir = 1, 2, 3.
  template <int Dir>
  void BuildFaceLimiter();
#endif

  TimeStepFunc UserTimeStep_;

  void AddDiffusionFluxes();
};
#endif // HYDRO_HYDRO_HPP_
