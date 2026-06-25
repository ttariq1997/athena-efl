//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file calculate_fluxes.cpp
//! \brief Calculate hydro/MHD fluxes

// C headers

// C++ headers
#include <algorithm>   // min,max
#include <cmath>       // isfinite
#include <cstring>     // strcmp
#include <limits>      // numeric_limits

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"   // reapply floors to face-centered reconstructed states
#include "../field/field.hpp"
#include "../field/field_diffusion/field_diffusion.hpp"
#include "../gravity/gravity.hpp"
#include "../reconstruct/reconstruction.hpp"
#include "../scalars/scalars.hpp"
#include "hydro.hpp"
#include "hydro_diffusion/hydro_diffusion.hpp"

// OpenMP header
#ifdef OPENMP_PARALLEL
#include <omp.h>
#endif

#if EFL_ENABLED && !MAGNETIC_FIELDS_ENABLED && RELATIVISTIC_DYNAMICS
namespace {
// SR-hydro EFL flux blender (no EMF, no CT weight -- pure hydro fluxes only).
// blend rule:
//   flux(:, k, j, i) = theta * flux_HO + (1 - theta) * flux_LO,
// where theta = efl_limiter_x{ivx}(k, j, i) is in [0, 1].  When the HO flux is
// non-finite (eigensystem failure marked via NaN by the HO driver), the LO
// flux is kept verbatim (no blend), exactly as in the SRMHD pipeline.
void CombineFluxesDirHydro(Hydro *ph,
                           const int k, const int j, const int il, const int iu,
                           AthenaArray<Real> &flux,
                           const AthenaArray<Real> &flux_ho,
                           const AthenaArray<Real> &limiter,
                           const AthenaArray<Real> &w,
                           const int dir) {
  const Real theta_tol = 1.0e-12;

  // Hybrid HO/LO scheme — fully separate from the EFL machinery below.
  // When hybrid_enable_ is true, the EFL entropy theta is bypassed.  Per face:
  //   theta = 1 (pure HO) if min(rho_L, rho_R) >= hybrid_rho_cutoff_,
  //   theta = 0 (pure LO) otherwise.
  // The non-finite HO flux fallback is preserved (still keeps LO).
  if (ph->hybrid_enable_) {
    const Real rho_cut = ph->hybrid_rho_cutoff_;
    for (int i = il; i <= iu; ++i) {
      Real rho_low;
      switch (dir) {
        case 1:  rho_low = std::min(w(IDN, k, j, i-1), w(IDN, k, j, i)); break;
        case 2:  rho_low = std::min(w(IDN, k, j-1, i), w(IDN, k, j, i)); break;
        case 3:  rho_low = std::min(w(IDN, k-1, j, i), w(IDN, k, j, i)); break;
        default: rho_low = std::numeric_limits<Real>::max();             break;
      }
      bool use_lo = (rho_low < rho_cut);
      if (!use_lo) {
        for (int n = 0; n < NHYDRO; ++n) {
          if (!std::isfinite(flux_ho(n, k, j, i))) { use_lo = true; break; }
        }
      }
      if (!use_lo) {
        for (int n = 0; n < NHYDRO; ++n) {
          flux(n, k, j, i) = flux_ho(n, k, j, i);
        }
      }
    }
    return;
  }

  // Standard EFL path.  Atm protection comes from CalculateEntropyResidual
  // forcing residual = cmax at body cells whose 7-cell mask touches atm.
  for (int i = il; i <= iu; ++i) {
    Real theta = limiter(k, j, i);
    theta = std::max((Real)0.0, std::min((Real)1.0, theta));

    bool use_lo = false;
    for (int n = 0; n < NHYDRO; ++n) {
      use_lo = use_lo || !std::isfinite(flux_ho(n, k, j, i));
    }
    if (!use_lo) {
#if EFL_DEBUG
      if (theta > theta_tol) {
        if (theta >= 1.0 - theta_tol) ++ph->ho_pure_ho_;
        else                          ++ph->ho_hybridized_;
      }
#endif
      for (int n = 0; n < NHYDRO; ++n) {
        flux(n, k, j, i) = theta * flux_ho(n, k, j, i)
                         + (1.0 - theta) * flux(n, k, j, i);
      }
    }
  }
}

}  // namespace
#endif

#if EFL_ENABLED && MAGNETIC_FIELDS_ENABLED && RELATIVISTIC_DYNAMICS
namespace {

void CombineFluxesDirMHD(Hydro *ph,
                        const int k, const int j, const int il, const int iu,
                        AthenaArray<Real> &flux,
                        AthenaArray<Real> &emf_a,
                        AthenaArray<Real> &emf_b,
                        AthenaArray<Real> &wct,
                        const AthenaArray<Real> &flux_ho,
                        const AthenaArray<Real> &emf_a_ho,
                        const AthenaArray<Real> &emf_b_ho,
                        const AthenaArray<Real> &limiter,
                        const AthenaArray<Real> &wl,
                        const AthenaArray<Real> &wr,
                        const AthenaArray<Real> &dxw,
                        const Real dt) {
  const Real theta_tol = 1.0e-12;

  // Atm protection comes from CalculateEntropyResidual forcing residual = cmax
  // at body cells whose 7-cell mask stencil touches atm (mask > 0.4).
  for (int i = il; i <= iu; ++i) {
    Real theta = limiter(k, j, i);
    theta = std::max((Real)0.0, std::min((Real)1.0, theta));

    bool use_lo = !std::isfinite(emf_a_ho(k, j, i))
               || !std::isfinite(emf_b_ho(k, j, i));
    for (int n = 0; n < NHYDRO; ++n) {
      use_lo = use_lo || !std::isfinite(flux_ho(n, k, j, i));
    }

    if (!use_lo) {
#if EFL_DEBUG
      if (theta > theta_tol) {
        if (theta >= 1.0 - theta_tol) ++ph->ho_pure_ho_;
        else                          ++ph->ho_hybridized_;
      }
#endif
      // Skip blend at theta <= tol — preserves bit-exact LO behavior at θ=0.
      if (theta > theta_tol) {
        for (int n = 0; n < NHYDRO; ++n) {
          flux(n, k, j, i) = theta * flux_ho(n, k, j, i)
                           + (1.0 - theta) * flux(n, k, j, i);
        }
        emf_a(k, j, i) = theta * emf_a_ho(k, j, i) + (1.0 - theta) * emf_a(k, j, i);
        emf_b(k, j, i) = theta * emf_b_ho(k, j, i) + (1.0 - theta) * emf_b(k, j, i);
      }
    }

    wct(k, j, i) = ph->GetWeightForCT(flux(IDN, k, j, i),
                                      wl(IDN, i), wr(IDN, i), dxw(i), dt);
  }
}

}  // namespace
#endif

//----------------------------------------------------------------------------------------
//! \fn  void Hydro::CalculateFluxes
//! \brief Calculate Hydrodynamic Fluxes using the Riemann solver

void Hydro::CalculateFluxes(AthenaArray<Real> &w, FaceField &b,
                            AthenaArray<Real> &bcc, const int order) {
  MeshBlock *pmb = pmy_block;
  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;
  int il, iu, jl, ju, kl, ku;

  // b,bcc are passed as fn parameters becausse clients may want to pass different bcc1,
  // b1, b2, etc., but the remaining members of the Field class are accessed directly via
  // pointers because they are unique. NOTE: b, bcc are nullptrs if no MHD.
#if MAGNETIC_FIELDS_ENABLED
  // used only to pass to (up-to) 2x RiemannSolver() calls per dimension:
  // x1:
  AthenaArray<Real> &b1 = b.x1f, &w_x1f = pmb->pfield->wght.x1f,
                  &e3x1 = pmb->pfield->e3_x1f, &e2x1 = pmb->pfield->e2_x1f;
  // x2:
  AthenaArray<Real> &b2 = b.x2f, &w_x2f = pmb->pfield->wght.x2f,
                  &e1x2 = pmb->pfield->e1_x2f, &e3x2 = pmb->pfield->e3_x2f;
  // x3:
  AthenaArray<Real> &b3 = b.x3f, &w_x3f = pmb->pfield->wght.x3f,
                  &e1x3 = pmb->pfield->e1_x3f, &e2x3 = pmb->pfield->e2_x3f;
#endif
  AthenaArray<Real> &flux_fc = scr1_nkji_;
  AthenaArray<Real> &laplacian_all_fc = scr2_nkji_;

#if MAGNETIC_FIELDS_ENABLED && RELATIVISTIC_DYNAMICS
#if RSOLVER_IS_RUSANOV
  if (std::strcmp(RIEMANN_SOLVER, "rusanov") == 0) {
    RusanovFlux(w, u, b, bcc,
                flux[X1DIR], e3x1, e2x1, w_x1f,
                flux[X2DIR], e1x2, e3x2, w_x2f,
                flux[X3DIR], e2x3, e1x3, w_x3f);
    if (!STS_ENABLED) {
      AddDiffusionFluxes();
    }
    return;
  }
#endif
#endif

#if !MAGNETIC_FIELDS_ENABLED && RELATIVISTIC_DYNAMICS
#if RSOLVER_IS_RUSANOV
  // SR hydro HO Rusanov: 5x5 Donat 1998 / Font (2008) eigensystem.
  // Driven from CalculateFluxes (no per-row line-solver) to mirror the SRMHD
  // dispatch above.  Phase 1: HO standalone (no EFL blending).
  if (std::strcmp(RIEMANN_SOLVER, "rusanov") == 0) {
    if (pmb->pmy_mesh->f2 || pmb->pmy_mesh->f3) {
      RusanovFlux(w, flux[X1DIR], flux[X2DIR], flux[X3DIR]);
    } else {
      RusanovFlux(w, flux[X1DIR]);
    }
    if (!STS_ENABLED) {
      AddDiffusionFluxes();
    }
    return;
  }
#endif
#endif

#if EFL_ENABLED && !MAGNETIC_FIELDS_ENABLED && RELATIVISTIC_DYNAMICS
  // SR-hydro EFL: precompute HO fluxes upfront, then blend per face inside the
  // directional loop below via CombineFluxesDirHydro.  do_efl is referenced
  // from those loops, so its scope spans the rest of CalculateFluxes.
  const bool do_efl = efl_enabled;
  if (do_efl) {
    if (pmb->pmy_mesh->f2 || pmb->pmy_mesh->f3) {
      RusanovFlux(w, ho_x1flux_, ho_x2flux_, ho_x3flux_);
    } else {
      RusanovFlux(w, ho_x1flux_);
    }
  }
#endif

#if EFL_ENABLED && MAGNETIC_FIELDS_ENABLED && RELATIVISTIC_DYNAMICS
  const bool do_efl = efl_enabled;
  if (do_efl) {
    if (pmb->pmy_mesh->f2 || pmb->pmy_mesh->f3) {
      RusanovFlux(w, u, b, bcc,
                  ho_x1flux_, ho_e3_x1f_, ho_e2_x1f_, w_x1f,
                  ho_x2flux_, ho_e1_x2f_, ho_e3_x2f_, w_x2f,
                  ho_x3flux_, ho_e2_x3f_, ho_e1_x3f_, w_x3f);
    } else {
      RusanovFlux(w, u, b, bcc, ho_x1flux_, ho_e3_x1f_, ho_e2_x1f_);
    }
  }
#endif

  //--------------------------------------------------------------------------------------
  // i-direction

  AthenaArray<Real> &x1flux = flux[X1DIR];
  // set the loop limits
  jl = js, ju = je, kl = ks, ku = ke;
  if (MAGNETIC_FIELDS_ENABLED || order == 4) {
    if (pmb->block_size.nx2 > 1) {
      if (pmb->block_size.nx3 == 1) // 2D
        jl = js-1, ju = je+1, kl = ks, ku = ke;
      else // 3D
        jl = js-1, ju = je+1, kl = ks-1, ku = ke+1;
    }
  }

  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      // reconstruct L/R states
      if (order == 1) {
        pmb->precon->DonorCellX1(k, j, is-1, ie+1, w, bcc, wl_, wr_);
      } else if (order == 2) {
        pmb->precon->PiecewiseLinearX1(k, j, is-1, ie+1, w, bcc, wl_, wr_);
      } else {
        pmb->precon->PiecewiseParabolicX1(k, j, is-1, ie+1, w, bcc, wl_, wr_);
      }

      pmb->pcoord->CenterWidth1(k, j, is, ie+1, dxw_);
#if !MAGNETIC_FIELDS_ENABLED  // Hydro:
      RiemannSolver(k, j, is, ie+1, IVX, wl_, wr_, x1flux, dxw_);
#if EFL_ENABLED && RELATIVISTIC_DYNAMICS
      if (do_efl) {
        CombineFluxesDirHydro(this, k, j, is, ie+1, x1flux, ho_x1flux_,
                              efl_limiter_x1, w, 1);
      }
#endif
#else  // MHD:
      // x1flux(IBY) = (v1*b2 - v2*b1) = -EMFZ
      // x1flux(IBZ) = (v1*b3 - v3*b1) =  EMFY
      RiemannSolver(k, j, is, ie+1, IVX, b1, wl_, wr_, x1flux, e3x1, e2x1, w_x1f, dxw_);
#if EFL_ENABLED && MAGNETIC_FIELDS_ENABLED && RELATIVISTIC_DYNAMICS
      if (do_efl) {
#if EFL_DEBUG
        // Diagnostic snapshot of the LO flux + LO EMFs before the blend.
        for (int n = 0; n < NHYDRO; ++n) {
          for (int i = is; i <= ie+1; ++i) {
            lo_x1flux_(n, k, j, i) = x1flux(n, k, j, i);
          }
        }
        for (int i = is; i <= ie+1; ++i) {
          lo_e3_x1f_(k, j, i) = e3x1(k, j, i);
          lo_e2_x1f_(k, j, i) = e2x1(k, j, i);
        }
#endif
        CombineFluxesDirMHD(this, k, j, is, ie+1, x1flux, e3x1, e2x1, w_x1f,
                            ho_x1flux_, ho_e3_x1f_, ho_e2_x1f_,
                            efl_limiter_x1,
                            wl_, wr_, dxw_, pmb->pmy_mesh->dt);
      }
#endif
#endif

      if (order == 4) {
        for (int n=0; n<NWAVE; n++) {
          for (int i=is; i<=ie+1; i++) {
            wl3d_(n,k,j,i) = wl_(n,i);
            wr3d_(n,k,j,i) = wr_(n,i);
          }
        }
      }
    }
  }

  if (order == 4) {
    // TODO(felker): assuming uniform mesh with dx1f=dx2f=dx3f, so this should factor out
    // TODO(felker): also, this may need to be dx1v, since Laplacian is cell-centered
    Real h = pmb->pcoord->dx1f(is);  // pco->dx1f(i); inside loop
    Real C = (h*h)/24.0;

    // construct Laplacian from x1flux
    pmb->pcoord->LaplacianX1All(x1flux, laplacian_all_fc, 0, NHYDRO-1,
                                kl, ku, jl, ju, is, ie+1);

    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
        // Compute Laplacian of primitive Riemann states on x1 faces
        for (int n=0; n<NWAVE; ++n) {
          pmb->pcoord->LaplacianX1(wl3d_, laplacian_l_fc_, n, k, j, is, ie+1);
          pmb->pcoord->LaplacianX1(wr3d_, laplacian_r_fc_, n, k, j, is, ie+1);
#pragma omp simd
          for (int i=is; i<=ie+1; ++i) {
            wl_(n,i) = wl3d_(n,k,j,i) - C*laplacian_l_fc_(i);
            wr_(n,i) = wr3d_(n,k,j,i) - C*laplacian_r_fc_(i);
          }
        }
#pragma omp simd
        for (int i=is; i<=ie+1; ++i) {
          pmb->peos->ApplyPrimitiveFloors(wl_, k, j, i);
          pmb->peos->ApplyPrimitiveFloors(wr_, k, j, i);
        }

        // Compute x1 interface fluxes from face-centered primitive variables
        // TODO(felker): check that e3x1,e2x1 arguments added in late 2017 work here
        pmb->pcoord->CenterWidth1(k, j, is, ie+1, dxw_);
#if !MAGNETIC_FIELDS_ENABLED  // Hydro:
        RiemannSolver(k, j, is, ie+1, IVX, wl_, wr_, flux_fc, dxw_);
#else  // MHD:
        RiemannSolver(k, j, is, ie+1, IVX, b1, wl_, wr_, flux_fc, e3x1, e2x1,
                      w_x1f, dxw_);
#endif
        // Apply Laplacian of second-order accurate face-averaged flux on x1 faces
        for (int n=0; n<NHYDRO; ++n) {
#pragma omp simd
          for (int i=is; i<=ie+1; i++) {
            x1flux(n,k,j,i) = flux_fc(n,k,j,i) + C*laplacian_all_fc(n,k,j,i);
            // TODO(felker): replace this loop-based deep copy with memcpy, or alternative
            if (n == IDN && NSCALARS > 0) {
              pmb->pscalars->mass_flux_fc[X1DIR](k,j,i) = flux_fc(n,k,j,i);
            }
          }
        }
      }
    }
  } // end if (order == 4)
  //------------------------------------------------------------------------------
  // end x1 fourth-order hydro

  //--------------------------------------------------------------------------------------
  // j-direction

  if (pmb->pmy_mesh->f2) {
    AthenaArray<Real> &x2flux = flux[X2DIR];
    // set the loop limits
    il = is-1, iu = ie+1, kl = ks, ku = ke;
    if (MAGNETIC_FIELDS_ENABLED || order == 4) {
      if (pmb->block_size.nx3 == 1) // 2D
        kl = ks, ku = ke;
      else // 3D
        kl = ks-1, ku = ke+1;
    }

    for (int k=kl; k<=ku; ++k) {
      // reconstruct the first row
      if (order == 1) {
        pmb->precon->DonorCellX2(k, js-1, il, iu, w, bcc, wl_, wr_);
      } else if (order == 2) {
        pmb->precon->PiecewiseLinearX2(k, js-1, il, iu, w, bcc, wl_, wr_);
      } else {
        pmb->precon->PiecewiseParabolicX2(k, js-1, il, iu, w, bcc, wl_, wr_);
      }
      for (int j=js; j<=je+1; ++j) {
        // reconstruct L/R states at j
        if (order == 1) {
          pmb->precon->DonorCellX2(k, j, il, iu, w, bcc, wlb_, wr_);
        } else if (order == 2) {
          pmb->precon->PiecewiseLinearX2(k, j, il, iu, w, bcc, wlb_, wr_);
        } else {
          pmb->precon->PiecewiseParabolicX2(k, j, il, iu, w, bcc, wlb_, wr_);
        }

        pmb->pcoord->CenterWidth2(k, j, il, iu, dxw_);
#if !MAGNETIC_FIELDS_ENABLED  // Hydro:
        RiemannSolver(k, j, il, iu, IVY, wl_, wr_, x2flux, dxw_);
#if EFL_ENABLED && RELATIVISTIC_DYNAMICS
        if (do_efl) {
          CombineFluxesDirHydro(this, k, j, il, iu, x2flux, ho_x2flux_,
                                efl_limiter_x2, w, 2);
        }
#endif
#else  // MHD:
        // flx(IBY) = (v2*b3 - v3*b2) = -EMFX
        // flx(IBZ) = (v2*b1 - v1*b2) =  EMFZ
        RiemannSolver(k, j, il, iu, IVY, b2, wl_, wr_, x2flux, e1x2, e3x2, w_x2f, dxw_);
#if EFL_ENABLED && MAGNETIC_FIELDS_ENABLED && RELATIVISTIC_DYNAMICS
        if (do_efl) {
#if EFL_DEBUG
          for (int n = 0; n < NHYDRO; ++n) {
            for (int i = il; i <= iu; ++i) {
              lo_x2flux_(n, k, j, i) = x2flux(n, k, j, i);
            }
          }
          for (int i = il; i <= iu; ++i) {
            lo_e1_x2f_(k, j, i) = e1x2(k, j, i);
            lo_e3_x2f_(k, j, i) = e3x2(k, j, i);
          }
#endif
          CombineFluxesDirMHD(this, k, j, il, iu, x2flux, e1x2, e3x2, w_x2f,
                              ho_x2flux_, ho_e1_x2f_, ho_e3_x2f_,
                              efl_limiter_x2,
                              wl_, wr_, dxw_, pmb->pmy_mesh->dt);
        }
#endif
#endif

        if (order == 4) {
          for (int n=0; n<NWAVE; n++) {
            for (int i=il; i<=iu; i++) {
              wl3d_(n,k,j,i) = wl_(n,i);
              wr3d_(n,k,j,i) = wr_(n,i);
            }
          }
        }

        // swap the arrays for the next step
        wl_.SwapAthenaArray(wlb_);
      }
    }
    if (order == 4) {
      // TODO(felker): assuming uniform mesh with dx1f=dx2f=dx3f, so factor this out
      // TODO(felker): also, this may need to be dx2v, since Laplacian is cell-centered
      Real h = pmb->pcoord->dx2f(js);  // pco->dx2f(j); inside loop
      Real C = (h*h)/24.0;

      // construct Laplacian from x2flux
      pmb->pcoord->LaplacianX2All(x2flux, laplacian_all_fc, 0, NHYDRO-1,
                                  kl, ku, js, je+1, il, iu);

      // Approximate x2 face-centered primitive Riemann states
      for (int k=kl; k<=ku; ++k) {
        for (int j=js; j<=je+1; ++j) {
          // Compute Laplacian of primitive Riemann states on x2 faces
          for (int n=0; n<NWAVE; ++n) {
            pmb->pcoord->LaplacianX2(wl3d_, laplacian_l_fc_, n, k, j, il, iu);
            pmb->pcoord->LaplacianX2(wr3d_, laplacian_r_fc_, n, k, j, il, iu);
#pragma omp simd
            for (int i=il; i<=iu; ++i) {
              wl_(n,i) = wl3d_(n,k,j,i) - C*laplacian_l_fc_(i);
              wr_(n,i) = wr3d_(n,k,j,i) - C*laplacian_r_fc_(i);
            }
          }
#pragma omp simd
          for (int i=il; i<=iu; ++i) {
            pmb->peos->ApplyPrimitiveFloors(wl_, k, j, i);
            pmb->peos->ApplyPrimitiveFloors(wr_, k, j, i);
          }

          // Compute x2 interface fluxes from face-centered primitive variables
          // TODO(felker): check that e1x2,e3x2 arguments added in late 2017 work here
          pmb->pcoord->CenterWidth2(k, j, il, iu, dxw_);
#if !MAGNETIC_FIELDS_ENABLED  // Hydro:
          RiemannSolver(k, j, il, iu, IVY, wl_, wr_, flux_fc, dxw_);
#else  // MHD:
          RiemannSolver(k, j, il, iu, IVY, b2, wl_, wr_, flux_fc, e1x2, e3x2,
                        w_x2f, dxw_);
#endif

          // Apply Laplacian of second-order accurate face-averaged flux on x1 faces
          for (int n=0; n<NHYDRO; ++n) {
#pragma omp simd
            for (int i=il; i<=iu; i++) {
              x2flux(n,k,j,i) = flux_fc(n,k,j,i) + C*laplacian_all_fc(n,k,j,i);
              if (n == IDN && NSCALARS > 0) {
                pmb->pscalars->mass_flux_fc[X2DIR](k,j,i) = flux_fc(n,k,j,i);
              }
            }
          }
        }
      }
    } // end if (order == 4)
  }

  //--------------------------------------------------------------------------------------
  // k-direction

  if (pmb->pmy_mesh->f3) {
    AthenaArray<Real> &x3flux = flux[X3DIR];
    // set the loop limits
    il = is, iu = ie, jl = js, ju = je;
    if (MAGNETIC_FIELDS_ENABLED || order == 4) {
      il = is-1, iu = ie+1, jl = js-1, ju = je+1;
    }

    for (int j=jl; j<=ju; ++j) { // this loop ordering is intentional
      // reconstruct the first row
      if (order == 1) {
        pmb->precon->DonorCellX3(ks-1, j, il, iu, w, bcc, wl_, wr_);
      } else if (order == 2) {
        pmb->precon->PiecewiseLinearX3(ks-1, j, il, iu, w, bcc, wl_, wr_);
      } else {
        pmb->precon->PiecewiseParabolicX3(ks-1, j, il, iu, w, bcc, wl_, wr_);
      }
      for (int k=ks; k<=ke+1; ++k) {
        // reconstruct L/R states at k
        if (order == 1) {
          pmb->precon->DonorCellX3(k, j, il, iu, w, bcc, wlb_, wr_);
        } else if (order == 2) {
          pmb->precon->PiecewiseLinearX3(k, j, il, iu, w, bcc, wlb_, wr_);
        } else {
          pmb->precon->PiecewiseParabolicX3(k, j, il, iu, w, bcc, wlb_, wr_);
        }

        pmb->pcoord->CenterWidth3(k, j, il, iu, dxw_);
#if !MAGNETIC_FIELDS_ENABLED  // Hydro:
        RiemannSolver(k, j, il, iu, IVZ, wl_, wr_, x3flux, dxw_);
#if EFL_ENABLED && RELATIVISTIC_DYNAMICS
        if (do_efl) {
          CombineFluxesDirHydro(this, k, j, il, iu, x3flux, ho_x3flux_,
                                efl_limiter_x3, w, 3);
        }
#endif
#else  // MHD:
        // flx(IBY) = (v3*b1 - v1*b3) = -EMFY
        // flx(IBZ) = (v3*b2 - v2*b3) =  EMFX
        RiemannSolver(k, j, il, iu, IVZ, b3, wl_, wr_, x3flux, e2x3, e1x3, w_x3f, dxw_);
#if EFL_ENABLED && MAGNETIC_FIELDS_ENABLED && RELATIVISTIC_DYNAMICS
        if (do_efl) {
#if EFL_DEBUG
          for (int n = 0; n < NHYDRO; ++n) {
            for (int i = il; i <= iu; ++i) {
              lo_x3flux_(n, k, j, i) = x3flux(n, k, j, i);
            }
          }
          for (int i = il; i <= iu; ++i) {
            lo_e2_x3f_(k, j, i) = e2x3(k, j, i);
            lo_e1_x3f_(k, j, i) = e1x3(k, j, i);
          }
#endif
          CombineFluxesDirMHD(this, k, j, il, iu, x3flux, e2x3, e1x3, w_x3f,
                              ho_x3flux_, ho_e2_x3f_, ho_e1_x3f_,
                              efl_limiter_x3,
                              wl_, wr_, dxw_, pmb->pmy_mesh->dt);
        }
#endif
#endif
        if (order == 4) {
          for (int n=0; n<NWAVE; n++) {
            for (int i=il; i<=iu; i++) {
              wl3d_(n,k,j,i) = wl_(n,i);
              wr3d_(n,k,j,i) = wr_(n,i);
            }
          }
        }

        // swap the arrays for the next step
        wl_.SwapAthenaArray(wlb_);
      }
    }
    if (order == 4) {
      // TODO(felker): assuming uniform mesh with dx1f=dx2f=dx3f, so factor this out
      // TODO(felker): also, this may need to be dx3v, since Laplacian is cell-centered
      Real h = pmb->pcoord->dx3f(ks);  // pco->dx3f(j); inside loop
      Real C = (h*h)/24.0;

      // construct Laplacian from x3flux
      pmb->pcoord->LaplacianX3All(x3flux, laplacian_all_fc, 0, NHYDRO-1,
                                  ks, ke+1, jl, ju, il, iu);

      // Approximate x3 face-centered primitive Riemann states
      for (int k=ks; k<=ke+1; ++k) {
        for (int j=jl; j<=ju; ++j) {
          // Compute Laplacian of primitive Riemann states on x3 faces
          for (int n=0; n<NWAVE; ++n) {
            pmb->pcoord->LaplacianX3(wl3d_, laplacian_l_fc_, n, k, j, il, iu);
            pmb->pcoord->LaplacianX3(wr3d_, laplacian_r_fc_, n, k, j, il, iu);
#pragma omp simd
            for (int i=il; i<=iu; ++i) {
              wl_(n,i) = wl3d_(n,k,j,i) - C*laplacian_l_fc_(i);
              wr_(n,i) = wr3d_(n,k,j,i) - C*laplacian_r_fc_(i);
            }
          }
#pragma omp simd
          for (int i=il; i<=iu; ++i) {
            pmb->peos->ApplyPrimitiveFloors(wl_, k, j, i);
            pmb->peos->ApplyPrimitiveFloors(wr_, k, j, i);
          }

          // Compute x3 interface fluxes from face-centered primitive variables
          // TODO(felker): check that e2x3,e1x3 arguments added in late 2017 work here
          pmb->pcoord->CenterWidth3(k, j, il, iu, dxw_);
#if !MAGNETIC_FIELDS_ENABLED  // Hydro:
          RiemannSolver(k, j, il, iu, IVZ, wl_, wr_, flux_fc, dxw_);
#else  // MHD:
          RiemannSolver(k, j, il, iu, IVZ, b3, wl_, wr_, flux_fc, e2x3, e1x3,
                        w_x3f, dxw_);
#endif
          // Apply Laplacian of second-order accurate face-averaged flux on x3 faces
          for (int n=0; n<NHYDRO; ++n) {
#pragma omp simd
            for (int i=il; i<=iu; i++) {
              x3flux(n,k,j,i) = flux_fc(n,k,j,i) + C*laplacian_all_fc(n,k,j,i);
              if (n == IDN && NSCALARS > 0) {
                pmb->pscalars->mass_flux_fc[X3DIR](k,j,i) = flux_fc(n,k,j,i);
              }
            }
          }
        }
      }
    } // end if (order == 4)
  }

  if (!STS_ENABLED)
    AddDiffusionFluxes();

  return;
}

//----------------------------------------------------------------------------------------
//! \fn  void Hydro::CalculateFluxes_STS
//! \brief Calculate Hydrodynamic Diffusion Fluxes for STS

void Hydro::CalculateFluxes_STS() {
  AddDiffusionFluxes();
}

void Hydro::AddDiffusionFluxes() {
  Field *pf = pmy_block->pfield;
  // add diffusion fluxes
  if (hdif.hydro_diffusion_defined) {
    if (hdif.nu_iso > 0.0 || hdif.nu_aniso > 0.0)
      hdif.AddDiffusionFlux(hdif.visflx,flux);
    if (NON_BAROTROPIC_EOS) {
      if (hdif.kappa_iso > 0.0 || hdif.kappa_aniso > 0.0)
        hdif.AddDiffusionEnergyFlux(hdif.cndflx,flux);
    }
  }
  if (MAGNETIC_FIELDS_ENABLED && NON_BAROTROPIC_EOS) {
    if (pf->fdif.field_diffusion_defined)
      pf->fdif.AddPoyntingFlux(pf->fdif.pflux);
  }
  return;
}
