//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file srmhd_cpaw.cpp
//! \brief Smooth large-amplitude circularly polarized Alfven wave for SRMHD.
//!
//! This implementation follows Del Zanna et al. (2007, ECHO), Sec. 4.1, equations
//! (82)-(85), as quoted in Martí & Müller (2015). The test is implemented for the
//! 1-D and 2-D cases used in the paper:
//! - 1-D: propagation along x over a periodic domain [0,2*pi]
//! - 2-D: propagation along the diagonal of a periodic [0,2*pi]^2 domain
//========================================================================================

#include <cmath>
#include <cstdio>
#include <sstream>
#include <string>

#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"

#if !RELATIVISTIC_DYNAMICS
#error "This problem generator must be used with relativity"
#endif

#if !MAGNETIC_FIELDS_ENABLED
#error "This problem generator requires magnetic fields"
#endif

#if GENERAL_RELATIVITY
#error "This problem generator is for SRMHD, not GRMHD"
#endif

namespace {

Real rho0, pgas0, b0, eta_amp;
Real gamma_adi, gamma_adi_red;
Real v_alfven, v_alfven_sq;
Real lambda_wave, k_wave, t_final;
AthenaArray<Real> volume;
AthenaArray<Real> bcc_scratch;   // scratch for computing cell-centered B at IC
AthenaArray<Real> initial;       // stored IC: (nblocks, NHYDRO+NFIELD, nx3, nx2, nx1)
bool compute_error;
bool is_2d_case;

// 4-point Gauss-Legendre quadrature on [-1, 1] (exact for polys up to degree 7)
constexpr int kNQuad = 4;
const Real gl_nodes[kNQuad] = {-0.8611363115940526, -0.3399810435848563,
                                0.3399810435848563,  0.8611363115940526};
const Real gl_weights[kNQuad] = {0.3478548451374538, 0.6521451548625461,
                                  0.6521451548625461, 0.3478548451374538};

struct PrimitiveState {
  Real rho;
  Real pgas;
  Real v1;
  Real v2;
  Real v3;
  Real b1;
  Real b2;
  Real b3;
};

struct CellAverage {
  Real rho;
  Real pgas;
  Real u1;   // 4-velocity components (gamma * v^i)
  Real u2;
  Real u3;
};

void GetWaveCoordinates(const Real x1, const Real x2, Real *pxi, Real *peta) {
  if (!is_2d_case) {
    *pxi = x1;
    *peta = x2;
    return;
  }
  const Real inv_sqrt2 = 1.0 / std::sqrt(2.0);
  *pxi = (x1 + x2) * inv_sqrt2;
  *peta = (-x1 + x2) * inv_sqrt2;
}

void RotateVectorToGrid(const Real v_xi, const Real v_eta, const Real v_z,
                        Real *pv1, Real *pv2, Real *pv3) {
  if (!is_2d_case) {
    *pv1 = v_xi;
    *pv2 = v_eta;
    *pv3 = v_z;
    return;
  }
  const Real inv_sqrt2 = 1.0 / std::sqrt(2.0);
  *pv1 = (v_xi - v_eta) * inv_sqrt2;
  *pv2 = (v_xi + v_eta) * inv_sqrt2;
  *pv3 = v_z;
}

void ExactStateAt(const Real x1, const Real x2, const Real time, PrimitiveState *state) {
  Real xi, eta_coord;
  GetWaveCoordinates(x1, x2, &xi, &eta_coord);
  const Real phase = k_wave * (xi - v_alfven * time);
  const Real c = std::cos(phase);
  const Real s = std::sin(phase);

  // Paper equations (82) and (83)
  const Real by_loc = eta_amp * b0 * c;
  const Real bz_loc = eta_amp * b0 * s;
  const Real vy_loc = -v_alfven * by_loc / b0;
  const Real vz_loc = -v_alfven * bz_loc / b0;

  if (!is_2d_case) {
    state->b1 = b0;
    state->b2 = by_loc;
    state->b3 = bz_loc;
    state->v1 = 0.0;
    state->v2 = vy_loc;
    state->v3 = vz_loc;
  } else {
    RotateVectorToGrid(b0, by_loc, bz_loc, &(state->b1), &(state->b2), &(state->b3));
    RotateVectorToGrid(0.0, vy_loc, vz_loc, &(state->v1), &(state->v2), &(state->v3));
  }
  state->rho = rho0;
  state->pgas = pgas0;
}

// Cell-averaged (rho, p, u^i) via tensor-product Gauss-Legendre quadrature.
// 1D case: quadrature over x1 only; 2D case: over (x1, x2).
void CellAveragedState(const Real x1c, const Real dx1, const Real x2c, const Real dx2,
                       const Real time, CellAverage *avg) {
  avg->rho = 0.0;
  avg->pgas = 0.0;
  avg->u1 = 0.0;
  avg->u2 = 0.0;
  avg->u3 = 0.0;
  const int nqy = is_2d_case ? kNQuad : 1;
  for (int qy = 0; qy < nqy; ++qy) {
    const Real x2q = is_2d_case ? x2c + 0.5 * dx2 * gl_nodes[qy] : x2c;
    const Real wy = is_2d_case ? 0.5 * gl_weights[qy] : 1.0;
    for (int qx = 0; qx < kNQuad; ++qx) {
      const Real x1q = x1c + 0.5 * dx1 * gl_nodes[qx];
      const Real wq = 0.5 * gl_weights[qx] * wy;
      PrimitiveState s;
      ExactStateAt(x1q, x2q, time, &s);
      const Real v_sq = SQR(s.v1) + SQR(s.v2) + SQR(s.v3);
      const Real lor = 1.0 / std::sqrt(1.0 - v_sq);
      avg->rho += wq * s.rho;
      avg->pgas += wq * s.pgas;
      avg->u1 += wq * lor * s.v1;
      avg->u2 += wq * lor * s.v2;
      avg->u3 += wq * lor * s.v3;
    }
  }
}

Real A1(const Real x1, const Real x2) {
  if (!is_2d_case) return 0.0;
  Real xi, eta_coord;
  GetWaveCoordinates(x1, x2, &xi, &eta_coord);
  const Real a_eta = -(eta_amp * b0 / k_wave) * std::cos(k_wave * xi);
  return -a_eta / std::sqrt(2.0);
}

Real A2(const Real x1, const Real x2) {
  Real xi, eta_coord;
  GetWaveCoordinates(x1, x2, &xi, &eta_coord);
  const Real a_eta = -(eta_amp * b0 / k_wave) * std::cos(k_wave * xi);
  if (!is_2d_case) return a_eta;
  return a_eta / std::sqrt(2.0);
}

Real A3(const Real x1, const Real x2) {
  Real xi, eta_coord;
  GetWaveCoordinates(x1, x2, &xi, &eta_coord);
  return -(eta_amp * b0 / k_wave) * std::sin(k_wave * xi) + b0 * eta_coord;
}

}  // namespace

void Mesh::InitUserMeshData(ParameterInput *pin) {
  rho0 = pin->GetOrAddReal("problem", "rho", 1.0);
  pgas0 = pin->GetOrAddReal("problem", "pres", 1.0);
  b0 = pin->GetOrAddReal("problem", "b0", 1.0);
  eta_amp = pin->GetOrAddReal("problem", "eta", 1.0);
  compute_error = pin->GetOrAddBoolean("problem", "compute_error", true);

  gamma_adi = pin->GetReal("hydro", "gamma");
  gamma_adi_red = gamma_adi / (gamma_adi - 1.0);

  if (mesh_size.nx2 > 1 && mesh_size.nx3 > 1) {
    std::stringstream msg;
    msg << "### FATAL ERROR in srmhd_cpaw.cpp InitUserMeshData" << std::endl
        << "This ECHO-style pgen is implemented only for the 1-D and 2-D cases."
        << std::endl;
    ATHENA_ERROR(msg);
  }

  if (mesh_size.nx2 == 1) {
    is_2d_case = false;
    lambda_wave = TWO_PI;
    t_final = 0.0;
  } else {
    is_2d_case = true;
    lambda_wave = PI * std::sqrt(2.0);
    t_final = 0.0;
  }
  k_wave = TWO_PI / lambda_wave;

  // Paper equation (85), choosing the smaller physical branch with v_A^2 < 1.
  const Real rhoh = rho0 * (1.0 + gamma_adi_red * pgas0 / rho0);
  const Real denom = rhoh + SQR(b0) * (1.0 + SQR(eta_amp));
  const Real root_arg = 1.0 - SQR(2.0 * eta_amp * SQR(b0) / denom);
  if (root_arg < -1.0e-12) {
    std::stringstream msg;
    msg << "### FATAL ERROR in srmhd_cpaw.cpp InitUserMeshData" << std::endl
        << "The paper wave-speed formula has no real solution for these parameters."
        << std::endl;
    ATHENA_ERROR(msg);
  }
  const Real root = std::sqrt(std::max(root_arg, 0.0));
  v_alfven_sq = (SQR(b0) / denom) / (0.5 * (1.0 + root));
  v_alfven = std::sqrt(v_alfven_sq);

  if (mesh_size.nx2 == 1) {
    t_final = lambda_wave / v_alfven;  // one period, paper 1-D setup
  } else {
    t_final = lambda_wave / v_alfven;  // T/2 for the paper 2-D setup
  }

  volume.NewAthenaArray(mesh_size.nx1 + 2 * NGHOST);
}

void Mesh::UserWorkAfterLoop(ParameterInput *pin) {
  if (!compute_error) return;

  // Multi-component L1 norm against the stored initial condition, following
  // Beckwith & Stone (2011) Section 4.2 eq. (24):
  //   delta U^n = (1/N^D) * sum | U^n_{i,j,k} - U^0_{i,j,k} |
  // where U includes all NHYDRO conserved variables and NFIELD cell-centered
  // magnetic field components. For CPAW at tlim = one Alfven period, the exact
  // solution returns to the IC, so comparing to U^0 IS the true scheme error.
  Real errors[NHYDRO + NFIELD + 1];
  for (int n = 0; n < NHYDRO + NFIELD + 1; ++n) errors[n] = 0.0;

  for (int b = 0; b < nblocal; ++b) {
    MeshBlock *pmb = my_blocks(b);
    // Ensure bcc is up-to-date at tlim for the B comparison
    pmb->pfield->CalculateCellCenteredField(
        pmb->pfield->b, pmb->pfield->bcc, pmb->pcoord,
        pmb->is, pmb->ie, pmb->js, pmb->je, pmb->ks, pmb->ke);
    for (int k = pmb->ks; k <= pmb->ke; ++k) {
      for (int j = pmb->js; j <= pmb->je; ++j) {
        pmb->pcoord->CellVolume(k, j, pmb->is, pmb->ie, volume);
        for (int i = pmb->is; i <= pmb->ie; ++i) {
          // Compare PRIMITIVES in 3-velocity space: (rho, v1, v2, v3, p).
          // v^i = util^i / W with W = sqrt(1 + util^2). This isolates genuine
          // scheme dissipation from the nonlinear v=util/W CPAW floor.
          const Real ux = pmb->phydro->w(IVX, k, j, i);
          const Real uy = pmb->phydro->w(IVY, k, j, i);
          const Real uz = pmb->phydro->w(IVZ, k, j, i);
          const Real W  = std::sqrt(1.0 + SQR(ux) + SQR(uy) + SQR(uz));
          const Real v_now[NHYDRO] = {
              pmb->phydro->w(IDN, k, j, i),
              ux / W,
              uy / W,
              uz / W,
              pmb->phydro->w(IPR, k, j, i)
          };
          for (int n = 0; n < NHYDRO; ++n) {
            errors[n] += std::abs(v_now[n]
                                   - initial(pmb->lid, n, k, j, i))
                         * volume(i);
          }
          for (int n = IB1; n <= IB3; ++n) {
            errors[NHYDRO + n] += std::abs(pmb->pfield->bcc(n, k, j, i)
                                             - initial(pmb->lid, NHYDRO + n, k, j, i))
                                  * volume(i);
          }
          errors[NHYDRO + NFIELD] += volume(i);  // total volume
        }
      }
    }
  }

#ifdef MPI_PARALLEL
  if (Globals::my_rank == 0) {
    MPI_Reduce(MPI_IN_PLACE, errors, NHYDRO + NFIELD + 1,
               MPI_ATHENA_REAL, MPI_SUM, 0, MPI_COMM_WORLD);
  } else {
    MPI_Reduce(errors, 0, NHYDRO + NFIELD + 1,
               MPI_ATHENA_REAL, MPI_SUM, 0, MPI_COMM_WORLD);
  }
#endif

  if (Globals::my_rank != 0) return;

  const Real total_vol = errors[NHYDRO + NFIELD];
  for (int n = 0; n < NHYDRO + NFIELD; ++n) errors[n] /= total_vol;

  // RMS across all components (root-mean-square of per-component L1s)
  Real rms = 0.0;
  for (int n = 0; n < NHYDRO + NFIELD; ++n) rms += SQR(errors[n]);
  rms = std::sqrt(rms / (NHYDRO + NFIELD));

  FILE *pfile = std::fopen("srmhd-cpaw-errors.dat", "r");
  if (pfile != nullptr) {
    pfile = std::freopen("srmhd-cpaw-errors.dat", "a", pfile);
  } else {
    pfile = std::fopen("srmhd-cpaw-errors.dat", "w");
    if (pfile != nullptr) {
      std::fprintf(pfile,
          "# Nx1 Nx2 Nx3 Ncycle RMS-L1  rho  v1  v2  v3  p  B1c  B2c  B3c\n");
    }
  }
  if (pfile == nullptr) {
    std::stringstream msg;
    msg << "### FATAL ERROR in srmhd_cpaw.cpp UserWorkAfterLoop" << std::endl
        << "Could not open srmhd-cpaw-errors.dat." << std::endl;
    ATHENA_ERROR(msg);
  }
  std::fprintf(pfile, "%d %d %d %d %.16e", mesh_size.nx1, mesh_size.nx2,
               mesh_size.nx3, ncycle, rms);
  for (int n = 0; n < NHYDRO + NFIELD; ++n) std::fprintf(pfile, " %.10e", errors[n]);
  std::fprintf(pfile, "\n");
  std::fclose(pfile);
}

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  AthenaArray<Real> a1, a2, a3;
  a1.NewAthenaArray(ncells3 + 1, ncells2 + 1, ncells1 + 1);
  a2.NewAthenaArray(ncells3 + 1, ncells2 + 1, ncells1 + 1);
  a3.NewAthenaArray(ncells3 + 1, ncells2 + 1, ncells1 + 1);

  for (int k = ks; k <= ke + 1; ++k) {
    for (int j = js; j <= je + 1; ++j) {
      for (int i = is; i <= ie + 1; ++i) {
        const Real x1f = pcoord->x1f(i);
        const Real x2f = pcoord->x2f(j);
        if (i <= ie) a1(k, j, i) = A1(pcoord->x1v(i), x2f);
        if (j <= je) a2(k, j, i) = A2(x1f, pcoord->x2v(j));
        if (k <= ke) a3(k, j, i) = A3(x1f, x2f);
      }
    }
  }

  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je; ++j) {
      for (int i = is; i <= ie + 1; ++i) {
        pfield->b.x1f(k, j, i) =
            (a3(k, j + 1, i) - a3(k, j, i)) / pcoord->dx2f(j);
      }
    }
  }
  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je + 1; ++j) {
      for (int i = is; i <= ie; ++i) {
        pfield->b.x2f(k, j, i) =
            -(a3(k, j, i + 1) - a3(k, j, i)) / pcoord->dx1f(i);
      }
    }
  }
  for (int k = ks; k <= ke + 1; ++k) {
    for (int j = js; j <= je; ++j) {
      for (int i = is; i <= ie; ++i) {
        pfield->b.x3f(k, j, i) =
            (a2(k, j, i + 1) - a2(k, j, i)) / pcoord->dx1f(i) -
            (a1(k, j + 1, i) - a1(k, j, i)) / pcoord->dx2f(j);
      }
    }
  }

  AthenaArray<Real> bb;
  bb.NewAthenaArray(3, ncells3, ncells2, ncells1);
  pfield->CalculateCellCenteredField(pfield->b, bb, pcoord, is, ie, js, je, ks, ke);

  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je; ++j) {
      for (int i = is; i <= ie; ++i) {
        // Point-value IC at cell center (no Gauss quadrature).
        PrimitiveState s;
        ExactStateAt(pcoord->x1v(i), pcoord->x2v(j), 0.0, &s);
        const Real v_sq = SQR(s.v1) + SQR(s.v2) + SQR(s.v3);
        const Real lor = 1.0 / std::sqrt(1.0 - v_sq);
        phydro->w(IDN, k, j, i) = phydro->w1(IDN, k, j, i) = s.rho;
        phydro->w(IPR, k, j, i) = phydro->w1(IPR, k, j, i) = s.pgas;
        phydro->w(IVX, k, j, i) = phydro->w1(IVX, k, j, i) = lor * s.v1;
        phydro->w(IVY, k, j, i) = phydro->w1(IVY, k, j, i) = lor * s.v2;
        phydro->w(IVZ, k, j, i) = phydro->w1(IVZ, k, j, i) = lor * s.v3;
      }
    }
  }

  peos->PrimitiveToConserved(phydro->w, bb, phydro->u, pcoord, is, ie, js, je, ks, ke);

  // Store initial U (conserved) and bcc for the multi-component L1 norm in
  // UserWorkAfterLoop. Allocate once per Mesh via lid==0 guard, following the
  // gr_linear_wave pgen pattern. Compared to at tlim = one Alfven period where
  // the exact solution returns to the IC (periodic, v_A·t = wavelength).
  if (compute_error && lid == 0) {
    const int num_blocks = pmy_mesh->nblocal;
    const int nx1 = block_size.nx1;
    const int nx2 = block_size.nx2;
    const int nx3 = block_size.nx3;
    initial.NewAthenaArray(num_blocks, NHYDRO + NFIELD,
                           nx3 + NGHOST, nx2 + NGHOST, nx1 + NGHOST);
  }
  if (compute_error) {
    // Store PRIMITIVES (rho, v^1, v^2, v^3, p) — 3-velocity not 4-velocity.
    // Athena stores util^i = W*v^i in w(IVX/IVY/IVZ); we convert via
    // W = sqrt(1 + util^2), v^i = util^i / W. Using 3-velocity isolates the
    // genuine numerical dissipation from the nonlinear v=util/W bias that is
    // a CPAW test property (amp^2 * (kh)^2) rather than scheme error.
    for (int k = ks; k <= ke; ++k) {
      for (int j = js; j <= je; ++j) {
        for (int i = is; i <= ie; ++i) {
          const Real ux = phydro->w(IVX, k, j, i);
          const Real uy = phydro->w(IVY, k, j, i);
          const Real uz = phydro->w(IVZ, k, j, i);
          const Real W  = std::sqrt(1.0 + SQR(ux) + SQR(uy) + SQR(uz));
          initial(lid, IDN, k, j, i) = phydro->w(IDN, k, j, i);
          initial(lid, IVX, k, j, i) = ux / W;
          initial(lid, IVY, k, j, i) = uy / W;
          initial(lid, IVZ, k, j, i) = uz / W;
          initial(lid, IPR, k, j, i) = phydro->w(IPR, k, j, i);
        }
      }
    }
    // Re-fill pfield->bcc so stored initial B_cc matches the value the error
    // routine reads via pfield->bcc at tlim.
    pfield->CalculateCellCenteredField(pfield->b, pfield->bcc, pcoord,
                                       is, ie, js, je, ks, ke);
    for (int n = IB1; n <= IB3; ++n) {
      for (int k = ks; k <= ke; ++k) {
        for (int j = js; j <= je; ++j) {
          for (int i = is; i <= ie; ++i) {
            initial(lid, NHYDRO + n, k, j, i) = pfield->bcc(n, k, j, i);
          }
        }
      }
    }
  }

  a1.DeleteAthenaArray();
  a2.DeleteAthenaArray();
  a3.DeleteAthenaArray();
  bb.DeleteAthenaArray();
}
