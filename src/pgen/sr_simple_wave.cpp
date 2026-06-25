//========================================================================================
// Athena++ astrophysical MHD code -- EFL extension
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file sr_simple_wave.cpp
//! \brief Problem generator for a right-going simple wave in special-relativistic
//!        hydrodynamics.
//!
//! A simple wave is an exact nonlinear solution in which one Riemann invariant
//! (J_-) is constant everywhere.  Given a velocity profile v(x), the density
//! and pressure follow from the invariant relation:
//!
//!   c_s(v) = sqrt(gamma-1) * [ C_term * V_term - 1 ] / [ C_term * V_term + 1 ]
//!   eps    = c_s^2 / [ gamma * (gamma-1 - c_s^2) ]
//!   rho    = [ eps*(gamma-1) / K ]^{1/(gamma-1)}
//!   p      = (gamma-1) * rho * eps
//!
//! where C_term and V_term depend on the reference sound speed c_0 at v=0 and
//! the local velocity v.  The velocity profile is a smooth bump:
//!   v = a * sin^6( pi*(x/(2X) - 0.5) )   for |x| < X
//!   v = 0                                  otherwise
//!
//! Reference: Rezzolla & Zanotti (2013), §4.6 "Simple Waves".
//!
//! This test is valuable for convergence studies because the exact solution is
//! known (by tracing characteristics) and does not suffer from the O(amp^2)
//! nonlinear vs. linear mismatch inherent to linearized-wave tests.

// C headers

// C++ headers
#include <cmath>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"

#if !RELATIVISTIC_DYNAMICS
#error "This problem generator requires special relativity (-s)"
#endif

namespace {
Real gamma_adi;
Real gmo;         // gamma - 1
Real sqrtgmo;     // sqrt(gamma - 1)
Real e_exp;       // sqrtgmo / 2
Real amp;         // velocity amplitude
Real X_half;      // half-width of the bump
Real rho_0;       // reference density (v=0 state)
Real eos_k;       // polytropic constant K
Real c0;          // reference sound speed at v=0
Real cterm;       // (sqrtgmo + c0) / (sqrtgmo - c0)
bool compute_error;
AthenaArray<Real> initial;   // stored initial conserved state for error calculation
AthenaArray<Real> volume;    // cell volumes for error weighting

void SimpleWaveState(Real v, Real &rho_out, Real &pgas_out) {
  const Real vterm = std::pow((1.0 + v) / (1.0 - v), e_exp);
  const Real cs = sqrtgmo * (cterm * vterm - 1.0) / (cterm * vterm + 1.0);
  const Real cs2 = cs * cs;
  const Real eps = cs2 / (gamma_adi * (gmo - cs2));
  rho_out  = std::pow(eps * gmo / eos_k, 1.0 / gmo);
  pgas_out = gmo * rho_out * eps;
}
}  // anonymous namespace

void Mesh::InitUserMeshData(ParameterInput *pin) {
  // nothing needed at mesh level
}

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  gamma_adi = peos->GetGamma();
  gmo       = gamma_adi - 1.0;
  sqrtgmo   = std::sqrt(gmo);
  e_exp     = sqrtgmo / 2.0;
  amp       = pin->GetOrAddReal("problem", "amp", 0.5);
  X_half    = pin->GetOrAddReal("problem", "X_half", 0.3);
  rho_0     = pin->GetOrAddReal("problem", "rho_0", 1.0);
  eos_k     = pin->GetOrAddReal("problem", "eos_k", 100.0);
  compute_error = pin->GetOrAddBoolean("problem", "compute_error", false);

  // Reference sound speed at v=0 (background state)
  const Real p0 = eos_k * std::pow(rho_0, gamma_adi);
  const Real h0 = 1.0 + gamma_adi / gmo * p0 / rho_0;
  c0 = std::sqrt(gamma_adi * p0 / (rho_0 * h0));
  cterm = (sqrtgmo + c0) / (sqrtgmo - c0);

  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je; ++j) {
      for (int i = is; i <= ie; ++i) {
        const Real x = pcoord->x1v(i);

        Real v = 0.0;
        if (std::abs(x) < X_half) {
          const Real arg = PI * (x / (2.0 * X_half) - 0.5);
          v = amp * std::pow(std::sin(arg), 6);
        }

        Real rho, pgas;
        SimpleWaveState(v, rho, pgas);

        const Real W = 1.0 / std::sqrt(1.0 - v * v);
        phydro->w(IDN, k, j, i) = rho;
        phydro->w(IPR, k, j, i) = pgas;
        phydro->w(IVX, k, j, i) = W * v;   // u^x = W v
        phydro->w(IVY, k, j, i) = 0.0;
        phydro->w(IVZ, k, j, i) = 0.0;
      }
    }
  }

  AthenaArray<Real> bb;
  bb.NewAthenaArray(3, ke + 1, je + 1, ie + 1);
  peos->PrimitiveToConserved(phydro->w, bb, phydro->u, pcoord,
                             is, ie, js, je, ks, ke);
  bb.DeleteAthenaArray();

  if (compute_error && lid == 0) {
    const int num_blocks = pmy_mesh->nblocal;
    const int nx1 = block_size.nx1;
    const int nx2 = block_size.nx2;
    const int nx3 = block_size.nx3;
    initial.NewAthenaArray(num_blocks, NHYDRO, nx3 + NGHOST,
                           nx2 + NGHOST, nx1 + NGHOST);
    volume.NewAthenaArray(nx1 + NGHOST);
  }
  if (compute_error) {
    for (int n = 0; n < NHYDRO; ++n) {
      for (int k = ks; k <= ke; ++k) {
        for (int j = js; j <= je; ++j) {
          for (int i = is; i <= ie; ++i) {
            initial(lid, n, k, j, i) = phydro->u(n, k, j, i);
          }
        }
      }
    }
  }
}

void Mesh::UserWorkAfterLoop(ParameterInput *pin) {
  if (!compute_error) return;

  Real errors[NHYDRO + 1];
  for (int n = 0; n < NHYDRO + 1; ++n) errors[n] = 0.0;

  for (int b = 0; b < nblocal; ++b) {
    MeshBlock *pmb = my_blocks(b);
    for (int k = pmb->ks; k <= pmb->ke; ++k) {
      for (int j = pmb->js; j <= pmb->je; ++j) {
        pmb->pcoord->CellVolume(k, j, pmb->is, pmb->ie, volume);
        for (int i = pmb->is; i <= pmb->ie; ++i) {
          for (int n = 0; n < NHYDRO; ++n) {
            errors[n] += std::abs(pmb->phydro->u(n, k, j, i)
                                  - initial(pmb->lid, n, k, j, i)) * volume(i);
          }
          errors[NHYDRO] += volume(i);
        }
      }
    }
  }

#ifdef MPI_PARALLEL
  if (Globals::my_rank == 0) {
    MPI_Reduce(MPI_IN_PLACE, errors, NHYDRO + 1, MPI_ATHENA_REAL,
               MPI_SUM, 0, MPI_COMM_WORLD);
  } else {
    MPI_Reduce(errors, 0, NHYDRO + 1, MPI_ATHENA_REAL,
               MPI_SUM, 0, MPI_COMM_WORLD);
  }
#endif

  if (Globals::my_rank == 0) {
    for (int n = 0; n < NHYDRO; ++n) errors[n] /= errors[NHYDRO];

    Real total_error = 0.0;
    for (int n = 0; n < NHYDRO; ++n) total_error += SQR(errors[n]);
    total_error = std::sqrt(total_error / NHYDRO);

    std::string filename("simple-wave-errors.dat");
    FILE *pfile = std::fopen(filename.c_str(), "r");
    if (pfile != nullptr) {
      pfile = std::freopen(filename.c_str(), "a", pfile);
    } else {
      pfile = std::fopen(filename.c_str(), "w");
      std::fprintf(pfile, "# Nx1  Nx2  Nx3  Ncycle  RMS-Error  D  E  M1  M2  M3\n");
    }
    std::fprintf(pfile, "%d  %d  %d  %d  %e", mesh_size.nx1, mesh_size.nx2,
                 mesh_size.nx3, ncycle, total_error);
    for (int n = 0; n < NHYDRO; ++n) std::fprintf(pfile, "  %e", errors[n]);
    std::fprintf(pfile, "\n");
    std::fclose(pfile);
  }
}
