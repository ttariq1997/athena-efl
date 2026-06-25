//========================================================================================
// Athena++ astrophysical MHD code -- EFL extension
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file srmhd_khi.cpp
//! \brief Relativistic magnetized Kelvin-Helmholtz instability.
//!
//! Two setups (selected via problem/iprob):
//!
//!   iprob = 1 : Mignone et al. (2012) PLUTO paper §6.6 (ApJS 198, 7).
//!               Single shear layer vx = V0*tanh(y/a), hot gas (p >> rho),
//!               uniform poloidal+toroidal magnetic field from sigma_pol/sigma_tor.
//!               Domain: x in [0,1], y in [-1,1].
//!
//!   iprob = 2 : Beckwith & Stone (2011) ApJS 193, 6 §4.7 — magnetized analog
//!               of the Newtonian test of kh.cpp iprob=3. Symmetric double shear
//!               layer at |y|=0.5 with density jump (rho=1 outside, rho=0.01
//!               inside), V_shear=0.5 (Γ_rel=2.29), single-mode sin(2πx)
//!               perturbation with Gaussian envelope localized on each shear
//!               layer and sign-flipped across y=0. Uniform weak Bx = 10^-3.
//!               Domain: x in [-0.5,0.5], y in [-1,1], periodic all sides.
//!               γ=4/3, pgas=1.0.
//!
//! Primitive velocity slots store u^i = W * v^i.

#include <cmath>
#include <sstream>
#include <string>

#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"

#if !RELATIVISTIC_DYNAMICS
#error "This problem generator requires relativity (-s or -g)"
#endif

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  const int iprob = pin->GetOrAddInteger("problem", "iprob", 1);

  Real Bx0 = 0.0;
  Real By0 = 0.0;
  Real Bz0 = 0.0;

  // ------------------------------------------------------------------ iprob 1
  // Mignone et al. 2012 PLUTO §6.6: single shear layer, hot gas.
  if (iprob == 1) {
    const Real rho0  = pin->GetOrAddReal("problem", "rho",  1.0);
    const Real pgas0 = pin->GetOrAddReal("problem", "pgas", 20.0);
    const Real V0    = pin->GetOrAddReal("problem", "V0",   0.25);
    const Real eps   = pin->GetOrAddReal("problem", "eps",  V0 / 100.0);
    const Real a     = pin->GetOrAddReal("problem", "a",    0.01);
    const Real b     = pin->GetOrAddReal("problem", "b",    0.1);

    const Real sigma_pol = pin->GetOrAddReal("problem", "sigma_pol", 0.01);
    const Real sigma_tor = pin->GetOrAddReal("problem", "sigma_tor", 1.0);
    Bx0 = std::sqrt(2.0 * sigma_pol * pgas0);
    Bz0 = std::sqrt(2.0 * sigma_tor * pgas0);

    for (int k = ks; k <= ke; ++k) {
      for (int j = js; j <= je; ++j) {
        for (int i = is; i <= ie; ++i) {
          const Real x = pcoord->x1v(i);
          const Real y = pcoord->x2v(j);

          const Real vx = V0 * std::tanh(y / a);
          const Real vy = eps * std::sin(2.0 * PI * x)
                        * std::exp(-y * y / (b * b));

          const Real vsq = vx * vx + vy * vy;
          const Real W = 1.0 / std::sqrt(1.0 - vsq);

          phydro->w(IDN, k, j, i) = rho0;
          phydro->w(IPR, k, j, i) = pgas0;
          phydro->w(IVX, k, j, i) = W * vx;
          phydro->w(IVY, k, j, i) = W * vy;
          phydro->w(IVZ, k, j, i) = 0.0;
        }
      }
    }
  }

  // ------------------------------------------------------------------ iprob 2
  // Beckwith & Stone 2011 §4.7: symmetric double shear layer, density jump,
  // single-mode sin(2πx) perturbation with Gaussian envelope localized on each
  // shear layer, sign-flipped at y=0. Weak uniform Bx.
  if (iprob == 2) {
    const Real V_shear = pin->GetOrAddReal("problem", "V_shear", 0.5);
    const Real amp     = pin->GetOrAddReal("problem", "amp",     0.1);
    const Real a       = pin->GetOrAddReal("problem", "a",       0.01);
    const Real sigma   = pin->GetOrAddReal("problem", "sigma",   0.1);
    const Real pgas0   = pin->GetOrAddReal("problem", "pgas",    1.0);
    const Real rho_h   = pin->GetOrAddReal("problem", "rho_h",   1.0);
    const Real rho_l   = pin->GetOrAddReal("problem", "rho_l",   0.01);
    Bx0                = pin->GetOrAddReal("problem", "Bx0",     1.0e-3);

    const Real rho_mid = 0.5 * (rho_h + rho_l);
    const Real drho    = 0.5 * (rho_h - rho_l);

    for (int k = ks; k <= ke; ++k) {
      for (int j = js; j <= je; ++j) {
        const Real y    = pcoord->x2v(j);
        const Real yabs = std::abs(y);
        const Real arg  = (yabs - 0.5) / a;
        const Real env  = std::exp(-SQR((yabs - 0.5) / sigma));
        const Real tanh_arg = std::tanh(arg);
        const Real rho  = rho_mid + drho * tanh_arg;
        const Real vx   = V_shear * tanh_arg;
        for (int i = is; i <= ie; ++i) {
          const Real x = pcoord->x1v(i);
          Real vy = amp * V_shear * std::sin(2.0 * PI * x) * env;
          if (y < 0.0) vy = -vy;

          const Real vsq = vx * vx + vy * vy;
          const Real W = 1.0 / std::sqrt(1.0 - vsq);

          phydro->w(IDN, k, j, i) = rho;
          phydro->w(IPR, k, j, i) = pgas0;
          phydro->w(IVX, k, j, i) = W * vx;
          phydro->w(IVY, k, j, i) = W * vy;
          phydro->w(IVZ, k, j, i) = 0.0;
        }
      }
    }
  }

  // Initialize uniform magnetic field (both setups use uniform Bx, Bz)
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k = ks; k <= ke; ++k) {
      for (int j = js; j <= je; ++j) {
        for (int i = is; i <= ie + 1; ++i) {
          pfield->b.x1f(k, j, i) = Bx0;
        }
      }
    }
    for (int k = ks; k <= ke; ++k) {
      for (int j = js; j <= je + 1; ++j) {
        for (int i = is; i <= ie; ++i) {
          pfield->b.x2f(k, j, i) = By0;
        }
      }
    }
    for (int k = ks; k <= ke + 1; ++k) {
      for (int j = js; j <= je; ++j) {
        for (int i = is; i <= ie; ++i) {
          pfield->b.x3f(k, j, i) = Bz0;
        }
      }
    }
  }

  // Compute conserved variables from primitives
  AthenaArray<Real> bb;
  bb.NewAthenaArray(3, ncells3, ncells2, ncells1);
  if (MAGNETIC_FIELDS_ENABLED) {
    pfield->CalculateCellCenteredField(pfield->b, bb, pcoord,
                                       is, ie, js, je, ks, ke);
  }
  peos->PrimitiveToConserved(phydro->w, bb, phydro->u, pcoord,
                             is, ie, js, je, ks, ke);
  bb.DeleteAthenaArray();
}
