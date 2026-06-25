//========================================================================================
// Athena++ astrophysical MHD code -- EFL extension
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file sr_khi.cpp
//! \brief Kelvin-Helmholtz instability in 2D special-relativistic hydrodynamics.
//!
//! Problem generator ported from athena_z4c/src/pgen/z4c_khi.cpp with Z4C/ADM
//! dependencies stripped.  Four setups (selected via problem/iprob=1..4):
//!
//!   iprob = 1 : sharp shear layer, +/-0.5 vx across |y|=0.25 with sine-wave
//!               perturbation in vy.
//!   iprob = 2 : THC paper Radice & Rezzolla (2012) §3.2.6 setup: symmetric
//!               double shear layer at y=+/-0.5 using tanh profiles in density
//!               and velocity, Gaussian-localized sinusoidal perturbation in vy.
//!   iprob = 3 : high-pressure sliding-wall setup with a weakly deformed y=0
//!               interface (pgas=1000, ρ=10 vs 1, v_x=0 vs 0.9).
//!   iprob = 4 : smooth single shear layer (as in Mignone 2012 SRHD KHI test).
//!
//! Primitive velocity slots store u^i = W * v^i.

// C++ headers
#include <cmath>

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
#error "This problem generator requires relativity (-s)"
#endif

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  const int iprob = pin->GetOrAddInteger("problem", "iprob", 2);
  AthenaArray<Real> &pos_x = pcoord->x1v;
  AthenaArray<Real> &pos_y = pcoord->x2v;

  Real rho, pgas, vx, vy, vz;

  // ------------------------------------------------------------------ iprob 1
  if (iprob == 1) {
    const Real p_amp = pin->GetOrAddReal("problem", "p_amp", 0.01);
    pgas = 2.5;
    vz = 0.0;
    const Real L = 1.0;
    for (int k = ks; k <= ke; ++k) {
      for (int j = js; j <= je; ++j) {
        for (int i = is; i <= ie; ++i) {
          if (std::abs(pos_y(j)) >= 0.25) {
            rho = 1.0;
            vx  = 0.5;
          } else {
            rho = 2.0;
            vx  = -0.5;
          }
          vy = p_amp * std::sin(2.0 * PI * pos_x(i) / L);

          const Real v2 = SQR(vx) + SQR(vy) + SQR(vz);
          const Real W  = 1.0 / std::sqrt(1.0 - v2);
          phydro->w(IDN, k, j, i) = rho;
          phydro->w(IPR, k, j, i) = pgas;
          phydro->w(IVX, k, j, i) = W * vx;
          phydro->w(IVY, k, j, i) = W * vy;
          phydro->w(IVZ, k, j, i) = W * vz;
        }
      }
    }
  }

  // ------------------------------------------------------------------ iprob 2
  // THC paper (Radice & Rezzolla 2012) §3.2.6 KHI: symmetric double shear
  // layer at y = +/-0.5 with tanh velocity + density jumps and a Gaussian-
  // localized sinusoidal vy perturbation.
  if (iprob == 2) {
    pgas                    = pin->GetOrAddReal("problem", "p",        1.0);
    const Real v_shear      = pin->GetOrAddReal("problem", "v_shear",  0.5);
    const Real char_size    = pin->GetOrAddReal("problem", "char_size", 0.01);
    const Real p_amp        = pin->GetOrAddReal("problem", "p_amp",    0.1);
    const Real char_len     = pin->GetOrAddReal("problem", "char_len", 0.1);
    const Real d_0          = pin->GetOrAddReal("problem", "d_0",      0.505);
    const Real d_1          = pin->GetOrAddReal("problem", "d_1",      0.495);
    vz = 0.0;
    for (int k = ks; k <= ke; ++k) {
      for (int j = js; j <= je; ++j) {
        for (int i = is; i <= ie; ++i) {
          const Real y = pos_y(j);
          const Real x = pos_x(i);
          if (y <= 0.0) {
            rho = d_0 - d_1 * std::tanh((y + 0.5) / char_size);
            vx  = -v_shear * std::tanh((y + 0.5) / char_size);
            vy  = -p_amp * v_shear * std::sin(2.0 * PI * x)
                * std::exp(-SQR((y + 0.5) / char_len));
          } else {
            rho = d_0 + d_1 * std::tanh((y - 0.5) / char_size);
            vx  =  v_shear * std::tanh((y - 0.5) / char_size);
            vy  =  p_amp * v_shear * std::sin(2.0 * PI * x)
                * std::exp(-SQR((y - 0.5) / char_len));
          }

          const Real v2 = SQR(vx) + SQR(vy) + SQR(vz);
          const Real W  = 1.0 / std::sqrt(1.0 - v2);
          phydro->w(IDN, k, j, i) = rho;
          phydro->w(IPR, k, j, i) = pgas;
          phydro->w(IVX, k, j, i) = W * vx;
          phydro->w(IVY, k, j, i) = W * vy;
          phydro->w(IVZ, k, j, i) = W * vz;
        }
      }
    }
  }

  // ------------------------------------------------------------------ iprob 3
  if (iprob == 3) {
    vz = 0.0;
    vy = 0.0;
    for (int k = ks; k <= ke; ++k) {
      for (int j = js; j <= je; ++j) {
        for (int i = is; i <= ie; ++i) {
          pgas = 1000.0;
          const Real y_bound = 0.01 * std::sin(2.0 * PI * pos_x(i));
          if (pos_y(j) < y_bound) {
            rho = 10.0;
            vx  = 0.0;
          } else {
            rho = 1.0;
            vx  = 0.9;
          }
          const Real v2 = SQR(vx) + SQR(vy) + SQR(vz);
          const Real W  = 1.0 / std::sqrt(1.0 - v2);
          phydro->w(IDN, k, j, i) = rho;
          phydro->w(IPR, k, j, i) = pgas;
          phydro->w(IVX, k, j, i) = W * vx;
          phydro->w(IVY, k, j, i) = W * vy;
          phydro->w(IVZ, k, j, i) = W * vz;
        }
      }
    }
  }

  // ------------------------------------------------------------------ iprob 4
  if (iprob == 4) {
    pgas               = pin->GetOrAddReal("problem", "p",    1.0);
    const Real amp     = pin->GetOrAddReal("problem", "amp",  0.1);
    const Real a       = 0.01;
    const Real sigma   = 0.1;
    const Real vflow   = 0.5;
    vz = 0.0;
    for (int k = ks; k <= ke; ++k) {
      for (int j = js; j <= je; ++j) {
        for (int i = is; i <= ie; ++i) {
          rho = 0.505 + 0.495 * std::tanh((std::abs(pcoord->x2v(j)) - 0.5) / a);
          vx  = vflow * std::tanh((std::abs(pcoord->x2v(j)) - 0.5) / a);
          vy  = amp * vflow * std::sin(TWO_PI * pcoord->x1v(i))
              * std::exp(-SQR((std::abs(pcoord->x2v(j)) - 0.5) / sigma));
          if (pcoord->x2v(j) < 0.0) vy = -vy;

          const Real v2 = SQR(vx) + SQR(vy) + SQR(vz);
          const Real W  = 1.0 / std::sqrt(1.0 - v2);
          phydro->w(IDN, k, j, i) = rho;
          phydro->w(IPR, k, j, i) = pgas;
          phydro->w(IVX, k, j, i) = W * vx;
          phydro->w(IVY, k, j, i) = W * vy;
          phydro->w(IVZ, k, j, i) = W * vz;
        }
      }
    }
  }

  // Compute conserved variables from primitives
  AthenaArray<Real> bb;
  bb.NewAthenaArray(3, ncells3, ncells2, ncells1);
  peos->PrimitiveToConserved(phydro->w, bb, phydro->u, pcoord,
                             is, ie, js, je, ks, ke);
  bb.DeleteAthenaArray();
}
