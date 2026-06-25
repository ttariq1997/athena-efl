//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file srmhd_jet.cpp
//! \brief 2D relativistic magnetized jet propagating into a uniform ambient medium.
//!
//! References:
//!   Martí, Müller, Font, Ibáñez & Marquina (1997), ApJ 479, 151
//!   Leismann, Antón, Aloy, Müller, Martí, Miralles & Ibáñez (2005), A&A 436, 503
//!   Mignone, Ugliano & Bodo (2009), MNRAS 393, 1141
//!
//! Setup: A pressure-matched beam is injected from x1min into a uniform ambient
//! medium at rest.  The beam occupies |x2| < r_jet at the left boundary and
//! carries a relativistic bulk velocity v_x = v_jet (Lorentz factor W_jet).
//! A uniform axial magnetic field Bx threads the entire domain; an optional
//! toroidal (By) field can be added inside the beam.
//!
//! Boundary conditions:
//!   x1min — reflecting wall with jet inlet injection (user BC)
//!   x1max — outflow
//!   x2min/x2max — outflow

#include <algorithm>
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

namespace {

// Jet parameters (set in InitUserMeshData)
Real rho_amb, p_amb, bx0;
Real rho_jet, p_jet, vx_jet, by_jet;
Real r_jet;
Real gamma_adi;
Real smooth_width;  // smoothing width for jet edge (0 = sharp)

}  // namespace

//----------------------------------------------------------------------------------------
// User-defined boundary at x1min: reflecting wall + jet inlet

void JetInnerX1(MeshBlock *pmb, Coordinates *pco,
                AthenaArray<Real> &prim, FaceField &b,
                Real time, Real dt,
                int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  for (int k = kl; k <= ku; ++k) {
    for (int j = jl; j <= ju; ++j) {
      const Real x2 = pco->x2v(j);
      // Smooth transition: 1 inside jet, 0 outside
      Real fjet;
      if (smooth_width > 0.0) {
        fjet = 0.5 * (1.0 - std::tanh((std::abs(x2) - r_jet) / smooth_width));
      } else {
        fjet = (std::abs(x2) <= r_jet) ? 1.0 : 0.0;
      }

      for (int i = 1; i <= ngh; ++i) {
        if (fjet > 0.5) {
          // Inside jet inlet: impose beam state
          prim(IDN, k, j, il - i) = rho_jet;
          prim(IVX, k, j, il - i) = vx_jet * std::sqrt(1.0 + SQR(vx_jet));
          // Store as 4-velocity u^x = W*v_x. Actually Athena stores
          // spatial 4-velocity components u^i = W*v^i.
          // v_jet is 3-velocity, so u^x = v_jet / sqrt(1 - v_jet^2)
          Real W_jet = 1.0 / std::sqrt(1.0 - SQR(vx_jet));
          prim(IVX, k, j, il - i) = W_jet * vx_jet;
          prim(IVY, k, j, il - i) = 0.0;
          prim(IVZ, k, j, il - i) = 0.0;
          prim(IPR, k, j, il - i) = p_jet;
        } else {
          // Outside jet: reflecting wall
          prim(IDN, k, j, il - i) = prim(IDN, k, j, il + i - 1);
          prim(IVX, k, j, il - i) = -prim(IVX, k, j, il + i - 1);  // reflect v_x
          prim(IVY, k, j, il - i) = prim(IVY, k, j, il + i - 1);
          prim(IVZ, k, j, il - i) = prim(IVZ, k, j, il + i - 1);
          prim(IPR, k, j, il - i) = prim(IPR, k, j, il + i - 1);
        }
      }
    }
  }

  // Magnetic field in ghost zones
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k = kl; k <= ku; ++k) {
      for (int j = jl; j <= ju; ++j) {
        const Real x2 = pco->x2v(j);
        Real fjet;
        if (smooth_width > 0.0) {
          fjet = 0.5 * (1.0 - std::tanh((std::abs(x2) - r_jet) / smooth_width));
        } else {
          fjet = (std::abs(x2) <= r_jet) ? 1.0 : 0.0;
        }

        for (int i = 1; i <= ngh; ++i) {
          if (fjet > 0.5) {
            b.x1f(k, j, il - i) = bx0;
          } else {
            b.x1f(k, j, il - i) = b.x1f(k, j, il + i - 1);
          }
        }
      }
    }
    for (int k = kl; k <= ku; ++k) {
      for (int j = jl; j <= ju + 1; ++j) {
        const Real x2f = pco->x2f(j);
        for (int i = 1; i <= ngh; ++i) {
          bool in_jet = (std::abs(x2f) <= r_jet + 2.0 * smooth_width);
          if (in_jet) {
            b.x2f(k, j, il - i) = by_jet;
          } else {
            b.x2f(k, j, il - i) = -b.x2f(k, j, il + i - 1);  // reflect
          }
        }
      }
    }
    for (int k = kl; k <= ku + 1; ++k) {
      for (int j = jl; j <= ju; ++j) {
        for (int i = 1; i <= ngh; ++i) {
          b.x3f(k, j, il - i) = b.x3f(k, j, il + i - 1);
        }
      }
    }
  }
}

//----------------------------------------------------------------------------------------
// InitUserMeshData

void Mesh::InitUserMeshData(ParameterInput *pin) {
  gamma_adi = pin->GetReal("hydro", "gamma");

  // Ambient medium
  rho_amb = pin->GetOrAddReal("problem", "rho_amb", 1.0);
  p_amb   = pin->GetOrAddReal("problem", "p_amb", 1.0);
  bx0     = pin->GetOrAddReal("problem", "bx0", 0.5);

  // Jet parameters
  Real eta = pin->GetOrAddReal("problem", "eta", 0.01);  // density ratio
  Real v_jet_3vel = pin->GetOrAddReal("problem", "v_jet", 0.99);  // 3-velocity
  by_jet  = pin->GetOrAddReal("problem", "by_jet", 0.0);
  r_jet   = pin->GetOrAddReal("problem", "r_jet", 1.0);
  smooth_width = pin->GetOrAddReal("problem", "smooth_width", 0.05);

  rho_jet = eta * rho_amb;
  p_jet   = p_amb;  // pressure-matched
  vx_jet  = v_jet_3vel;

  Real W_jet = 1.0 / std::sqrt(1.0 - SQR(vx_jet));
  if (Globals::my_rank == 0) {
    std::printf("SRMHD Jet: eta=%.4f  v_jet=%.4f  W_jet=%.2f  rho_jet=%.4e  "
                "p_jet=%.4e  Bx=%.4f  By_jet=%.4f  r_jet=%.2f\n",
                eta, vx_jet, W_jet, rho_jet, p_jet, bx0, by_jet, r_jet);
  }

  // Enroll user boundary
  EnrollUserBoundaryFunction(BoundaryFace::inner_x1, JetInnerX1);
}

//----------------------------------------------------------------------------------------
// ProblemGenerator

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  // Initialize uniform ambient medium
  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je; ++j) {
      for (int i = is; i <= ie; ++i) {
        phydro->w(IDN, k, j, i) = rho_amb;
        phydro->w(IVX, k, j, i) = 0.0;
        phydro->w(IVY, k, j, i) = 0.0;
        phydro->w(IVZ, k, j, i) = 0.0;
        phydro->w(IPR, k, j, i) = p_amb;
      }
    }
  }

  // Initialize magnetic field: uniform Bx, zero By/Bz
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k = ks; k <= ke; ++k) {
      for (int j = js; j <= je; ++j) {
        for (int i = is; i <= ie + 1; ++i) {
          pfield->b.x1f(k, j, i) = bx0;
        }
      }
    }
    for (int k = ks; k <= ke; ++k) {
      for (int j = js; j <= je + 1; ++j) {
        for (int i = is; i <= ie; ++i) {
          pfield->b.x2f(k, j, i) = 0.0;
        }
      }
    }
    for (int k = ks; k <= ke + 1; ++k) {
      for (int j = js; j <= je; ++j) {
        for (int i = is; i <= ie; ++i) {
          pfield->b.x3f(k, j, i) = 0.0;
        }
      }
    }
  }

  // Compute conserved variables from primitives
  peos->PrimitiveToConserved(phydro->w, pfield->bcc, phydro->u, pcoord,
                             is, ie, js, je, ks, ke);
}
