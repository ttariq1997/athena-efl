//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file srmhd_rotor.cpp
//! \brief Problem generator for the 2D/3D special-relativistic MHD rotor test.
//!
//! Reference setup:
//!   Martí & Müller (2015), Living Reviews in Computational Astrophysics 1:3,
//!   Section 6.6.2 "The relativistic rotor", summarizing Del Zanna et al. (2003).
//!
//! Interface convention:
//!   `use_smooth_interface=true`  (default): linear ramp of rho and v on
//!         r in [r_inner, r_outer]. Matches Del Zanna 2003 / B&S 2011 /
//!         Mignone-Bodo 2006 numerical convention (avoids grid-aligned aliasing).
//!   `use_smooth_interface=false`:           sharp step at r = r_inner. Matches
//!         the strict description of Anton et al. 2006 sect 8.2.2.
//========================================================================================

#include <algorithm>
#include <cmath>
#include <cstring>
#include <sstream>

#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
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

// Rotor-interior mass fraction as a function of radius.
// Returns 1 fully inside the rotor, 0 fully outside, and (when
// use_smooth_interface = true) a linear ramp on r in [r_inner, r_outer]
// that avoids grid-aligned aliasing. When use_smooth_interface = false
// this collapses to a sharp step at r_inner (Anton et al. 2006 sect 8.2.2).
Real RotorInteriorFraction(const Real radius, const Real r_inner,
                           const Real r_outer, const bool use_smooth_interface) {
  if (!use_smooth_interface) {
    return (radius <= r_inner) ? 1.0 : 0.0;
  }
  if (radius <= r_inner) return 1.0;
  if (radius >= r_outer) return 0.0;
  return (r_outer - radius) / (r_outer - r_inner);
}

}  // namespace

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  if (block_size.nx2 == 1) {
    std::stringstream msg;
    msg << "### FATAL ERROR in srmhd_rotor.cpp ProblemGenerator" << std::endl
        << "The SRMHD rotor requires at least a 2D mesh." << std::endl;
    ATHENA_ERROR(msg);
  }
  if (std::strcmp(COORDINATE_SYSTEM, "cartesian") != 0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in srmhd_rotor.cpp ProblemGenerator" << std::endl
        << "The SRMHD rotor is implemented only in Cartesian coordinates."
        << std::endl;
    ATHENA_ERROR(msg);
  }

  const Real x_center = pin->GetOrAddReal("problem", "x_center",
                                          0.5 * (pmy_mesh->mesh_size.x1min
                                               + pmy_mesh->mesh_size.x1max));
  const Real y_center = pin->GetOrAddReal("problem", "y_center",
                                          0.5 * (pmy_mesh->mesh_size.x2min
                                               + pmy_mesh->mesh_size.x2max));
  const Real r_inner = pin->GetOrAddReal("problem", "r_inner", 0.1);
  const Real r_outer = pin->GetOrAddReal("problem", "r_outer", 0.115);
  const Real rho_disk = pin->GetOrAddReal("problem", "rho_disk", 10.0);
  const Real rho_ambient = pin->GetOrAddReal("problem", "rho_ambient", 1.0);
  const Real pgas = pin->GetOrAddReal("problem", "pgas", 1.0);
  const Real omega = pin->GetOrAddReal("problem", "omega", 9.95);
  const Real bx0 = pin->GetOrAddReal("problem", "bx0", 1.0);
  const Real by0 = pin->GetOrAddReal("problem", "by0", 0.0);
  const Real bz0 = pin->GetOrAddReal("problem", "bz0", 0.0);
  const bool use_smooth_interface = pin->GetOrAddBoolean(
      "problem", "use_smooth_interface", true);

  if (use_smooth_interface && r_outer <= r_inner) {
    std::stringstream msg;
    msg << "### FATAL ERROR in srmhd_rotor.cpp ProblemGenerator" << std::endl
        << "Need r_outer > r_inner when use_smooth_interface = true "
        << "(smooth-interface ramp region)." << std::endl;
    ATHENA_ERROR(msg);
  }
  if (omega * r_inner >= 1.0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in srmhd_rotor.cpp ProblemGenerator" << std::endl
        << "omega * r_inner must be < 1 so that the rotor edge remains subluminal."
        << std::endl;
    ATHENA_ERROR(msg);
  }

  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je; ++j) {
      for (int i = is; i <= ie; ++i) {
        const Real x = pcoord->x1v(i) - x_center;
        const Real y = pcoord->x2v(j) - y_center;
        const Real radius = std::sqrt(SQR(x) + SQR(y));
        const Real f_in = RotorInteriorFraction(radius, r_inner, r_outer,
                                                use_smooth_interface);

        const Real rho = rho_ambient + (rho_disk - rho_ambient) * f_in;
        const Real vx = -omega * y * f_in;
        const Real vy =  omega * x * f_in;
        const Real v2 = SQR(vx) + SQR(vy);
        if (v2 >= 1.0) {
          std::stringstream msg;
          msg << "### FATAL ERROR in srmhd_rotor.cpp ProblemGenerator" << std::endl
              << "Superluminal rotor velocity at i=" << i
              << ", j=" << j << ", k=" << k << std::endl;
          ATHENA_ERROR(msg);
        }
        const Real lor = 1.0 / std::sqrt(1.0 - v2);

        phydro->w(IDN, k, j, i) = phydro->w1(IDN, k, j, i) = rho;
        phydro->w(IPR, k, j, i) = phydro->w1(IPR, k, j, i) = pgas;
        phydro->w(IVX, k, j, i) = phydro->w1(IVX, k, j, i) = lor * vx;
        phydro->w(IVY, k, j, i) = phydro->w1(IVY, k, j, i) = lor * vy;
        phydro->w(IVZ, k, j, i) = phydro->w1(IVZ, k, j, i) = 0.0;
      }
    }
  }

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
        pfield->b.x2f(k, j, i) = by0;
      }
    }
  }
  for (int k = ks; k <= ke + 1; ++k) {
    for (int j = js; j <= je; ++j) {
      for (int i = is; i <= ie; ++i) {
        pfield->b.x3f(k, j, i) = bz0;
      }
    }
  }

  // Keep the cell-centered magnetic field used by SRMHD characteristic
  // construction consistent with the initialized face field.
  pfield->CalculateCellCenteredField(pfield->b, pfield->bcc, pcoord,
                                     is, ie, js, je, ks, ke);
  peos->PrimitiveToConserved(phydro->w, pfield->bcc, phydro->u, pcoord,
                             is, ie, js, je, ks, ke);
}
