//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file srmhd_cyl_blast.cpp
//! \brief Problem generator for the 2D/3D SRMHD cylindrical blast wave.
//!
//! Reference setup:
//!   Martí & Müller (2015), LRCA 1:3, review of Komissarov's magnetized blast wave.
//!   Standard values: rho_in=1e-2, p_in=1, rho_out=1e-4, p_out=3e-5, B=(0.1,0,0),
//!   r_in=0.8, r_out=1.0, Gamma=4/3 on a square box of side length 12.
//========================================================================================

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

Real SmoothLogProfile(const Real radius, const Real r_inner, const Real r_outer,
                      const Real inner, const Real outer) {
  if (radius <= r_inner) return inner;
  if (radius >= r_outer) return outer;
  const Real frac = (r_inner - radius) / (r_inner - r_outer);
  return std::exp(std::log(inner) + frac * std::log(outer / inner));
}

}  // namespace

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  if (block_size.nx2 == 1) {
    std::stringstream msg;
    msg << "### FATAL ERROR in srmhd_cyl_blast.cpp ProblemGenerator" << std::endl
        << "The SRMHD cylindrical blast requires at least a 2D mesh." << std::endl;
    ATHENA_ERROR(msg);
  }
  if (std::strcmp(COORDINATE_SYSTEM, "cartesian") != 0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in srmhd_cyl_blast.cpp ProblemGenerator" << std::endl
        << "The SRMHD cylindrical blast is implemented only in Cartesian coordinates."
        << std::endl;
    ATHENA_ERROR(msg);
  }

  const Real x_center = pin->GetOrAddReal("problem", "x_center",
                                          0.5 * (pmy_mesh->mesh_size.x1min
                                               + pmy_mesh->mesh_size.x1max));
  const Real y_center = pin->GetOrAddReal("problem", "y_center",
                                          0.5 * (pmy_mesh->mesh_size.x2min
                                               + pmy_mesh->mesh_size.x2max));
  const Real r_inner = pin->GetOrAddReal("problem", "r_inner", 0.8);
  const Real r_outer = pin->GetOrAddReal("problem", "r_outer", 1.0);
  const Real rho_inner = pin->GetOrAddReal("problem", "rho_inner", 1.0e-2);
  const Real p_inner = pin->GetOrAddReal("problem", "p_inner", 1.0);
  const Real rho_outer = pin->GetOrAddReal("problem", "rho_outer", 1.0e-4);
  const Real p_outer = pin->GetOrAddReal("problem", "p_outer", 3.0e-5);
  const Real bx0 = pin->GetOrAddReal("problem", "bx0", 0.1);
  const Real by0 = pin->GetOrAddReal("problem", "by0", 0.0);
  const Real bz0 = pin->GetOrAddReal("problem", "bz0", 0.0);

  if (r_outer <= r_inner) {
    std::stringstream msg;
    msg << "### FATAL ERROR in srmhd_cyl_blast.cpp ProblemGenerator" << std::endl
        << "Need r_outer > r_inner for the transition layer." << std::endl;
    ATHENA_ERROR(msg);
  }

  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je; ++j) {
      for (int i = is; i <= ie; ++i) {
        const Real x = pcoord->x1v(i) - x_center;
        const Real y = pcoord->x2v(j) - y_center;
        const Real radius = std::sqrt(SQR(x) + SQR(y));

        phydro->w(IDN, k, j, i) = phydro->w1(IDN, k, j, i) =
            SmoothLogProfile(radius, r_inner, r_outer, rho_inner, rho_outer);
        phydro->w(IPR, k, j, i) = phydro->w1(IPR, k, j, i) =
            SmoothLogProfile(radius, r_inner, r_outer, p_inner, p_outer);
        phydro->w(IVX, k, j, i) = phydro->w1(IVX, k, j, i) = 0.0;
        phydro->w(IVY, k, j, i) = phydro->w1(IVY, k, j, i) = 0.0;
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
