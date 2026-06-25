//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file minkowski.cpp
//! \brief implements functions for Minkowski (flat) spacetime and Cartesian (t,x,y,z)
//! coordinates in a derived class of the Coordinates abstract base class.
//!
//! Notes:
//!  - coordinates: t, x, y, z
//!  - metric:
//!   \f[
//!     ds^2 = -dt^2 + dx^2 + dy^2 + dz^2
//!   \f]

// C headers

// C++ headers
#include <cmath>  // sqrt()
#include <sstream>   // for stringstream in StencilPrimToLocal* stubs
#include <stdexcept>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../defs.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "coordinates.hpp"

//----------------------------------------------------------------------------------------
//! \brief Minkowski coordinates initialization
//!
//! Notes:
//!  - coordinates: t, x, y, z
//!  - metric:
//!   \f[
//!     ds^2 = -dt^2 + dx^2 + dy^2 + dz^2
//!   \f]

void Coordinates::Initialize(ParameterInput *pin) {
  // Initialize volume-averaged coordinates and spacings: x-direction
  for (int i=il-ng; i<=iu+ng; ++i) {
    x1v(i) = 0.5 * (x1f(i) + x1f(i+1));
  }
  for (int i=il-ng; i<=iu+ng-1; ++i) {
    dx1v(i) = x1v(i+1) - x1v(i);
  }

  // Initialize volume-averaged coordinates and spacings: y-direction
  if (pmy_block->block_size.nx2 == 1) {
    x2v(jl) = 0.5 * (x2f(jl) + x2f(jl+1));
    dx2v(jl) = dx2f(jl);
  } else {
    for (int j=jl-ng; j<=ju+ng; ++j) {
      x2v(j) = 0.5 * (x2f(j) + x2f(j+1));
    }
    for (int j=jl-ng; j<=ju+ng-1; ++j) {
      dx2v(j) = x2v(j+1) - x2v(j);
    }
  }

  // Initialize volume-averaged coordinates and spacings: z-direction
  if (pmy_block->block_size.nx3 == 1) {
    x3v(kl) = 0.5 * (x3f(kl) + x3f(kl+1));
    dx3v(kl) = dx3f(kl);
  } else {
    for (int k=kl-ng; k<=ku+ng; ++k) {
      x3v(k) = 0.5 * (x3f(k) + x3f(k+1));
    }
    for (int k=kl-ng; k<=ku+ng-1; ++k) {
      dx3v(k) = x3v(k+1) - x3v(k);
    }
  }

  // Initialize area-averaged coordinates used with MHD AMR
  if (pmy_block->pmy_mesh->multilevel && MAGNETIC_FIELDS_ENABLED) {
    for (int i=il-ng; i<=iu+ng; ++i) {
      x1s2(i) = x1s3(i) = x1v(i);
    }
    if (pmy_block->block_size.nx2 == 1) {
      x2s1(jl) = x2s3(jl) = x2v(jl);
    } else {
      for (int j=jl-ng; j<=ju+ng; ++j) {
        x2s1(j) = x2s3(j) = x2v(j);
      }
    }
    if (pmy_block->block_size.nx3 == 1) {
      x3s1(kl) = x3s2(kl) = x3v(kl);
    } else {
      for (int k=kl-ng; k<=ku+ng; ++k) {
        x3s1(k) = x3s2(k) = x3v(k);
      }
    }
  }
}


//----------------------------------------------------------------------------------------
// Function for computing cell-centered metric coefficients
// Inputs:
//   k,j: z- and y-indices
//   il,iu: x-index bounds
// Outputs:
//   g: array of metric components in 1D
//   g_inv: array of inverse metric components in 1D

void Coordinates::CellMetric(const int k, const int j, const int il, const int iu,
                           AthenaArray<Real> &g, AthenaArray<Real> &g_inv) {
#pragma omp simd
  for (int i=il; i<=iu; ++i) {
    g(I00,i) = -1.0;
    g(I11,i) = 1.0;
    g(I22,i) = 1.0;
    g(I33,i) = 1.0;
    g_inv(I00,i) = -1.0;
    g_inv(I11,i) = 1.0;
    g_inv(I22,i) = 1.0;
    g_inv(I33,i) = 1.0;
  }
  return;
}

//----------------------------------------------------------------------------------------
// Functions for computing face-centered metric coefficients
// Inputs:
//   k,j: z- and y-indices
//   il,iu: x-index bounds
// Outputs:
//   g: array of metric components in 1D
//   g_inv: array of inverse metric components in 1D

void Coordinates::Face1Metric(const int k, const int j, const int il, const int iu,
                            AthenaArray<Real> &g, AthenaArray<Real> &g_inv) {
#pragma omp simd
  for (int i=il; i<=iu; ++i) {
    g(I00,i) = -1.0;
    g(I11,i) = 1.0;
    g(I22,i) = 1.0;
    g(I33,i) = 1.0;
    g_inv(I00,i) = -1.0;
    g_inv(I11,i) = 1.0;
    g_inv(I22,i) = 1.0;
    g_inv(I33,i) = 1.0;
  }
  return;
}

void Coordinates::Face2Metric(const int k, const int j, const int il, const int iu,
                            AthenaArray<Real> &g, AthenaArray<Real> &g_inv) {
#pragma omp simd
  for (int i=il; i<=iu; ++i) {
    g(I00,i) = -1.0;
    g(I11,i) = 1.0;
    g(I22,i) = 1.0;
    g(I33,i) = 1.0;
    g_inv(I00,i) = -1.0;
    g_inv(I11,i) = 1.0;
    g_inv(I22,i) = 1.0;
    g_inv(I33,i) = 1.0;
  }
  return;
}

void Coordinates::Face3Metric(const int k, const int j, const int il, const int iu,
                            AthenaArray<Real> &g, AthenaArray<Real> &g_inv) {
#pragma omp simd
  for (int i=il; i<=iu; ++i) {
    g(I00,i) = -1.0;
    g(I11,i) = 1.0;
    g(I22,i) = 1.0;
    g(I33,i) = 1.0;
    g_inv(I00,i) = -1.0;
    g_inv(I11,i) = 1.0;
    g_inv(I22,i) = 1.0;
    g_inv(I33,i) = 1.0;
  }
  return;
}

//----------------------------------------------------------------------------------------
// Functions for transforming face-centered primitives to locally flat frame
// Inputs
//   k,j: z- and y-indices
//   il,iu: x-index bounds
//   bb1: 1D array of normal components B^1 of magnetic field, in global coordinates
//   prim_l: 1D array of left primitives, using global coordinates
//   prim_r: 1D array of right primitives, using global coordinates
// Outputs:
//   prim_l: values overwritten in local coordinates
//   prim_r: values overwritten in local coordinates
//   bbx: 1D array of normal magnetic fields, in local coordinates
// Notes:
//   transformation is trivial

void Coordinates::PrimToLocal1(
    const int k, const int j, const int il, const int iu,
    const AthenaArray<Real> &bb1, AthenaArray<Real> &prim_l, AthenaArray<Real> &prim_r,
    AthenaArray<Real> &bbx) {
  if (MAGNETIC_FIELDS_ENABLED) {
#pragma omp simd
    for (int i = il; i <= iu; ++i) {
      bbx(i) = bb1(i);
    }
  }
  return;
}

void Coordinates::PrimToLocal2(
    const int k, const int j, const int il, const int iu,
    const AthenaArray<Real> &bb2, AthenaArray<Real> &prim_l, AthenaArray<Real> &prim_r,
    AthenaArray<Real> &bbx) {
  if (MAGNETIC_FIELDS_ENABLED) {
#pragma omp simd
    for (int i = il; i <= iu; ++i) {
      bbx(i) = bb2(i);
    }
  }
  return;
}

void Coordinates::PrimToLocal3(
    const int k, const int j, const int il, const int iu,
    const AthenaArray<Real> &bb3, AthenaArray<Real> &prim_l, AthenaArray<Real> &prim_r,
    AthenaArray<Real> &bbx) {
  if (MAGNETIC_FIELDS_ENABLED) {
#pragma omp simd
    for (int i = il; i <= iu; ++i) {
      bbx(i) = bb3(i);
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
// Function for transforming fluxes to global frame: X-interface
// Inputs:
//   k,j: z- and y-indices
//   il,iu: x-index bounds
//   cons: 1D array of conserved quantities, using local coordinates (not used)
//   bbx: 1D array of longitudinal magnetic fields, in local coordinates (not used)
//   flux: 3D array of hydrodynamical fluxes, using local coordinates
//   ey,ez: 3D arrays of magnetic fluxes (electric fields) (not used)
// Outputs:
//   flux: values overwritten in global coordinates
// Notes:
//   transformation is trivial except for sign change from lowering time index

void Coordinates::FluxToGlobal1(const int k, const int j, const int il, const int iu,
    const AthenaArray<Real> &cons, const AthenaArray<Real> &bbx, AthenaArray<Real> &flux,
    AthenaArray<Real> &ey, AthenaArray<Real> &ez) {
#pragma omp simd
  for (int i=il; i<=iu; ++i) {
    const Real &txt = flux(IEN,k,j,i);
    Real &t10 = flux(IEN,k,j,i);
    t10 = -txt;
  }
  return;
}

void Coordinates::FluxToGlobal2(const int k, const int j, const int il, const int iu,
    const AthenaArray<Real> &cons, const AthenaArray<Real> &bbx, AthenaArray<Real> &flux,
    AthenaArray<Real> &ey, AthenaArray<Real> &ez) {
#pragma omp simd
  for (int i=il; i<=iu; ++i) {
    const Real &tyt = flux(IEN,k,j,i);
    Real &t20 = flux(IEN,k,j,i);
    t20 = -tyt;
  }
  return;
}

void Coordinates::FluxToGlobal3(const int k, const int j, const int il, const int iu,
    const AthenaArray<Real> &cons, const AthenaArray<Real> &bbx, AthenaArray<Real> &flux,
    AthenaArray<Real> &ey, AthenaArray<Real> &ez) {
#pragma omp simd
  for (int i=il; i<=iu; ++i) {
    const Real &tzt = flux(IEN,k,j,i);
    Real &t30 = flux(IEN,k,j,i);
    t30 = -tzt;
  }
  return;
}

//----------------------------------------------------------------------------------------
// Function for raising covariant components of a vector
// Inputs:
//   a_0,a_1,a_2,a_3: covariant components of vector
//   k,j,i: indices of cell in which transformation is desired
// Outputs:
//   pa0,pa1,pa2,pa3: pointers to contravariant 4-vector components

void Coordinates::RaiseVectorCell(Real a_0, Real a_1, Real a_2, Real a_3, int k, int j,
                                  int i, Real *pa0, Real *pa1, Real *pa2, Real *pa3) {
  *pa0 = -a_0;
  *pa1 = a_1;
  *pa2 = a_2;
  *pa3 = a_3;
  return;
}

//----------------------------------------------------------------------------------------
// Function for lowering contravariant components of a vector
// Inputs:
//   a0,a1,a2,a3: contravariant components of vector
//   k,j,i: indices of cell in which transformation is desired
// Outputs:
//   pa_0,pa_1,pa_2,pa_3: pointers to covariant 4-vector components

void Coordinates::LowerVectorCell(Real a0, Real a1, Real a2, Real a3, int k, int j,
                                  int i, Real *pa_0, Real *pa_1, Real *pa_2, Real *pa_3) {
  *pa_0 = -a0;
  *pa_1 = a1;
  *pa_2 = a2;
  *pa_3 = a3;
  return;
}

//----------------------------------------------------------------------------------------
// Stencil-aware tetrad transform for Minkowski. Tetrad is identity, metric is η.
// Algebraic simplification: B^(face)_tetrad = B^(face)_lab, tangential
// B^(i)_tetrad = B^(i)_lab. Direction-permuted slot layout matches the GR pgens.
//
// Bn convention follows Antón 2006: shared face-Bn for all stencil cells, cell
// tangential Bcc. This matches the GR-Kerr-Schild implementation and the LO
// path's Bn handling.

void Coordinates::StencilPrimToLocal1(
    const int k, const int j, const int i_face,
    const Real bn_face,
    const AthenaArray<Real> &prim,
    const AthenaArray<Real> &bcc,
    Real stencil_prim[6][NWAVE],
    Real stencil_bbx[6]) {
#if !MAGNETIC_FIELDS_ENABLED
  (void)bn_face; (void)bcc; (void)stencil_bbx;
#endif

#pragma omp simd
  for (int s = 0; s < 6; ++s) {
    const int ii = i_face + (s - 3);
    stencil_prim[s][IDN] = prim(IDN, k, j, ii);
    stencil_prim[s][IPR] = prim(IPR, k, j, ii);
    // Direction-1 layout: IVX = tetrad-x (face-normal), IVY = tetrad-y, IVZ = tetrad-z
    stencil_prim[s][IVX] = prim(IVX, k, j, ii);
    stencil_prim[s][IVY] = prim(IVY, k, j, ii);
    stencil_prim[s][IVZ] = prim(IVZ, k, j, ii);
#if MAGNETIC_FIELDS_ENABLED
    // Algebraic identity in Minkowski (identity tetrad):
    //   B^(face)_tetrad = γ b^(face) − util^(face) b^0 = Bn  (face value)
    //   B^(tang)_tetrad = Bcc^(tang)                       (cell value)
    stencil_prim[s][IBY] = bcc(IB2, k, j, ii);   // tang-y
    stencil_prim[s][IBZ] = bcc(IB3, k, j, ii);   // tang-z
    stencil_bbx[s]       = bn_face;              // face-normal x
#endif
  }
}

void Coordinates::StencilPrimToLocal2(
    const int k, const int j, const int i_face,
    const Real bn_face,
    const AthenaArray<Real> &prim,
    const AthenaArray<Real> &bcc,
    Real stencil_prim[6][NWAVE],
    Real stencil_bbx[6]) {
#if !MAGNETIC_FIELDS_ENABLED
  (void)bn_face; (void)bcc; (void)stencil_bbx;
#endif

#pragma omp simd
  for (int s = 0; s < 6; ++s) {
    const int jj = j + (s - 3);
    stencil_prim[s][IDN] = prim(IDN, k, jj, i_face);
    stencil_prim[s][IPR] = prim(IPR, k, jj, i_face);
    // Direction-2 layout: IVY ← tetrad-x (face-normal=y), IVZ ← tetrad-y,
    //                     IVX ← tetrad-z (z-direction goes here per dir-map)
    stencil_prim[s][IVY] = prim(IVY, k, jj, i_face);  // face-normal (y)
    stencil_prim[s][IVZ] = prim(IVZ, k, jj, i_face);  // tang-1 (z)
    stencil_prim[s][IVX] = prim(IVX, k, jj, i_face);  // tang-2 (x)
#if MAGNETIC_FIELDS_ENABLED
    stencil_prim[s][IBY] = bcc(IB3, k, jj, i_face);   // tang-1: B^z
    stencil_prim[s][IBZ] = bcc(IB1, k, jj, i_face);   // tang-2: B^x
    stencil_bbx[s]       = bn_face;                    // face-normal: B^y
#endif
  }
}

void Coordinates::StencilPrimToLocal3(
    const int k, const int j, const int i_face,
    const Real bn_face,
    const AthenaArray<Real> &prim,
    const AthenaArray<Real> &bcc,
    Real stencil_prim[6][NWAVE],
    Real stencil_bbx[6]) {
#if !MAGNETIC_FIELDS_ENABLED
  (void)bn_face; (void)bcc; (void)stencil_bbx;
#endif

#pragma omp simd
  for (int s = 0; s < 6; ++s) {
    const int kk = k + (s - 3);
    stencil_prim[s][IDN] = prim(IDN, kk, j, i_face);
    stencil_prim[s][IPR] = prim(IPR, kk, j, i_face);
    // Direction-3 layout: IVZ ← tetrad-x (face-normal=z), IVX ← tetrad-y (x),
    //                     IVY ← tetrad-z (y)
    stencil_prim[s][IVZ] = prim(IVZ, kk, j, i_face);  // face-normal (z)
    stencil_prim[s][IVX] = prim(IVX, kk, j, i_face);  // tang-1 (x)
    stencil_prim[s][IVY] = prim(IVY, kk, j, i_face);  // tang-2 (y)
#if MAGNETIC_FIELDS_ENABLED
    stencil_prim[s][IBY] = bcc(IB1, kk, j, i_face);   // tang-1: B^x
    stencil_prim[s][IBZ] = bcc(IB2, kk, j, i_face);   // tang-2: B^y
    stencil_bbx[s]       = bn_face;                    // face-normal: B^z
#endif
  }
}

