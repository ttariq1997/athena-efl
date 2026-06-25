//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file recon_weno5.cpp
//! \brief fifth-order WENO and WENO-Z reconstruction kernels for primitive variables

#include "../athena.hpp"
#include "reconstruction.hpp"

namespace {

#pragma omp declare simd
inline void rec1d_p_JS_smoothness(
    Real &b_0, Real &b_1, Real &b_2,
    const Real uimt, const Real uimo, const Real ui,
    const Real uipo, const Real uipt) {
  const Real othreeotwo = 13.0/12.0;
  b_0 = othreeotwo * SQR(uimt - 2.0*uimo + ui)
      + 0.25 * SQR(uimt - 4.0*uimo + 3.0*ui);
  b_1 = othreeotwo * SQR(uimo - 2.0*ui + uipo)
      + 0.25 * SQR(uimo - uipo);
  b_2 = othreeotwo * SQR(ui - 2.0*uipo + uipt)
      + 0.25 * SQR(3.0*ui - 4.0*uipo + uipt);
}

#pragma omp declare simd
inline void rec1d_p_weno5stencils(
    Real &u_0, Real &u_1, Real &u_2,
    const Real uimt, const Real uimo, const Real ui,
    const Real uipo, const Real uipt) {
  u_0 = (2.0*uimt - 7.0*uimo + 11.0*ui) / 6.0;
  u_1 = (-uimo + 5.0*ui + 2.0*uipo) / 6.0;
  u_2 = (2.0*ui + 5.0*uipo - uipt) / 6.0;
}

static const Real optimw[3]  = {1./10., 3./5., 3./10.};
static const Real EPSL       = 1e-40;

#pragma omp declare simd
Real rec1d_p_weno5(const Real uimt,
                   const Real uimo,
                   const Real ui,
                   const Real uipo,
                   const Real uipt) {
  Real b[3], uk[3], alpha[3], w[3];
  rec1d_p_JS_smoothness(b[0], b[1], b[2], uimt, uimo, ui, uipo, uipt);
  rec1d_p_weno5stencils(uk[0], uk[1], uk[2], uimt, uimo, ui, uipo, uipt);

  alpha[0] = optimw[0] / SQR(EPSL + b[0]);
  alpha[1] = optimw[1] / SQR(EPSL + b[1]);
  alpha[2] = optimw[2] / SQR(EPSL + b[2]);

  const Real inv_sum = 1.0 / (alpha[0] + alpha[1] + alpha[2]);
  w[0] = alpha[0] * inv_sum;
  w[1] = alpha[1] * inv_sum;
  w[2] = alpha[2] * inv_sum;
  return w[0]*uk[0] + w[1]*uk[1] + w[2]*uk[2];
}

#pragma omp declare simd
Real rec1d_p_weno5z(const Real uimt,
                    const Real uimo,
                    const Real ui,
                    const Real uipo,
                    const Real uipt) {
  Real b[3], uk[3], alpha[3], w[3];
  rec1d_p_JS_smoothness(b[0], b[1], b[2], uimt, uimo, ui, uipo, uipt);
  rec1d_p_weno5stencils(uk[0], uk[1], uk[2], uimt, uimo, ui, uipo, uipt);

  const Real tau5 = std::abs(b[0] - b[2]);
  alpha[0] = optimw[0] * (1.0 + tau5 / (b[0] + EPSL));
  alpha[1] = optimw[1] * (1.0 + tau5 / (b[1] + EPSL));
  alpha[2] = optimw[2] * (1.0 + tau5 / (b[2] + EPSL));

  const Real inv_sum = 1.0 / (alpha[0] + alpha[1] + alpha[2]);
  w[0] = alpha[0] * inv_sum;
  w[1] = alpha[1] * inv_sum;
  w[2] = alpha[2] * inv_sum;
  return w[0]*uk[0] + w[1]*uk[1] + w[2]*uk[2];
}

}  // namespace

void Reconstruction::ReconstructWeno5X1(const AthenaArray<Real> &z,
                                        AthenaArray<Real> &zl_,
                                        AthenaArray<Real> &zr_,
                                        const int n_tar,
                                        const int n_src,
                                        const int k,
                                        const int j,
                                        const int il, const int iu) {
#pragma omp simd
  for (int i = il; i <= iu; ++i) {
    const Real zimt = z(n_src, k, j, i-2);
    const Real zimo = z(n_src, k, j, i-1);
    const Real zi   = z(n_src, k, j, i);
    const Real zipo = z(n_src, k, j, i+1);
    const Real zipt = z(n_src, k, j, i+2);
    zl_(n_tar, i+1) = rec1d_p_weno5(zimt, zimo, zi, zipo, zipt);
    zr_(n_tar, i  ) = rec1d_p_weno5(zipt, zipo, zi, zimo, zimt);
  }
}

void Reconstruction::ReconstructWeno5X2(const AthenaArray<Real> &z,
                                        AthenaArray<Real> &zl_,
                                        AthenaArray<Real> &zr_,
                                        const int n_tar,
                                        const int n_src,
                                        const int k,
                                        const int j,
                                        const int il, const int iu) {
#pragma omp simd
  for (int i = il; i <= iu; ++i) {
    const Real zimt = z(n_src, k, j-2, i);
    const Real zimo = z(n_src, k, j-1, i);
    const Real zi   = z(n_src, k, j,   i);
    const Real zipo = z(n_src, k, j+1, i);
    const Real zipt = z(n_src, k, j+2, i);
    zl_(n_tar, i) = rec1d_p_weno5(zimt, zimo, zi, zipo, zipt);
    zr_(n_tar, i) = rec1d_p_weno5(zipt, zipo, zi, zimo, zimt);
  }
}

void Reconstruction::ReconstructWeno5X3(const AthenaArray<Real> &z,
                                        AthenaArray<Real> &zl_,
                                        AthenaArray<Real> &zr_,
                                        const int n_tar,
                                        const int n_src,
                                        const int k,
                                        const int j,
                                        const int il, const int iu) {
#pragma omp simd
  for (int i = il; i <= iu; ++i) {
    const Real zimt = z(n_src, k-2, j, i);
    const Real zimo = z(n_src, k-1, j, i);
    const Real zi   = z(n_src, k,   j, i);
    const Real zipo = z(n_src, k+1, j, i);
    const Real zipt = z(n_src, k+2, j, i);
    zl_(n_tar, i) = rec1d_p_weno5(zimt, zimo, zi, zipo, zipt);
    zr_(n_tar, i) = rec1d_p_weno5(zipt, zipo, zi, zimo, zimt);
  }
}

void Reconstruction::ReconstructWeno5ZX1(const AthenaArray<Real> &z,
                                         AthenaArray<Real> &zl_,
                                         AthenaArray<Real> &zr_,
                                         const int n_tar,
                                         const int n_src,
                                         const int k,
                                         const int j,
                                         const int il, const int iu) {
#pragma omp simd
  for (int i = il; i <= iu; ++i) {
    const Real zimt = z(n_src, k, j, i-2);
    const Real zimo = z(n_src, k, j, i-1);
    const Real zi   = z(n_src, k, j, i);
    const Real zipo = z(n_src, k, j, i+1);
    const Real zipt = z(n_src, k, j, i+2);
    zl_(n_tar, i+1) = rec1d_p_weno5z(zimt, zimo, zi, zipo, zipt);
    zr_(n_tar, i  ) = rec1d_p_weno5z(zipt, zipo, zi, zimo, zimt);
  }
}

void Reconstruction::ReconstructWeno5ZX2(const AthenaArray<Real> &z,
                                         AthenaArray<Real> &zl_,
                                         AthenaArray<Real> &zr_,
                                         const int n_tar,
                                         const int n_src,
                                         const int k,
                                         const int j,
                                         const int il, const int iu) {
#pragma omp simd
  for (int i = il; i <= iu; ++i) {
    const Real zimt = z(n_src, k, j-2, i);
    const Real zimo = z(n_src, k, j-1, i);
    const Real zi   = z(n_src, k, j,   i);
    const Real zipo = z(n_src, k, j+1, i);
    const Real zipt = z(n_src, k, j+2, i);
    zl_(n_tar, i) = rec1d_p_weno5z(zimt, zimo, zi, zipo, zipt);
    zr_(n_tar, i) = rec1d_p_weno5z(zipt, zipo, zi, zimo, zimt);
  }
}

void Reconstruction::ReconstructWeno5ZX3(const AthenaArray<Real> &z,
                                         AthenaArray<Real> &zl_,
                                         AthenaArray<Real> &zr_,
                                         const int n_tar,
                                         const int n_src,
                                         const int k,
                                         const int j,
                                         const int il, const int iu) {
#pragma omp simd
  for (int i = il; i <= iu; ++i) {
    const Real zimt = z(n_src, k-2, j, i);
    const Real zimo = z(n_src, k-1, j, i);
    const Real zi   = z(n_src, k,   j, i);
    const Real zipo = z(n_src, k+1, j, i);
    const Real zipt = z(n_src, k+2, j, i);
    zl_(n_tar, i) = rec1d_p_weno5z(zimt, zimo, zi, zipo, zipt);
    zr_(n_tar, i) = rec1d_p_weno5z(zipt, zipo, zi, zimo, zimt);
  }
}
