//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file gr_tov.cpp
//! \brief Problem generator: TOV neutron star in Cowling approximation.
//!
//! Static spacetime supplied via Mesh::EnrollUserMetric.  Solves the TOV ODEs
//! with RK4 in Schwarzschild radial coords and rescales to isotropic radius
//! at the surface.  Inside R_iso the metric and matter come from the table;
//! outside, the matter is at the floor and the metric is the Schwarzschild
//! exterior in isotropic form
//!     ds^2 = -alpha^2 dt^2 + psi^4 (dx^2 + dy^2 + dz^2)
//!     alpha_ext(r) = (2r - M) / (2r + M),   psi_ext^4(r) = (1 + M/(2r))^4
//!
//! Polytropic Gamma-law closure: p = k_adi * rho^gamma.  Use --coord=gr_user
//! and --prob=gr_tov.  The pgen is purely hydro; magnetic seeding can be added
//! later.
//!
//! Ported from gr-athena/src/pgen/gr_tov.cpp (Z4c / USETM / M1 / boost branches
//! removed).  ODE solver and surface matching are kept faithful to that source.

// C++ headers
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../bvals/bvals.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"

#if not GENERAL_RELATIVITY
#error "This problem generator must be used with general relativity"
#endif

namespace {

// TOV ODE state indices
enum {TOV_IRHO, TOV_IMASS, TOV_IPHI, TOV_IINT, TOV_NVAR};

// 1D table column indices
enum {itov_rsch, itov_riso, itov_rho, itov_mass, itov_phi,
      itov_pre, itov_psi4, itov_lapse, itov_nv};

struct TOVData {
  int npts;          // number of stored radial samples
  int interp_npts;
  Real interp_dr;
  Real surf_dr;
  Real lapse_0, psi4_0;   // regularised values at r=0
  Real *data[itov_nv];
  Real R, Riso, M;
};

TOVData *tov = nullptr;

// EOS parameters (Gamma-law)
Real gamma_adi;
Real k_adi;
Real rho_zero;     // surface density used to terminate the ODE

// Magnetic seeding parameters (only used when MAGNETIC_FIELDS_ENABLED).  These
// mirror gr-athena++/src/pgen/gr_tov.cpp::SeedMagneticFields exactly: the
// vector potential is
//   A_x = -y * b_amp * max(p - p_cut_abs, 0) * (1 - rho/rho_max)^magindex
//   A_y =  x * b_amp * max(p - p_cut_abs, 0) * (1 - rho/rho_max)^magindex
//   A_z =  0
// where p_cut_abs = pcut * p_max (pcut is a fraction of the central pressure)
// and rho_max is the central density read from the TOV table.  This produces
// a buried, axisymmetric poloidal field that vanishes outside the star.
//
// We discretise via Stokes' theorem (B_face = circulation_of_A / face_area)
// rather than gr-athena's CC-curl-then-average shortcut: athena-efl's CT
// scheme requires div(B) = 0 to machine precision at startup, otherwise
// monopoles grow during evolution.  The analytic A is identical; only the
// face-B construction differs.
Real b_amp = 0.0;
Real p_cut_abs = 0.0;
Real rho_max_tov = 0.0;
int  magindex = 1;

// Forward decls
int  TOV_rhs(Real r, Real *u, Real *k);
int  TOV_solve(Real rhoc, Real rmin, Real dr, int *npts);
int  interp_locate(Real *x, int Nx, Real xval);
void interp_lag4(Real *f, Real *x, int Nx, Real xv,
                 Real *fv_p, Real *dfv_p, Real *ddfv_p);

// Vector potential helpers — same analytic A as gr-athena, point-evaluated.
// Returns A_1, A_2, A_3 (covariant Cartesian components).
void TOVVectorPotential(Real x, Real y, Real z,
                        Real *a1, Real *a2, Real *a3);
// 3-point Simpson line integrals of A along an edge.
Real IntegratedA1(Real x_lo, Real x_hi, Real y, Real z);
Real IntegratedA2(Real x, Real y_lo, Real y_hi, Real z);
Real IntegratedA3(Real x, Real y, Real z_lo, Real z_hi);

// History output functions (per-block reductions; Mesh::OutputHistory does
// the global Allreduce with the operation specified at enrolment time).
Real HistRhoMax (MeshBlock *pmb, int iout);
Real HistPgasMax(MeshBlock *pmb, int iout);
Real HistVMax   (MeshBlock *pmb, int iout);

}  // namespace

// User metric callback (must be at file scope to match MetricFunc signature)
void TOVMetric(Real x1, Real x2, Real x3, ParameterInput *pin,
               AthenaArray<Real> &g, AthenaArray<Real> &g_inv,
               AthenaArray<Real> &dg_dx1, AthenaArray<Real> &dg_dx2,
               AthenaArray<Real> &dg_dx3);

//----------------------------------------------------------------------------------------
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//! \brief Solve the TOV ODEs once at startup and enrol the metric callback.

void Mesh::InitUserMeshData(ParameterInput *pin) {
  // Problem parameters
  Real rhoc = pin->GetReal("problem", "rhoc");
  Real rmin = pin->GetOrAddReal("problem", "rmin", 0.0);
  Real dr   = pin->GetReal("problem", "dr");
  int  npts = pin->GetInteger("problem", "npts");

  gamma_adi = pin->GetOrAddReal("hydro", "gamma", 2.0);
  k_adi     = pin->GetOrAddReal("problem", "k_adi", 100.0);
  rho_zero  = pin->GetOrAddReal("hydro", "dfloor",
                                std::sqrt(1024.0*FLT_MIN));

  // Allocate the TOV table
  tov = new TOVData;
  tov->npts        = npts;
  tov->interp_npts = pin->GetOrAddInteger("problem", "interp_npts", npts);
  tov->interp_dr   = pin->GetOrAddReal(   "problem", "interp_dr",   dr);
  tov->surf_dr     = pin->GetOrAddReal(   "problem", "surf_dr",     dr / 1.0e3);

  for (int v = 0; v < itov_nv; v++) {
    tov->data[v] = static_cast<Real*>(
        std::malloc(tov->interp_npts * sizeof(Real)));
  }

  TOV_solve(rhoc, rmin, dr, &npts);

  // Magnetic seeding: read only when MAGNETIC_FIELDS_ENABLED.  Defaults make
  // b_amp = 0 effectively (no field) so the same input file works without -b.
  if (MAGNETIC_FIELDS_ENABLED) {
    b_amp        = pin->GetOrAddReal("problem", "b_amp",   0.0);
    Real pcut    = pin->GetOrAddReal("problem", "pcut",    0.04);
    magindex     = pin->GetOrAddInteger("problem", "magindex", 2);
    rho_max_tov  = tov->data[itov_rho][0];                         // central
    Real p_max   = k_adi * std::pow(rho_max_tov, gamma_adi);       // central
    p_cut_abs    = pcut * p_max;
    if (Globals::my_rank == 0) {
      std::printf("TOV B-field seeding:\n");
      std::printf("  rho_max = %.6e  p_max = %.6e\n", rho_max_tov, p_max);
      std::printf("  pcut    = %.6e (fraction)  ->  p_cut_abs = %.6e\n", pcut, p_cut_abs);
      std::printf("  magindex= %d  b_amp = %.6e\n", magindex, b_amp);
    }
  }

  // Enrol the static spacetime
  EnrollUserMetric(TOVMetric);

  // No user boundary enrolment.  The input file uses ix*_bc/ox*_bc =
  // outflow on every face — this matches gr-athena++'s gr_sommerfeld
  // boundary, which for cell-centred hydro variables is identical to
  // plain zero-gradient outflow (see gr-athena/src/bvals/cc/gr_sommerfeld_cc.cpp).
  // Athena-efl's stock OutflowInner/OuterX{1,2,3} write to the correct
  // ghost zones; user functions are not required.

  // TOV-specific history diagnostics, on top of the default columns
  // (time, dt, mass, momentum, KE, tot-E):
  //   rho_max  = central density rho_c — the standard F-mode tracker
  //   pgas_max = peak pressure
  //   vmax     = max |util^i| — F-mode amplitude
  AllocateUserHistoryOutput(3);
  EnrollUserHistoryOutput(0, HistRhoMax,  "rho_max",  UserHistoryOperation::max);
  EnrollUserHistoryOutput(1, HistPgasMax, "pgas_max", UserHistoryOperation::max);
  EnrollUserHistoryOutput(2, HistVMax,    "v_max",    UserHistoryOperation::max);
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//! \brief Initialise primitives from the TOV table; convert to conserved.

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  const Real R = tov->Riso;        // isotropic stellar radius
  const Real rho_floor = peos->GetDensityFloor();
  const Real p_floor   = peos->GetPressureFloor();

  // Cold polytropic atmosphere outside the star: pressure consistent with
  // rho = rho_floor on the same EOS as the stellar interior, so there is
  // no thermal jump at the surface.  The pressure floor is taken only as
  // a robustness backstop.  Failure recovery during evolution is handled
  // by athena-efl's existing C2P (revert to prim_old on solver failure),
  // mirroring the failure-recovery semantics of gr-athena++'s PS backend.
  const Real rho_atm = rho_floor;
  const Real p_atm   = std::max(k_adi * std::pow(rho_floor, gamma_adi), p_floor);

  // Optional star displacement (defaults to origin)
  Real x0_1 = pin->GetOrAddReal("problem", "x_0_x1", 0.0);
  Real x0_2 = pin->GetOrAddReal("problem", "x_0_x2", 0.0);
  Real x0_3 = pin->GetOrAddReal("problem", "x_0_x3", 0.0);

  AthenaArray<Real> &w  = phydro->w;
  AthenaArray<Real> &w1 = phydro->w1;
  Real dummy;

  for (int k = 0; k < ncells3; ++k) {
    Real z = pcoord->x3v(k) - x0_3;
    for (int j = 0; j < ncells2; ++j) {
      Real y = pcoord->x2v(j) - x0_2;
      for (int i = 0; i < ncells1; ++i) {
        Real x = pcoord->x1v(i) - x0_1;
        Real r = std::sqrt(x*x + y*y + z*z);

        Real rho_v, p_v;
        if (r < R) {
          // Interior: interpolate rho from the TOV table; pressure from the
          // same polytrope used to construct the table.
          interp_lag4(tov->data[itov_rho], tov->data[itov_riso],
                      tov->interp_npts, r, &rho_v, &dummy, &dummy);
          if (rho_v < rho_floor) rho_v = rho_floor;
          p_v = std::max(k_adi * std::pow(rho_v, gamma_adi), p_floor);
        } else {
          rho_v = rho_atm;
          p_v   = p_atm;
        }

        // Write both w and w1: athena's mesh init runs ConservedToPrimitive
        // using w1 as the Newton-iteration initial guess (mesh.cpp:1896).
        // If w1 is left zero, the c2p solve fails on the IC and prim is
        // overwritten with prim_old (= 0) — wiping the star.
        w(IDN, k, j, i) = w1(IDN, k, j, i) = rho_v;
        w(IPR, k, j, i) = w1(IPR, k, j, i) = p_v;
        w(IVX, k, j, i) = w1(IVX, k, j, i) = 0.0;   // static star: util^i = 0
        w(IVY, k, j, i) = w1(IVY, k, j, i) = 0.0;
        w(IVZ, k, j, i) = w1(IVZ, k, j, i) = 0.0;
      }
    }
  }

#if MAGNETIC_FIELDS_ENABLED
  // ----- Magnetic seeding via vector potential + Stokes' theorem ----------
  // Same analytic A as gr-athena (axisymmetric poloidal, buried inside R).
  // Discretised via face-circulation so that div(B) = 0 to machine precision
  // at startup (required by athena++ constrained transport).
  //
  // For each face, B^i * area = sum of A · dl around the face boundary,
  // taken with the right-hand rule.  See gr_torus.cpp:440-510 for the same
  // pattern.  In our setup A_3 = 0, so any term involving an A_3 line
  // integral vanishes; the loops below evaluate them anyway for clarity.

  // Active range plus 1 ghost layer on each side, mirroring gr_torus.
  const int il_b = is - 1;
  const int iu_b = ie + 1;
  const int jl_b = js - 1;
  const int ju_b = je + 1;
  const int kl_b = ks - 1;
  const int ku_b = ke + 1;

  // x1-faces: B^1 area = a_3_2p - a_3_2m - a_2_3p + a_2_3m
  for (int k = kl_b; k <= ku_b;     ++k) {
    for (int j = jl_b; j <= ju_b;     ++j) {
      for (int i = il_b; i <= iu_b + 1; ++i) {
        Real x1 = pcoord->x1f(i);
        Real x2_m = pcoord->x2f(j);
        Real x2_p = pcoord->x2f(j+1);
        Real x3_m = pcoord->x3f(k);
        Real x3_p = pcoord->x3f(k+1);
        Real a_2_3m = IntegratedA2(x1, x2_m, x2_p, x3_m);
        Real a_2_3p = IntegratedA2(x1, x2_m, x2_p, x3_p);
        Real a_3_2m = IntegratedA3(x1, x2_m, x3_m, x3_p);
        Real a_3_2p = IntegratedA3(x1, x2_p, x3_m, x3_p);
        Real area   = pcoord->GetFace1Area(k, j, i);
        if (area > 0.0) {
          pfield->b.x1f(k, j, i) = (a_3_2p - a_3_2m - a_2_3p + a_2_3m) / area;
        } else {
          pfield->b.x1f(k, j, i) = 0.0;
        }
      }
    }
  }
  // x2-faces: B^2 area = a_1_3p - a_1_3m - a_3_1p + a_3_1m
  for (int k = kl_b; k <= ku_b;     ++k) {
    for (int j = jl_b; j <= ju_b + 1; ++j) {
      for (int i = il_b; i <= iu_b;     ++i) {
        Real x1_m = pcoord->x1f(i);
        Real x1_p = pcoord->x1f(i+1);
        Real x2 = pcoord->x2f(j);
        Real x3_m = pcoord->x3f(k);
        Real x3_p = pcoord->x3f(k+1);
        Real a_1_3m = IntegratedA1(x1_m, x1_p, x2, x3_m);
        Real a_1_3p = IntegratedA1(x1_m, x1_p, x2, x3_p);
        Real a_3_1m = IntegratedA3(x1_m, x2, x3_m, x3_p);
        Real a_3_1p = IntegratedA3(x1_p, x2, x3_m, x3_p);
        Real area   = pcoord->GetFace2Area(k, j, i);
        if (area > 0.0) {
          pfield->b.x2f(k, j, i) = (a_1_3p - a_1_3m - a_3_1p + a_3_1m) / area;
        } else {
          pfield->b.x2f(k, j, i) = 0.0;
        }
      }
    }
  }
  // x3-faces: B^3 area = a_2_1p - a_2_1m - a_1_2p + a_1_2m
  for (int k = kl_b; k <= ku_b + 1; ++k) {
    for (int j = jl_b; j <= ju_b;     ++j) {
      for (int i = il_b; i <= iu_b;     ++i) {
        Real x1_m = pcoord->x1f(i);
        Real x1_p = pcoord->x1f(i+1);
        Real x2_m = pcoord->x2f(j);
        Real x2_p = pcoord->x2f(j+1);
        Real x3 = pcoord->x3f(k);
        Real a_1_2m = IntegratedA1(x1_m, x1_p, x2_m, x3);
        Real a_1_2p = IntegratedA1(x1_m, x1_p, x2_p, x3);
        Real a_2_1m = IntegratedA2(x1_m, x2_m, x2_p, x3);
        Real a_2_1p = IntegratedA2(x1_p, x2_m, x2_p, x3);
        Real area   = pcoord->GetFace3Area(k, j, i);
        if (area > 0.0) {
          pfield->b.x3f(k, j, i) = (a_2_1p - a_2_1m - a_1_2p + a_1_2m) / area;
        } else {
          pfield->b.x3f(k, j, i) = 0.0;
        }
      }
    }
  }

  // Compute cell-centred B from face B
  pfield->CalculateCellCenteredField(pfield->b, pfield->bcc, pcoord,
                                     0, ncells1-1, 0, ncells2-1, 0, ncells3-1);
#endif  // MAGNETIC_FIELDS_ENABLED

  // Conserved variables.  In hydro builds bcc is empty (unused); in MHD
  // builds bcc has been populated above and is needed for the energy term.
  peos->PrimitiveToConserved(w, pfield->bcc, phydro->u, pcoord,
                             0, ncells1-1, 0, ncells2-1, 0, ncells3-1);
  return;
}

//----------------------------------------------------------------------------------------
//! \fn TOVMetric
//! \brief User metric: conformally-flat isotropic Cartesian.
//!
//! Inside R_iso, alpha and psi^4 come from the TOV table.  Outside, the
//! Schwarzschild exterior in isotropic coordinates is used so that the metric
//! is C^1-continuous across the surface (true for a TOV star matched at the
//! surface to its Schwarzschild exterior).

void TOVMetric(Real x1, Real x2, Real x3, ParameterInput *pin,
               AthenaArray<Real> &g, AthenaArray<Real> &g_inv,
               AthenaArray<Real> &dg_dx1, AthenaArray<Real> &dg_dx2,
               AthenaArray<Real> &dg_dx3) {
  const Real R = tov->Riso;
  const Real M = tov->M;

  Real x = x1, y = x2, z = x3;
  Real r = std::sqrt(x*x + y*y + z*z);

  Real alpha, psi4, dalpha_dr, dpsi4_dr;

  if (r < 1.0e-30) {
    // r = 0 origin: regularised values, derivatives vanish by symmetry
    alpha     = tov->lapse_0;
    psi4      = tov->psi4_0;
    dalpha_dr = 0.0;
    dpsi4_dr  = 0.0;
  } else if (r < R) {
    Real dummy;
    interp_lag4(tov->data[itov_lapse], tov->data[itov_riso],
                tov->interp_npts, r, &alpha, &dalpha_dr, &dummy);
    interp_lag4(tov->data[itov_psi4], tov->data[itov_riso],
                tov->interp_npts, r, &psi4, &dpsi4_dr, &dummy);
  } else {
    // Schwarzschild exterior in isotropic coords
    Real two_r = 2.0 * r;
    Real denom = two_r + M;
    alpha = (two_r - M) / denom;
    Real psi = 1.0 + M/(two_r);
    psi4 = psi*psi*psi*psi;
    // dalpha/dr = 4M / (2r + M)^2
    dalpha_dr = 4.0 * M / (denom * denom);
    // dpsi^4/dr = 4 psi^3 * (-M/(2 r^2)) = -2 M psi^3 / r^2
    dpsi4_dr = -2.0 * M * psi*psi*psi / (r*r);
  }

  // covariant g_{mu nu}
  for (int n = 0; n < NMETRIC; ++n) {
    g(n) = 0.0;
    g_inv(n) = 0.0;
    dg_dx1(n) = 0.0;
    dg_dx2(n) = 0.0;
    dg_dx3(n) = 0.0;
  }
  g(I00) = -alpha*alpha;
  g(I11) =  psi4;
  g(I22) =  psi4;
  g(I33) =  psi4;

  // contravariant g^{mu nu}
  g_inv(I00) = -1.0 / (alpha*alpha);
  g_inv(I11) =  1.0 / psi4;
  g_inv(I22) =  1.0 / psi4;
  g_inv(I33) =  1.0 / psi4;

  // Cartesian derivatives.  For f(r), df/dx_i = (df/dr)(x_i / r).  At r=0
  // these are zero; the branch above already set both df/dr to zero, so the
  // r=0 case is handled cleanly with chi_i = 0.
  Real chi1 = (r > 1.0e-30) ? (x / r) : 0.0;
  Real chi2 = (r > 1.0e-30) ? (y / r) : 0.0;
  Real chi3 = (r > 1.0e-30) ? (z / r) : 0.0;

  Real dg00_dr = -2.0 * alpha * dalpha_dr;
  Real dgii_dr =  dpsi4_dr;

  dg_dx1(I00) = dg00_dr * chi1;
  dg_dx2(I00) = dg00_dr * chi2;
  dg_dx3(I00) = dg00_dr * chi3;

  dg_dx1(I11) = dgii_dr * chi1;
  dg_dx1(I22) = dgii_dr * chi1;
  dg_dx1(I33) = dgii_dr * chi1;

  dg_dx2(I11) = dgii_dr * chi2;
  dg_dx2(I22) = dgii_dr * chi2;
  dg_dx2(I33) = dgii_dr * chi2;

  dg_dx3(I11) = dgii_dr * chi3;
  dg_dx3(I22) = dgii_dr * chi3;
  dg_dx3(I33) = dgii_dr * chi3;
  return;
}

//----------------------------------------------------------------------------------------
// Internal: TOV ODE machinery
//----------------------------------------------------------------------------------------

namespace {

int TOV_rhs(Real r, Real *u, Real *k) {
  Real rho = u[TOV_IRHO];
  Real m   = u[TOV_IMASS];
  Real I   = u[TOV_IINT];   // running integral for the isotropic radius

  Real p      = k_adi * std::pow(rho, gamma_adi);
  Real e      = rho + p / (gamma_adi - 1.0);
  Real dpdrho = gamma_adi * p / rho;

  Real num     = m + 4.0*PI*r*r*r*p;
  Real den     = r * (r - 2.0*m);
  Real dphidr  = (r == 0.0) ? 0.0 : num / den;

  Real drhodr  = -(e + p) * dphidr / dpdrho;
  Real dmdr    = 4.0 * PI * r * r * e;

  Real f       = (r > 0.0) ? std::sqrt(1.0 - 2.0*m/r) : 1.0;
  Real dIdr    = (r > 0.0) ? (1.0 - f) / (r * f)      : 0.0;

  k[TOV_IRHO]  = drhodr;
  k[TOV_IMASS] = dmdr;
  k[TOV_IPHI]  = dphidr;
  k[TOV_IINT]  = dIdr;

  int knotfinite = 0;
  for (int v = 0; v < TOV_NVAR; v++) {
    if (!std::isfinite(k[v])) knotfinite++;
  }
  return knotfinite;
}

int TOV_solve(Real rhoc, Real rmin, Real dr, int *npts) {
  std::stringstream msg;
  const int maxsize = *npts - 1;
  Real u[TOV_NVAR];

  const Real pc = k_adi * std::pow(rhoc, gamma_adi);
  const Real ec = rhoc + pc / (gamma_adi - 1.0);

  Real r = rmin;
  u[TOV_IRHO]  = rhoc;
  u[TOV_IMASS] = 4.0/3.0 * PI * ec * rmin*rmin*rmin;
  u[TOV_IPHI]  = 0.0;
  u[TOV_IINT]  = 0.0;

  if (Globals::my_rank == 0) {
    std::printf("TOV_solve: solve TOV star (only once)\n");
    std::printf("TOV_solve: dr   = %.16e\n", dr);
    std::printf("TOV_solve: npts_max = %d\n", maxsize);
    std::printf("TOV_solve: rhoc = %.16e\n", rhoc);
    std::printf("TOV_solve: ec   = %.16e\n", ec);
    std::printf("TOV_solve: pc   = %.16e\n", pc);
  }

  Real rhoo = rhoc;
  int stop = 0;
  int n = 0;

  int interp_n = 1;
  const int interp_maxsize = tov->interp_npts - 1;

  // Store r=rmin sample
  tov->data[itov_rsch][0] = r;
  tov->data[itov_rho][0]  = u[TOV_IRHO];
  tov->data[itov_mass][0] = u[TOV_IMASS];
  tov->data[itov_phi][0]  = u[TOV_IPHI];
  tov->data[itov_riso][0] = r * std::exp(u[TOV_IINT]);   // rescaled below

  bool tol_surf_achieved = false;

  // Classical RK4 with dr-halving near the surface
  Real ut[TOV_NVAR];
  Real k1[TOV_NVAR], k2[TOV_NVAR], k3[TOV_NVAR], k4[TOV_NVAR];
  const Real c2 = 0.5,  a21 = 0.5;
  const Real c3 = 0.5,  a32 = 0.5;
  const Real c4 = 1.0,  a43 = 1.0;
  const Real b1 = 1.0/6.0, b2 = 1.0/3.0, b3 = 1.0/3.0, b4 = 1.0/6.0;

  while (n < maxsize) {
    for (int v = 0; v < TOV_NVAR; v++) ut[v] = u[v];
    stop += TOV_rhs(r, ut, k1);

    for (int v = 0; v < TOV_NVAR; v++) ut[v] = u[v] + a21 * k1[v] * dr;
    stop += TOV_rhs(r + c2*dr, ut, k2);

    for (int v = 0; v < TOV_NVAR; v++) ut[v] = u[v] + a32 * k2[v] * dr;
    stop += TOV_rhs(r + c3*dr, ut, k3);

    for (int v = 0; v < TOV_NVAR; v++) ut[v] = u[v] + a43 * k3[v] * dr;
    stop += TOV_rhs(r + c4*dr, ut, k4);

    for (int v = 0; v < TOV_NVAR; v++) {
      u[v] += dr * (b1*k1[v] + b2*k2[v] + b3*k3[v] + b4*k4[v]);
    }

    if (stop) {
      msg << "### FATAL ERROR in [TOV_solve]\nTOV r.h.s. not finite";
      ATHENA_ERROR(msg);
    }

    rhoo = u[TOV_IRHO];
    bool exceeded = (rhoo < rho_zero);

    if (exceeded) {
      // revert last step
      for (int v = 0; v < TOV_NVAR; v++) {
        u[v] -= dr * (b1*k1[v] + b2*k2[v] + b3*k3[v] + b4*k4[v]);
      }
      if (dr < tov->surf_dr) {
        tol_surf_achieved = true;
      } else {
        dr *= 0.5;
      }
    } else {
      n++;
      r += dr;
    }

    if ((r >= interp_n * tov->interp_dr) || tol_surf_achieved) {
      tov->data[itov_rsch][interp_n] = r;
      tov->data[itov_rho][interp_n]  = u[TOV_IRHO];
      tov->data[itov_mass][interp_n] = u[TOV_IMASS];
      tov->data[itov_phi][interp_n]  = u[TOV_IPHI];
      tov->data[itov_riso][interp_n] = r * std::exp(u[TOV_IINT]);
      interp_n++;
      if (interp_n > interp_maxsize - 1) {
        msg << "### FATAL ERROR in [TOV_solve]\ninterp_n exhausted at r=" << r;
        ATHENA_ERROR(msg);
      }
    }

    if (tol_surf_achieved) break;
  }

  if (Globals::my_rank == 0) std::printf("TOV_solve: integ. done\n");
  if (n >= maxsize) {
    msg << "### FATAL ERROR in [TOV_solve]\nStar radius not reached. (Try increasing 'npts')";
    ATHENA_ERROR(msg);
  }

  *npts = n;
  tov->npts        = n;
  tov->interp_npts = interp_n;

  tov->R = tov->data[itov_rsch][tov->interp_npts - 1];
  tov->M = tov->data[itov_mass][tov->interp_npts - 1];

  // Trim allocations to what was actually used
  for (int v = 0; v < itov_nv; v++) {
    tov->data[v] = static_cast<Real*>(
        std::realloc(tov->data[v], tov->interp_npts * sizeof(Real)));
  }

  // Match interior to Schwarzschild exterior at the surface
  const Real phiR  = tov->data[itov_phi][tov->interp_npts - 1];
  const Real IR    = std::log(tov->data[itov_riso][tov->interp_npts - 1] / tov->R);
  const Real phiRa = 0.5 * std::log(1.0 - 2.0*tov->M / tov->R);
  const Real C     = 1.0 / (2.0 * tov->R)
                     * (std::sqrt(tov->R*tov->R - 2.0*tov->M*tov->R)
                        + tov->R - tov->M)
                     * std::exp(-IR);

  for (int idx = 0; idx < tov->interp_npts; idx++) {
    tov->data[itov_phi][idx]  += -phiR + phiRa;
    tov->data[itov_riso][idx] *= C;
  }
  tov->Riso = tov->data[itov_riso][tov->interp_npts - 1];

  for (int idx = 0; idx < tov->interp_npts; idx++) {
    tov->data[itov_pre][idx]   = k_adi * std::pow(tov->data[itov_rho][idx], gamma_adi);
    tov->data[itov_psi4][idx]  = std::pow(tov->data[itov_rsch][idx]
                                          / tov->data[itov_riso][idx], 2);
    tov->data[itov_lapse][idx] = std::exp(tov->data[itov_phi][idx]);
  }

  // Regular origin values
  tov->lapse_0 = std::exp(-phiR + phiRa);
  tov->psi4_0  = 1.0 / (C*C);

  if (Globals::my_rank == 0) {
    std::printf("TOV_solve: npts        = %d\n", tov->npts);
    std::printf("TOV_solve: interp_npts = %d\n", tov->interp_npts);
    std::printf("TOV_solve: R(sch)      = %.16e\n", tov->R);
    std::printf("TOV_solve: R(iso)      = %.16e\n", tov->Riso);
    std::printf("TOV_solve: M           = %.16e\n", tov->M);
    std::printf("TOV_solve: lapse(0)    = %.16e\n", tov->lapse_0);
    std::printf("TOV_solve: psi4(0)     = %.16e\n", tov->psi4_0);
    std::printf("TOV_solve: lapse(R)    = %.16e\n",
                tov->data[itov_lapse][tov->interp_npts - 1]);
    std::printf("TOV_solve: psi4(R)     = %.16e\n",
                tov->data[itov_psi4][tov->interp_npts - 1]);
  }
  return 0;
}

int interp_locate(Real *x, int Nx, Real xval) {
  int ju, jm, jl;
  jl = -1;
  ju = Nx;
  if (xval <= x[0])      return 0;
  if (xval >= x[Nx-1])   return Nx - 1;
  const bool ascnd = (x[Nx-1] >= x[0]);
  while (ju - jl > 1) {
    jm = (ju + jl) >> 1;
    if ((xval >= x[jm]) == ascnd) jl = jm;
    else                           ju = jm;
  }
  return jl;
}

void interp_lag4(Real *f, Real *x, int Nx, Real xv,
                 Real *fv_p, Real *dfv_p, Real *ddfv_p) {
  int i = interp_locate(x, Nx, xv);
  if (i < 1)        i = 1;
  if (i > (Nx - 3)) i = Nx - 3;
  const Real ximo = x[i-1];
  const Real xi   = x[i];
  const Real xipo = x[i+1];
  const Real xipt = x[i+2];
  const Real C1   = (f[i]   - f[i-1]) / (xi   - ximo);
  const Real C2   = (f[i+1] - f[i]  ) / (xipo - xi);
  const Real C3   = (f[i+2] - f[i+1]) / (xipt - xipo);
  const Real CC1  = (C2 - C1) / (xipo - ximo);
  const Real CC2  = (C3 - C2) / (xipt - xi);
  const Real CCC1 = (CC2 - CC1) / (xipt - ximo);
  *fv_p   = f[i-1] + (xv - ximo) * (C1 + (xv - xi) * (CC1 + CCC1*(xv - xipo)));
  *dfv_p  = C1 - (CC1 - CCC1*(xi + xipo - 2.0*xv)) * (ximo - xv)
            + (xv - xi) * (CC1 + CCC1*(xv - xipo));
  *ddfv_p = 2.0 * (CC1 - CCC1*(xi + ximo + xipo - 3.0*xv));
}

//----------------------------------------------------------------------------------------
// History reductions — return a per-block extremum; the framework does the
// global reduction with the op specified at enrolment time (max here).
// Loops are restricted to the active region [is..ie]x[js..je]x[ks..ke].

Real HistRhoMax(MeshBlock *pmb, int iout) {
  Real rho_max = 0.0;
  const AthenaArray<Real> &w = pmb->phydro->w;
  for (int k = pmb->ks; k <= pmb->ke; ++k) {
    for (int j = pmb->js; j <= pmb->je; ++j) {
      for (int i = pmb->is; i <= pmb->ie; ++i) {
        if (w(IDN, k, j, i) > rho_max) rho_max = w(IDN, k, j, i);
      }
    }
  }
  return rho_max;
}

Real HistPgasMax(MeshBlock *pmb, int iout) {
  Real p_max = 0.0;
  const AthenaArray<Real> &w = pmb->phydro->w;
  for (int k = pmb->ks; k <= pmb->ke; ++k) {
    for (int j = pmb->js; j <= pmb->je; ++j) {
      for (int i = pmb->is; i <= pmb->ie; ++i) {
        if (w(IPR, k, j, i) > p_max) p_max = w(IPR, k, j, i);
      }
    }
  }
  return p_max;
}

Real HistVMax(MeshBlock *pmb, int iout) {
  // util^i magnitude.  In athena-efl GR-hydro, IVX/IVY/IVZ store util^i
  // directly (the orthonormal-frame velocity multiplied by Lorentz factor),
  // not coordinate v^i.  The Euclidean norm here is monotonic in v,
  // sufficient for tracking F-mode growth.
  Real v_max_sq = 0.0;
  const AthenaArray<Real> &w = pmb->phydro->w;
  for (int k = pmb->ks; k <= pmb->ke; ++k) {
    for (int j = pmb->js; j <= pmb->je; ++j) {
      for (int i = pmb->is; i <= pmb->ie; ++i) {
        const Real ux = w(IVX, k, j, i);
        const Real uy = w(IVY, k, j, i);
        const Real uz = w(IVZ, k, j, i);
        const Real v2 = ux*ux + uy*uy + uz*uz;
        if (v2 > v_max_sq) v_max_sq = v2;
      }
    }
  }
  return std::sqrt(v_max_sq);
}

//----------------------------------------------------------------------------------------
// Vector potential and edge-line integrals for magnetic seeding.
//
// The analytic A is identical to gr-athena++ gr_tov.cpp:1657-1661:
//   A_x = -y * b_amp * max(p(r) - p_cut_abs, 0) * (1 - rho(r)/rho_max)^magindex
//   A_y =  x * b_amp * max(p(r) - p_cut_abs, 0) * (1 - rho(r)/rho_max)^magindex
//   A_z =  0
//
// where p(r), rho(r) come from the TOV table interpolation at r = sqrt(x^2+y^2+z^2).
// Outside the star (r >= R_iso) we set A = 0 (rho_v -> 0 and p_v -> p_cut_abs from
// below, both branches give zero amplitude).
//
// Edge integrals use 3-point Simpson, which is exact for cubic polynomials
// in the integrand and matches the order of accuracy of the surrounding
// reconstruction (PPM/WENO5z).

void TOVVectorPotential(Real x, Real y, Real z,
                        Real *a1, Real *a2, Real *a3) {
  *a3 = 0.0;
  if (b_amp == 0.0) { *a1 = 0.0; *a2 = 0.0; return; }

  const Real R = tov->Riso;
  const Real r = std::sqrt(x*x + y*y + z*z);
  if (r >= R) { *a1 = 0.0; *a2 = 0.0; return; }

  // Interpolate rho from the TOV table; p from the same polytrope used to
  // construct the table (consistent with the IC).
  Real rho_v, dummy;
  interp_lag4(tov->data[itov_rho], tov->data[itov_riso],
              tov->interp_npts, r, &rho_v, &dummy, &dummy);
  if (rho_v < 0.0) rho_v = 0.0;
  Real p_v = k_adi * std::pow(rho_v, gamma_adi);

  Real overshoot = p_v - p_cut_abs;
  if (overshoot <= 0.0) { *a1 = 0.0; *a2 = 0.0; return; }

  Real taper_base = 1.0 - rho_v / rho_max_tov;
  if (taper_base < 0.0) taper_base = 0.0;
  Real taper = std::pow(taper_base, magindex);

  Real F = b_amp * overshoot * taper;
  *a1 = -y * F;
  *a2 =  x * F;
}

// Simpson 3-point: ∫_{a}^{b} f(x) dx ≈ (b-a)/6 * (f(a) + 4 f(m) + f(b))
Real IntegratedA1(Real x_lo, Real x_hi, Real y, Real z) {
  Real a1_a, a1_m, a1_b, dummy;
  Real x_m = 0.5 * (x_lo + x_hi);
  TOVVectorPotential(x_lo, y, z, &a1_a, &dummy, &dummy);
  TOVVectorPotential(x_m,  y, z, &a1_m, &dummy, &dummy);
  TOVVectorPotential(x_hi, y, z, &a1_b, &dummy, &dummy);
  return (x_hi - x_lo) / 6.0 * (a1_a + 4.0 * a1_m + a1_b);
}

Real IntegratedA2(Real x, Real y_lo, Real y_hi, Real z) {
  Real a2_a, a2_m, a2_b, dummy;
  Real y_m = 0.5 * (y_lo + y_hi);
  TOVVectorPotential(x, y_lo, z, &dummy, &a2_a, &dummy);
  TOVVectorPotential(x, y_m,  z, &dummy, &a2_m, &dummy);
  TOVVectorPotential(x, y_hi, z, &dummy, &a2_b, &dummy);
  return (y_hi - y_lo) / 6.0 * (a2_a + 4.0 * a2_m + a2_b);
}

Real IntegratedA3(Real x, Real y, Real z_lo, Real z_hi) {
  // A_3 == 0 in this setup, so the line integral vanishes identically.
  // Kept for symmetry with the gr_torus.cpp idiom.
  return 0.0;
}

}  // namespace
