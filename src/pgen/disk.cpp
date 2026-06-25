//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file disk.cpp
//! \brief HAWLEY 2000 ApJ 528, 462 model GT1/GT2 reproduction --
//!        3D global MHD constant-ell torus in a pseudo-Newtonian potential.
//!
//! Equations (ideal MHD, eqs. 1--4 of Hawley 2000) in spherical-polar
//! coordinates (mathematically equivalent to Hawley's cylindrical).
//!
//! Gravitational potential: Paczynski-Wiita pseudo-Newtonian (eq. 5),
//!     Phi(r) = -GM / (r - r_g)
//! mimicking Schwarzschild ISCO; implemented as an EnrollUserExplicit
//! source term (built-in PointMass disabled by setting GM=0 in <problem>).
//!
//! Keplerian specific angular momentum at R_Kep (eq. 6):
//!     ell_K = sqrt(GM R_Kep) * R_Kep / (R_Kep - r_g)
//! For q=2 (constant-ell torus), ell_0 = ell_K(R_Kep); v_phi = ell_0 / R.
//!
//! Density from polytropic Bernoulli (eq. 7, q=2 case):
//!     Gamma K rho^(Gamma-1) / (Gamma - 1)  =  C - Phi_eff
//!     Phi_eff(R, z) = -GM/(r - r_g) + ell_0^2/(2 R^2),  r = sqrt(R^2+z^2)
//! C set so rho = 0 at the torus inner edge R_in (midplane);
//! K set so rho = rho_max at the pressure maximum (R_Kep, 0).
//!
//! Closed poloidal seed (Hawley 2000 eq. discussed below Table 1):
//!     A_phi = b_norm * max(rho(R, z) - rho_cut, 0)
//! No radial window needed -- the torus closes A_phi naturally.
//! b_norm auto-calibrated so global beta_min = beta_target.
//!
//! References:
//!   Hawley 2000 ApJ 528, 462 (GT1/GT2 setup, this implementation target)
//!   Paczynski & Wiita 1980 A&A 88, 23 (pseudo-Newtonian potential)
//!   Papaloizou & Pringle 1984 MNRAS 208, 721 (constant-ell torus equilibrium)

// C headers

// C++ headers
#include <algorithm>  // min
#include <cmath>      // sqrt
#include <cstdint>    // int64_t
#include <cstdlib>    // srand
#include <cstring>    // strcmp()
#include <fstream>
#include <iostream>   // endl
#include <limits>
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()

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
#include "../orbital_advection/orbital_advection.hpp"
#include "../parameter_input.hpp"
#include "../scalars/scalars.hpp"
#include "../utils/utils.hpp"

namespace {
void GetCylCoord(Coordinates *pco,Real &rad,Real &phi,Real &z,int i,int j,int k);
Real DenProfileCyl(const Real rad, const Real phi, const Real z);
Real PoverR(const Real rad, const Real phi, const Real z);
Real VelProfileCyl(const Real rad, const Real phi, const Real z);
Real VectorPotentialPhi(const Real rad, const Real z);
// Hawley 2000 GT1/GT2 parameters
Real gm0_pn;       // GM for pseudo-Newtonian source (and torus IC)
Real r_g;          // gravitational radius in Paczynski-Wiita potential
Real R_Kep;        // pressure maximum (Keplerian L set here)
Real R_in;         // inner edge of torus (rho = 0 at midplane here)
Real rho_max;      // density at the pressure maximum
Real ell0;         // specific angular momentum (constant, q=2 torus)
Real K_poly;       // polytropic constant: p = K_poly * rho^Gamma
Real bernoulli_C;  // Bernoulli constant of integration
Real gamma_gas;
Real dfloor;
Real Omega0;
// MRI / tracer parameters
Real b_norm, rho_cut, pert_amp;
Real beta_target;
int  seed_rng;
// Pseudo-Newtonian source function (defined later, declared here for enrollment)
void PseudoNewtonianSource(MeshBlock *pmb, const Real time, const Real dt,
                           const AthenaArray<Real> &prim,
                           const AthenaArray<Real> &prim_scalar,
                           const AthenaArray<Real> &bcc,
                           AthenaArray<Real> &cons,
                           AthenaArray<Real> &cons_scalar);
} // namespace

// User-defined boundary conditions for disk simulations
void DiskInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiskOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiskInnerX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiskOuterX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiskInnerX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiskOuterX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh);

//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//! \brief Function to initialize problem-specific data in mesh class.  Can also be used
//! to initialize variables which are global to (and therefore can be passed to) other
//! functions in this file.  Called in Mesh constructor.
//========================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin) {
  // Hawley 2000 GT1/GT2 torus parameters --------------------------------
  // Built-in PointMass disabled by setting <problem>/GM = 0 in input deck;
  // we use a separate parameter for the pseudo-Newtonian gravity.
  gm0_pn  = pin->GetReal("problem", "GM_pn");         // GM for our PN source
  r_g     = pin->GetReal("problem", "r_g");           // PN potential parameter
  R_Kep   = pin->GetReal("problem", "R_Kep");         // pressure-max radius
  R_in    = pin->GetReal("problem", "R_in");          // inner torus edge
  rho_max = pin->GetReal("problem", "rho_max");       // density at R_Kep

  // Pseudo-Newtonian Keplerian specific angular momentum at R_Kep (Hawley eq. 6)
  //   ell_K = sqrt(GM R) * R / (R - r_g)
  // For q=2 torus (constant ell): ell_0 = ell_K(R_Kep).
  if (R_Kep <= r_g) {
    std::stringstream msg;
    msg << "### FATAL ERROR: R_Kep must exceed r_g." << std::endl;
    ATHENA_ERROR(msg);
  }
  ell0 = std::sqrt(gm0_pn * R_Kep) * R_Kep / (R_Kep - r_g);

  // EOS: adiabatic (polytropic IC).  Require non-barotropic build.
  if (!NON_BAROTROPIC_EOS) {
    std::stringstream msg;
    msg << "### FATAL ERROR in disk.cpp InitUserMeshData: "
        << "Hawley 2000 torus setup requires --eos=adiabatic." << std::endl;
    ATHENA_ERROR(msg);
  }
  gamma_gas = pin->GetReal("hydro", "gamma");

  // Polytropic Bernoulli (Hawley eq. 7 with q=2):
  //   Gamma K rho^(Gamma-1) / (Gamma-1) = C - Phi_eff(R, z)
  // Conditions:
  //   (1) rho(R_in, 0) = 0    ->   C = Phi_eff(R_in, 0)
  //   (2) rho(R_Kep, 0) = rho_max
  // From (2): K = (Gamma-1) * (C - Phi_eff_max) / (Gamma rho_max^(Gamma-1))
  auto Phi_eff = [](Real R, Real z) {
    Real r = std::sqrt(R*R + z*z);
    return -gm0_pn/(r - r_g) + (ell0*ell0)/(2.0 * R*R);
  };
  bernoulli_C = Phi_eff(R_in, 0.0);
  Real Phi_eff_max = Phi_eff(R_Kep, 0.0);
  Real delta = bernoulli_C - Phi_eff_max;
  if (delta <= 0.0) {
    std::stringstream msg;
    msg << "### FATAL ERROR: Bernoulli C - Phi_eff_max = " << delta
        << " (must be > 0). Check R_Kep, R_in, r_g, GM_pn." << std::endl;
    ATHENA_ERROR(msg);
  }
  K_poly = (gamma_gas - 1.0) * delta
           / (gamma_gas * std::pow(rho_max, gamma_gas - 1.0));

  // Density / pressure floors
  Real float_min = std::numeric_limits<float>::min();
  dfloor = pin->GetOrAddReal("hydro", "dfloor", (1024 * (float_min)));

  Omega0 = pin->GetOrAddReal("orbital_advection", "Omega0", 0.0);

  // Enroll the pseudo-Newtonian gravity source term.
  EnrollUserExplicitSourceFunction(PseudoNewtonianSource);

  if (Globals::my_rank == 0) {
    std::cout << "[Hawley torus IC] R_Kep=" << R_Kep << "  R_in=" << R_in
              << "  r_g=" << r_g << "  ell_0=" << ell0
              << "  Bernoulli_C=" << bernoulli_C
              << "  K_poly=" << K_poly << std::endl;
  }

  // MRI / tracer parameters ---------------------------------------------
  //  b_norm:   overall amplitude on A_phi.  Ignored when beta_target > 0.
  //  beta_target: auto-calibrate b_norm so global beta_min = beta_target.
  //  rho_cut:  density threshold below which A_phi = 0
  //            (Hawley 2000 default: ~0.5 * rho_max).
  //  pert_amp: amplitude of random vel perturbation / local cs.
  //  seed_rng: RNG seed offset (per-block seed = -1 - seed_rng - gid).
  b_norm      = pin->GetOrAddReal("problem", "b_norm",      0.0);
  beta_target = pin->GetOrAddReal("problem", "beta_target", 0.0);
  rho_cut     = pin->GetOrAddReal("problem", "rho_cut",     0.5);
  pert_amp    = pin->GetOrAddReal("problem", "pert_amp",    0.0);
  seed_rng    = pin->GetOrAddInteger("problem", "seed_rng", 1);

#if MAGNETIC_FIELDS_ENABLED
  if (beta_target > 0.0) {
    // -----------------------------------------------------------------
    // Auto-calibrate b_norm: scan the analytic IC to find where beta is
    // minimum (= field is strongest relative to pressure), then solve
    // for the b_norm that puts beta_min exactly at beta_target.
    //
    // The IC is axisymmetric, so the scan is 2-D (r, theta) only.
    // For each scan point we compute B at full b_norm = 1 via centred
    // finite differences of A_phi (= curl A), then beta = 2p/B^2.
    // Since B is linear in b_norm, beta scales as 1/b_norm^2, so:
    //     beta_target = beta_min(b_norm=1) / b_norm^2
    //  -> b_norm     = sqrt(beta_min(b_norm=1) / beta_target).
    // -----------------------------------------------------------------
    const Real r_min  = pin->GetReal("mesh", "x1min");
    const Real r_max  = pin->GetReal("mesh", "x1max");
    const Real th_min = pin->GetReal("mesh", "x2min");
    const Real th_max = pin->GetReal("mesh", "x2max");

    const int Nr  = 400;
    const int Nth = 200;
    const Real dr_fd  = (r_max  - r_min ) / Real(Nr  * 8);
    const Real dth_fd = (th_max - th_min) / Real(Nth * 8);

    Real beta_min_scan = std::numeric_limits<Real>::max();

    for (int ir = 1; ir < Nr; ++ir) {
      Real r_c = r_min + (r_max - r_min) * Real(ir) / Real(Nr);
      Real dr  = dr_fd;
      if (r_c - dr < r_min) dr = (r_c - r_min) * 0.5;
      if (r_c + dr > r_max) dr = (r_max - r_c) * 0.5;
      if (dr <= 0.0) continue;

      for (int ith = 1; ith < Nth; ++ith) {
        Real th_c = th_min + (th_max - th_min) * Real(ith) / Real(Nth);
        Real dth  = dth_fd;
        if (th_c - dth < th_min) dth = (th_c - th_min) * 0.5;
        if (th_c + dth > th_max) dth = (th_max - th_c) * 0.5;
        if (dth <= 0.0) continue;

        // A_phi at 4 stencil points around (r_c, th_c)
        Real Aphi_rp = VectorPotentialPhi((r_c+dr)*std::sin(th_c),
                                          (r_c+dr)*std::cos(th_c));
        Real Aphi_rm = VectorPotentialPhi((r_c-dr)*std::sin(th_c),
                                          (r_c-dr)*std::cos(th_c));
        Real Aphi_tp = VectorPotentialPhi(r_c*std::sin(th_c+dth),
                                          r_c*std::cos(th_c+dth));
        Real Aphi_tm = VectorPotentialPhi(r_c*std::sin(th_c-dth),
                                          r_c*std::cos(th_c-dth));

        // B at b_norm = 1 (VectorPotentialPhi returns unscaled A_phi)
        //   B^r = (1/(r sin th)) d_th( sin th * A_phi )
        //   B^th = -(1/r) d_r( r * A_phi )
        Real sin_th = std::sin(th_c);
        if (sin_th < 1e-12) continue;
        Real Br = (std::sin(th_c+dth)*Aphi_tp - std::sin(th_c-dth)*Aphi_tm)
                  / (2.0 * dth * r_c * sin_th);
        Real Bth = -((r_c+dr)*Aphi_rp - (r_c-dr)*Aphi_rm)
                   / (2.0 * dr * r_c);
        Real B2 = Br*Br + Bth*Bth;
        if (B2 < 1.0e-30) continue;

        // Local pressure
        Real R_cyl = r_c * sin_th;
        Real z_loc = r_c * std::cos(th_c);
        Real rho = DenProfileCyl(R_cyl, 0.0, z_loc);
        Real p   = PoverR(R_cyl, 0.0, z_loc) * rho;
        if (p < 1.0e-30) continue;

        Real beta = 2.0 * p / B2;            // at b_norm = 1
        if (beta < beta_min_scan) beta_min_scan = beta;
      }
    }

    if (beta_min_scan < std::numeric_limits<Real>::max()) {
      Real b_norm_auto = std::sqrt(beta_min_scan / beta_target);
      if (Globals::my_rank == 0) {
        std::cout << "[disk auto-norm] beta_target = " << beta_target
                  << "  scan_beta_min(b_norm=1) = " << beta_min_scan
                  << "  ->  b_norm = " << b_norm_auto
                  << "   (overrides input b_norm = " << b_norm << ")"
                  << std::endl;
      }
      b_norm = b_norm_auto;
    } else if (Globals::my_rank == 0) {
      std::cout << "[disk auto-norm] WARNING: scan found no nonzero B^2 "
                << "in loop region; keeping input b_norm = " << b_norm
                << std::endl;
    }
  }
#endif

  // enroll user-defined boundary condition
  if (mesh_bcs[BoundaryFace::inner_x1] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::inner_x1, DiskInnerX1);
  }
  if (mesh_bcs[BoundaryFace::outer_x1] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::outer_x1, DiskOuterX1);
  }
  if (mesh_bcs[BoundaryFace::inner_x2] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::inner_x2, DiskInnerX2);
  }
  if (mesh_bcs[BoundaryFace::outer_x2] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::outer_x2, DiskOuterX2);
  }
  if (mesh_bcs[BoundaryFace::inner_x3] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::inner_x3, DiskInnerX3);
  }
  if (mesh_bcs[BoundaryFace::outer_x3] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::outer_x3, DiskOuterX3);
  }

  return;
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//! \brief Initializes Keplerian accretion disk.
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  Real rad(0.0), phi(0.0), z(0.0);
  Real den, vel;
  Real x1, x2, x3;

  OrbitalVelocityFunc &vK = porb->OrbitalVelocity;
  //  Initialize density and momenta
  for (int k=ks; k<=ke; ++k) {
    x3 = pcoord->x3v(k);
    for (int j=js; j<=je; ++j) {
      x2 = pcoord->x2v(j);
      for (int i=is; i<=ie; ++i) {
        x1 = pcoord->x1v(i);
        GetCylCoord(pcoord,rad,phi,z,i,j,k); // convert to cylindrical coordinates
        // compute initial conditions in cylindrical coordinates
        den = DenProfileCyl(rad,phi,z);
        vel = VelProfileCyl(rad,phi,z);
        if (porb->orbital_advection_defined)
          vel -= vK(porb, x1, x2, x3);
        phydro->u(IDN,k,j,i) = den;
        phydro->u(IM1,k,j,i) = 0.0;
        if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
          phydro->u(IM2,k,j,i) = den*vel;
          phydro->u(IM3,k,j,i) = 0.0;
        } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
          phydro->u(IM2,k,j,i) = 0.0;
          phydro->u(IM3,k,j,i) = den*vel;
        }

        if (NON_BAROTROPIC_EOS) {
          Real p_over_r = PoverR(rad,phi,z);
          phydro->u(IEN,k,j,i) = p_over_r*phydro->u(IDN,k,j,i)/(gamma_gas - 1.0);
          phydro->u(IEN,k,j,i) += 0.5*(SQR(phydro->u(IM1,k,j,i))+SQR(phydro->u(IM2,k,j,i))
                                       + SQR(phydro->u(IM3,k,j,i)))/phydro->u(IDN,k,j,i);
        }
      }
    }
  }

  // ------------------------------------------------------------------
  // Random velocity perturbations (seed MRI when B is present).
  // Applied only inside the disk body (rho > 0.1 * analytic profile).
  // Perturbs azimuthal component, amplitude pert_amp * local cs.
  // ------------------------------------------------------------------
  if (pert_amp > 0.0) {
    // Perturb ALL three velocity components with independent random draws
    // (Stone+1996 / Hawley+2001 convention).  Apply ONLY inside the torus
    // body, defined by rho_profile > 0.1 * rho_max (i.e. analytic profile,
    // not local density -- this excludes the atmosphere reliably).
    std::int64_t iseed = -1 - seed_rng - gid;
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=is; i<=ie; ++i) {
          Real rad_l, phi_l, z_l;
          GetCylCoord(pcoord, rad_l, phi_l, z_l, i, j, k);
          Real rho_local = phydro->u(IDN, k, j, i);
          Real rho_profile = DenProfileCyl(rad_l, phi_l, z_l);
          if (rho_profile > 0.1 * rho_max) {              // inside torus
            Real cs = std::sqrt(PoverR(rad_l, phi_l, z_l));
            Real dv1 = pert_amp * cs * (2.0 * ran2(&iseed) - 1.0);
            Real dv2 = pert_amp * cs * (2.0 * ran2(&iseed) - 1.0);
            Real dv3 = pert_amp * cs * (2.0 * ran2(&iseed) - 1.0);
            phydro->u(IM1, k, j, i) += rho_local * dv1;
            phydro->u(IM2, k, j, i) += rho_local * dv2;
            phydro->u(IM3, k, j, i) += rho_local * dv3;
          }
        }
      }
    }
  }

#if MAGNETIC_FIELDS_ENABLED
  // ------------------------------------------------------------------
  // Vector-potential magnetic field via Stokes integration.
  //   A_phi(R, z) = b_norm * max(rho(R, z) - rho_cut, 0)    (Hawley 2000)
  //   B = curl A, computed from line integrals of A_phi along phi-edges:
  //     B^1 = (1/area1) * [a_3_2p - a_3_2m]
  //     B^2 = (-1/area2) * [a_3_1p - a_3_1m]
  //     B^3 = 0  (no initial toroidal field)
  //   where a_3 = integral of A_phi along a phi-edge.
  // Only spherical_polar coords supported here (cylindrical-MRI variant TBD).
  // ------------------------------------------------------------------
  if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") != 0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in disk.cpp ProblemGenerator: MHD path only "
        << "implemented for spherical_polar coords; got "
        << COORDINATE_SYSTEM << std::endl;
    ATHENA_ERROR(msg);
  }

  // B^1 = b.x1f on constant-r faces
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie+1; ++i) {
        Real x1   = pcoord->x1f(i);
        Real x2_m = pcoord->x2f(j);
        Real x2_p = pcoord->x2f(j+1);
        Real x3_m = pcoord->x3f(k);
        Real x3_p = pcoord->x3f(k+1);
        Real aphi_m = VectorPotentialPhi(x1*std::sin(x2_m), x1*std::cos(x2_m));
        Real aphi_p = VectorPotentialPhi(x1*std::sin(x2_p), x1*std::cos(x2_p));
        Real a_3_2m = aphi_m * x1 * std::sin(x2_m) * (x3_p - x3_m);
        Real a_3_2p = aphi_p * x1 * std::sin(x2_p) * (x3_p - x3_m);
        Real area = pcoord->GetFace1Area(k, j, i);
        pfield->b.x1f(k, j, i) =
            (area > 0.0) ? b_norm * (a_3_2p - a_3_2m) / area : 0.0;
      }
    }
  }

  // B^2 = b.x2f on constant-theta faces
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je+1; ++j) {
      for (int i=is; i<=ie; ++i) {
        Real x1_m = pcoord->x1f(i);
        Real x1_p = pcoord->x1f(i+1);
        Real x2   = pcoord->x2f(j);
        Real x3_m = pcoord->x3f(k);
        Real x3_p = pcoord->x3f(k+1);
        Real aphi_m = VectorPotentialPhi(x1_m*std::sin(x2), x1_m*std::cos(x2));
        Real aphi_p = VectorPotentialPhi(x1_p*std::sin(x2), x1_p*std::cos(x2));
        Real a_3_1m = aphi_m * x1_m * std::sin(x2) * (x3_p - x3_m);
        Real a_3_1p = aphi_p * x1_p * std::sin(x2) * (x3_p - x3_m);
        Real area = pcoord->GetFace2Area(k, j, i);
        pfield->b.x2f(k, j, i) =
            (area > 0.0) ? -b_norm * (a_3_1p - a_3_1m) / area : 0.0;
      }
    }
  }

  // B^3 = 0 on phi-faces (no initial toroidal field)
  for (int k=ks; k<=ke+1; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        pfield->b.x3f(k, j, i) = 0.0;
      }
    }
  }

  // Diagnostic: compute local beta_min on this block and print only when
  // this block actually overlaps the loop region (has nonzero field).
  // Without this guard, blocks outside the loop window print confusing
  // "B^2_max=0, beta_min=DBL_MAX" lines.
  {
    Real beta_local_min = std::numeric_limits<Real>::max();
    Real b2_max = 0.0;
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=is; i<=ie; ++i) {
          Real rad_l, phi_l, z_l;
          GetCylCoord(pcoord, rad_l, phi_l, z_l, i, j, k);
          Real rho_l = phydro->u(IDN, k, j, i);
          Real p_l = PoverR(rad_l, phi_l, z_l) * rho_l;
          Real br = 0.5*(pfield->b.x1f(k,j,i)   + pfield->b.x1f(k,j,i+1));
          Real bt = 0.5*(pfield->b.x2f(k,j,i)   + pfield->b.x2f(k,j+1,i));
          Real bp = 0.5*(pfield->b.x3f(k,j,i)   + pfield->b.x3f(k+1,j,i));
          Real b2 = SQR(br) + SQR(bt) + SQR(bp);
          if (b2 > b2_max) b2_max = b2;
          if (b2 > 1e-30 && rho_l > 0.1 * DenProfileCyl(rad_l, phi_l, z_l)) {
            Real beta = 2.0 * p_l / b2;
            if (beta < beta_local_min) beta_local_min = beta;
          }
        }
      }
    }
    if (b2_max > 0.0) {
      std::cout << "[disk MHD IC] gid=" << gid
                << "  has field: B^2_max=" << b2_max
                << "  local beta_min=" << beta_local_min << std::endl;
    }
  }

  // Add magnetic energy to total energy for non-barotropic EOS.
  // (Isothermal EOS has no IEN slot; skip.)
  if (NON_BAROTROPIC_EOS) {
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=is; i<=ie; ++i) {
          Real br = 0.5*(pfield->b.x1f(k,j,i) + pfield->b.x1f(k,j,i+1));
          Real bt = 0.5*(pfield->b.x2f(k,j,i) + pfield->b.x2f(k,j+1,i));
          Real bp = 0.5*(pfield->b.x3f(k,j,i) + pfield->b.x3f(k+1,j,i));
          phydro->u(IEN, k, j, i) += 0.5 * (SQR(br) + SQR(bt) + SQR(bp));
        }
      }
    }
  }
#endif // MAGNETIC_FIELDS_ENABLED

#if NSCALARS > 0
  // ------------------------------------------------------------------
  // Passive scalar tracers, conservative form (= rho * specific value).
  //   s[0] = rho * (R_init / R_ref)         "where did this fluid start"
  //   s[1] = rho * (ell_init / ell_0)       "initial Lz per unit mass"
  // For PP torus, ell_init = ell_0 everywhere in the torus body, so s[1]
  // is essentially rho * 1.0 -- still tracks where the fluid came from
  // (any deviation from ell_0 in the saturated state is MRI transport).
  // Outside the torus body, scalars are 0.
  // ------------------------------------------------------------------
  {
    // Reference normalization: outer edge of the simulation domain.
    Real R_ref = pcoord->x1v(ie);
    Real ell_ref = std::max(ell0, 1.0e-30);

    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=is; i<=ie; ++i) {
          Real rad_l, phi_l, z_l;
          GetCylCoord(pcoord, rad_l, phi_l, z_l, i, j, k);
          Real rho_local = phydro->u(IDN, k, j, i);
          Real rho_profile = DenProfileCyl(rad_l, phi_l, z_l);
          if (rho_profile > 0.1 * rho_max) {              // inside torus
            pscalars->s(0, k, j, i) = rho_local * (rad_l / R_ref);
            if (NSCALARS > 1) {
              Real ell_l = rad_l * VelProfileCyl(rad_l, phi_l, z_l);
              pscalars->s(1, k, j, i) = rho_local * (ell_l / ell_ref);
            }
          } else {
            pscalars->s(0, k, j, i) = 0.0;
            if (NSCALARS > 1) pscalars->s(1, k, j, i) = 0.0;
          }
        }
      }
    }
  }
#endif

  return;
}

namespace {
//----------------------------------------------------------------------------------------
//! transform to cylindrical coordinate

void GetCylCoord(Coordinates *pco,Real &rad,Real &phi,Real &z,int i,int j,int k) {
  if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    rad=pco->x1v(i);
    phi=pco->x2v(j);
    z=pco->x3v(k);
  } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    rad=std::abs(pco->x1v(i)*std::sin(pco->x2v(j)));
    phi=pco->x3v(k);
    z=pco->x1v(i)*std::cos(pco->x2v(j));
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn Real DenProfileCyl(...)
//! \brief Polytropic Bernoulli density for the Hawley 2000 torus.
//!
//!   rho(R, z) = [ (Gamma-1)/(Gamma K) * (C - Phi_eff) ]^{1/(Gamma-1)}
//!   Phi_eff(R, z) = -GM/(r - r_g) + ell_0^2 / (2 R^2),  r = sqrt(R^2 + z^2)
//!
//! Outside the torus (where C < Phi_eff), returns dfloor.

Real DenProfileCyl(const Real rad, const Real phi, const Real z) {
  if (rad < 1.0e-8) return dfloor;
  Real r = std::sqrt(rad*rad + z*z);
  if (r <= r_g) return dfloor;     // inside the PN "horizon"
  Real Phi_eff = -gm0_pn/(r - r_g) + (ell0*ell0)/(2.0*rad*rad);
  Real delta = bernoulli_C - Phi_eff;
  if (delta <= 0.0) return dfloor;
  Real rho_g = std::pow((gamma_gas - 1.0) * delta / (gamma_gas * K_poly),
                        1.0/(gamma_gas - 1.0));
  return std::max(rho_g, dfloor);
}

//----------------------------------------------------------------------------------------
//! \fn Real PoverR(...)
//! \brief Polytropic p/rho = K * rho^(Gamma-1), evaluated at the analytic profile.

Real PoverR(const Real rad, const Real phi, const Real z) {
  Real rho_here = DenProfileCyl(rad, 0.0, z);
  return K_poly * std::pow(rho_here, gamma_gas - 1.0);
}

//----------------------------------------------------------------------------------------
//! \fn Real VelProfileCyl(...)
//! \brief Constant specific angular momentum (q=2 torus): v_phi = ell_0 / R.

Real VelProfileCyl(const Real rad, const Real phi, const Real z) {
  if (rad < 1.0e-8) return 0.0;
  return ell0 / rad - rad * Omega0;
}

//----------------------------------------------------------------------------------------
//! \fn PseudoNewtonianSource(...)
//! \brief Source term for the Paczynski-Wiita pseudo-Newtonian potential
//!        Phi(r) = -GM/(r - r_g),  spherically symmetric in spherical-polar.
//!
//! Force per unit mass (radial direction in spherical):
//!     F_r = -dPhi/dr = -GM / (r - r_g)^2
//! Energy update (non-barotropic): rho * F_r * v_r.

void PseudoNewtonianSource(MeshBlock *pmb, const Real time, const Real dt,
                           const AthenaArray<Real> &prim,
                           const AthenaArray<Real> &prim_scalar,
                           const AthenaArray<Real> &bcc,
                           AthenaArray<Real> &cons,
                           AthenaArray<Real> &cons_scalar) {
  for (int k = pmb->ks; k <= pmb->ke; ++k) {
    for (int j = pmb->js; j <= pmb->je; ++j) {
      for (int i = pmb->is; i <= pmb->ie; ++i) {
        Real r_sph = pmb->pcoord->x1v(i);
        if (r_sph <= r_g) continue;
        Real F_r = -gm0_pn / SQR(r_sph - r_g);
        Real rho = prim(IDN, k, j, i);
        cons(IM1, k, j, i) += dt * rho * F_r;
        if (NON_BAROTROPIC_EOS) {
          Real v_r = prim(IVX, k, j, i);
          cons(IEN, k, j, i) += dt * rho * F_r * v_r;
        }
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn Real VectorPotentialPhi(const Real rad, const Real z)
//! \brief Hawley 2000 closed-loop A_phi seed for the PP torus.
//!
//!   A_phi(R, z) = b_norm * max(rho(R, z) - rho_cut, 0)
//!
//! No radial window is needed: on the PP torus, rho falls smoothly to zero
//! on all torus boundaries, so the max(...) automatically gives a closed
//! poloidal loop with zero net flux.
//! Returns the unscaled potential; b_norm is applied in ProblemGenerator.

Real VectorPotentialPhi(const Real rad, const Real z) {
  Real rho_at_corner = DenProfileCyl(rad, 0.0, z);
  return std::max(rho_at_corner - rho_cut, 0.0);
}
} // namespace

//----------------------------------------------------------------------------------------
//! User-defined boundary Conditions: sets solution in ghost zones to initial values

void DiskInnerX1(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  Real rad(0.0), phi(0.0), z(0.0);
  Real vel;
  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;
  if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
        for (int i=1; i<=ngh; ++i) {
          GetCylCoord(pco,rad,phi,z,il-i,j,k);
          prim(IDN,k,j,il-i) = DenProfileCyl(rad,phi,z);
          vel = VelProfileCyl(rad,phi,z);
          if (pmb->porb->orbital_advection_defined)
            vel -= vK(pmb->porb, pco->x1v(il-i), pco->x2v(j), pco->x3v(k));
          prim(IM1,k,j,il-i) = 0.0;
          prim(IM2,k,j,il-i) = vel;
          prim(IM3,k,j,il-i) = 0.0;
          if (NON_BAROTROPIC_EOS)
            prim(IEN,k,j,il-i) = PoverR(rad, phi, z)*prim(IDN,k,j,il-i);
        }
      }
    }
  } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
        for (int i=1; i<=ngh; ++i) {
          GetCylCoord(pco,rad,phi,z,il-i,j,k);
          prim(IDN,k,j,il-i) = DenProfileCyl(rad,phi,z);
          vel = VelProfileCyl(rad,phi,z);
          if (pmb->porb->orbital_advection_defined)
            vel -= vK(pmb->porb, pco->x1v(il-i), pco->x2v(j), pco->x3v(k));
          prim(IM1,k,j,il-i) = 0.0;
          prim(IM2,k,j,il-i) = 0.0;
          prim(IM3,k,j,il-i) = vel;
          if (NON_BAROTROPIC_EOS)
            prim(IEN,k,j,il-i) = PoverR(rad, phi, z)*prim(IDN,k,j,il-i);
        }
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! User-defined boundary Conditions: sets solution in ghost zones to initial values

void DiskOuterX1(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  Real rad(0.0), phi(0.0), z(0.0);
  Real vel;
  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;
  if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
        for (int i=1; i<=ngh; ++i) {
          GetCylCoord(pco,rad,phi,z,iu+i,j,k);
          prim(IDN,k,j,iu+i) = DenProfileCyl(rad,phi,z);
          vel = VelProfileCyl(rad,phi,z);
          if (pmb->porb->orbital_advection_defined)
            vel -= vK(pmb->porb, pco->x1v(iu+i), pco->x2v(j), pco->x3v(k));
          prim(IM1,k,j,iu+i) = 0.0;
          prim(IM2,k,j,iu+i) = vel;
          prim(IM3,k,j,iu+i) = 0.0;
          if (NON_BAROTROPIC_EOS)
            prim(IEN,k,j,iu+i) = PoverR(rad, phi, z)*prim(IDN,k,j,iu+i);
        }
      }
    }
  } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
        for (int i=1; i<=ngh; ++i) {
          GetCylCoord(pco,rad,phi,z,iu+i,j,k);
          prim(IDN,k,j,iu+i) = DenProfileCyl(rad,phi,z);
          vel = VelProfileCyl(rad,phi,z);
          if (pmb->porb->orbital_advection_defined)
            vel -= vK(pmb->porb, pco->x1v(iu+i), pco->x2v(j), pco->x3v(k));
          prim(IM1,k,j,iu+i) = 0.0;
          prim(IM2,k,j,iu+i) = 0.0;
          prim(IM3,k,j,iu+i) = vel;
          if (NON_BAROTROPIC_EOS)
            prim(IEN,k,j,iu+i) = PoverR(rad, phi, z)*prim(IDN,k,j,iu+i);
        }
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! User-defined boundary Conditions: sets solution in ghost zones to initial values

void DiskInnerX2(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  Real rad(0.0), phi(0.0), z(0.0);
  Real vel;
  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;
  if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=1; j<=ngh; ++j) {
        for (int i=il; i<=iu; ++i) {
          GetCylCoord(pco,rad,phi,z,i,jl-j,k);
          prim(IDN,k,jl-j,i) = DenProfileCyl(rad,phi,z);
          vel = VelProfileCyl(rad,phi,z);
          if (pmb->porb->orbital_advection_defined)
            vel -= vK(pmb->porb, pco->x1v(i), pco->x2v(jl-j), pco->x3v(k));
          prim(IM1,k,jl-j,i) = 0.0;
          prim(IM2,k,jl-j,i) = vel;
          prim(IM3,k,jl-j,i) = 0.0;
          if (NON_BAROTROPIC_EOS)
            prim(IEN,k,jl-j,i) = PoverR(rad, phi, z)*prim(IDN,k,jl-j,i);
        }
      }
    }
  } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=1; j<=ngh; ++j) {
        for (int i=il; i<=iu; ++i) {
          GetCylCoord(pco,rad,phi,z,i,jl-j,k);
          prim(IDN,k,jl-j,i) = DenProfileCyl(rad,phi,z);
          vel = VelProfileCyl(rad,phi,z);
          if (pmb->porb->orbital_advection_defined)
            vel -= vK(pmb->porb, pco->x1v(i), pco->x2v(jl-j), pco->x3v(k));
          prim(IM1,k,jl-j,i) = 0.0;
          prim(IM2,k,jl-j,i) = 0.0;
          prim(IM3,k,jl-j,i) = vel;
          if (NON_BAROTROPIC_EOS)
            prim(IEN,k,jl-j,i) = PoverR(rad, phi, z)*prim(IDN,k,jl-j,i);
        }
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! User-defined boundary Conditions: sets solution in ghost zones to initial values

void DiskOuterX2(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  Real rad(0.0), phi(0.0), z(0.0);
  Real vel;
  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;
  if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=1; j<=ngh; ++j) {
        for (int i=il; i<=iu; ++i) {
          GetCylCoord(pco,rad,phi,z,i,ju+j,k);
          prim(IDN,k,ju+j,i) = DenProfileCyl(rad,phi,z);
          vel = VelProfileCyl(rad,phi,z);
          if (pmb->porb->orbital_advection_defined)
            vel -= vK(pmb->porb, pco->x1v(i), pco->x2v(ju+j), pco->x3v(k));
          prim(IM1,k,ju+j,i) = 0.0;
          prim(IM2,k,ju+j,i) = vel;
          prim(IM3,k,ju+j,i) = 0.0;
          if (NON_BAROTROPIC_EOS)
            prim(IEN,k,ju+j,i) = PoverR(rad, phi, z)*prim(IDN,k,ju+j,i);
        }
      }
    }
  } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=1; j<=ngh; ++j) {
        for (int i=il; i<=iu; ++i) {
          GetCylCoord(pco,rad,phi,z,i,ju+j,k);
          prim(IDN,k,ju+j,i) = DenProfileCyl(rad,phi,z);
          vel = VelProfileCyl(rad,phi,z);
          if (pmb->porb->orbital_advection_defined)
            vel -= vK(pmb->porb, pco->x1v(i), pco->x2v(ju+j), pco->x3v(k));
          prim(IM1,k,ju+j,i) = 0.0;
          prim(IM2,k,ju+j,i) = 0.0;
          prim(IM3,k,ju+j,i) = vel;
          if (NON_BAROTROPIC_EOS)
            prim(IEN,k,ju+j,i) = PoverR(rad, phi, z)*prim(IDN,k,ju+j,i);
        }
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! User-defined boundary Conditions: sets solution in ghost zones to initial values

void DiskInnerX3(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  Real rad(0.0), phi(0.0), z(0.0);
  Real vel;
  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;
  if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    for (int k=1; k<=ngh; ++k) {
      for (int j=jl; j<=ju; ++j) {
        for (int i=il; i<=iu; ++i) {
          GetCylCoord(pco,rad,phi,z,i,j,kl-k);
          prim(IDN,kl-k,j,i) = DenProfileCyl(rad,phi,z);
          vel = VelProfileCyl(rad,phi,z);
          if (pmb->porb->orbital_advection_defined)
            vel -= vK(pmb->porb, pco->x1v(i), pco->x2v(j), pco->x3v(kl-k));
          prim(IM1,kl-k,j,i) = 0.0;
          prim(IM2,kl-k,j,i) = vel;
          prim(IM3,kl-k,j,i) = 0.0;
          if (NON_BAROTROPIC_EOS)
            prim(IEN,kl-k,j,i) = PoverR(rad, phi, z)*prim(IDN,kl-k,j,i);
        }
      }
    }
  } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    for (int k=1; k<=ngh; ++k) {
      for (int j=jl; j<=ju; ++j) {
        for (int i=il; i<=iu; ++i) {
          GetCylCoord(pco,rad,phi,z,i,j,kl-k);
          prim(IDN,kl-k,j,i) = DenProfileCyl(rad,phi,z);
          vel = VelProfileCyl(rad,phi,z);
          if (pmb->porb->orbital_advection_defined)
            vel -= vK(pmb->porb, pco->x1v(i), pco->x2v(j), pco->x3v(kl-k));
          prim(IM1,kl-k,j,i) = 0.0;
          prim(IM2,kl-k,j,i) = 0.0;
          prim(IM3,kl-k,j,i) = vel;
          if (NON_BAROTROPIC_EOS)
            prim(IEN,kl-k,j,i) = PoverR(rad, phi, z)*prim(IDN,kl-k,j,i);
        }
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! User-defined boundary Conditions: sets solution in ghost zones to initial values

void DiskOuterX3(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  Real rad(0.0), phi(0.0), z(0.0);
  Real vel;
  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;
  if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    for (int k=1; k<=ngh; ++k) {
      for (int j=jl; j<=ju; ++j) {
        for (int i=il; i<=iu; ++i) {
          GetCylCoord(pco,rad,phi,z,i,j,ku+k);
          prim(IDN,ku+k,j,i) = DenProfileCyl(rad,phi,z);
          vel = VelProfileCyl(rad,phi,z);
          if (pmb->porb->orbital_advection_defined)
            vel -= vK(pmb->porb, pco->x1v(i), pco->x2v(j), pco->x3v(ku+k));
          prim(IM1,ku+k,j,i) = 0.0;
          prim(IM2,ku+k,j,i) = vel;
          prim(IM3,ku+k,j,i) = 0.0;
          if (NON_BAROTROPIC_EOS)
            prim(IEN,ku+k,j,i) = PoverR(rad, phi, z)*prim(IDN,ku+k,j,i);
        }
      }
    }
  } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    for (int k=1; k<=ngh; ++k) {
      for (int j=jl; j<=ju; ++j) {
        for (int i=il; i<=iu; ++i) {
          GetCylCoord(pco,rad,phi,z,i,j,ku+k);
          prim(IDN,ku+k,j,i) = DenProfileCyl(rad,phi,z);
          vel = VelProfileCyl(rad,phi,z);
          if (pmb->porb->orbital_advection_defined)
            vel -= vK(pmb->porb, pco->x1v(i), pco->x2v(j), pco->x3v(ku+k));
          prim(IM1,ku+k,j,i) = 0.0;
          prim(IM2,ku+k,j,i) = 0.0;
          prim(IM3,ku+k,j,i) = vel;
          if (NON_BAROTROPIC_EOS)
            prim(IEN,ku+k,j,i) = PoverR(rad, phi, z)*prim(IDN,ku+k,j,i);
        }
      }
    }
  }
}
