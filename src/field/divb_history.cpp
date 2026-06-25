//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file divb_history.cpp
//! \brief face-centered max|div(B)| diagnostic for history output
//!
//! Computes the maximum absolute value of the discrete divergence of B over
//! all cells in the block.  For CT this should stay at machine roundoff.
//! Consistent with the diagnostic used in Gardiner & Stone (2005, 2008)
//! and White, Stone & Gammie (2016).

#include <algorithm>
#include <cmath>

#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../mesh/mesh.hpp"
#include "field.hpp"
#include "divb_history.hpp"

Real DivBMaxAbsHistory(MeshBlock *pmb, int iout) {
  int is = pmb->is, ie = pmb->ie;
  int js = pmb->js, je = pmb->je;
  int ks = pmb->ks, ke = pmb->ke;

  AthenaArray<Real> face1, face2p, face2m, face3p, face3m;
  face1.NewAthenaArray((ie - is) + 2*NGHOST + 1);
  face2p.NewAthenaArray((ie - is) + 2*NGHOST + 1);
  face2m.NewAthenaArray((ie - is) + 2*NGHOST + 1);
  face3p.NewAthenaArray((ie - is) + 2*NGHOST + 1);
  face3m.NewAthenaArray((ie - is) + 2*NGHOST + 1);

  Field *pfield = pmb->pfield;
  Real max_abs = 0.0;

  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je; ++j) {
      pmb->pcoord->Face1Area(k,   j,   is, ie + 1, face1);
      pmb->pcoord->Face2Area(k,   j+1, is, ie,     face2p);
      pmb->pcoord->Face2Area(k,   j,   is, ie,     face2m);
      pmb->pcoord->Face3Area(k+1, j,   is, ie,     face3p);
      pmb->pcoord->Face3Area(k,   j,   is, ie,     face3m);
      for (int i = is; i <= ie; ++i) {
        Real cellvol = pmb->pcoord->GetCellVolume(k, j, i);
        Real divb = ( face1(i+1) * pfield->b.x1f(k, j, i+1)
                    - face1(i)   * pfield->b.x1f(k, j, i)
                    + face2p(i)  * pfield->b.x2f(k, j+1, i)
                    - face2m(i)  * pfield->b.x2f(k, j, i)
                    + face3p(i)  * pfield->b.x3f(k+1, j, i)
                    - face3m(i)  * pfield->b.x3f(k, j, i)) / cellvol;
        max_abs = std::max(max_abs, std::abs(divb));
      }
    }
  }

  return max_abs;
}
