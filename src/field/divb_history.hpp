#ifndef FIELD_DIVB_HISTORY_HPP_
#define FIELD_DIVB_HISTORY_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file divb_history.hpp
//! \brief face-centered max|div(B)| diagnostic for history output

#include "../athena.hpp"

class MeshBlock;

// max |div(B)| over the block — standard Athena CT diagnostic
Real DivBMaxAbsHistory(MeshBlock *pmb, int iout);

#endif // FIELD_DIVB_HISTORY_HPP_
