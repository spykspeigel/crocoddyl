///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2022, Centro E Piaggio, Unipi
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/multibody/states/soft-multibody.hpp"
#include <pinocchio/algorithm/joint-configuration.hpp>

namespace crocoddyl {

template <typename Scalar>
StateSoftMultibodyTpl<Scalar>::StateSoftMultibodyTpl(boost::shared_ptr<PinocchioModel> model)
    : Base(model), pinocchio_(model), x0_(VectorXs::Zero(3*(model->nv)-12 + model->nq)){

  x0_.head(model->nq) = pinocchio::neutral(*pinocchio_.get());

  // Define internally the limits of the first joint

  const std::size_t nq0 = model->joints[1].nq();

  nq_m_ = model->nv-6;
  nv_m_ = model->nv-6;
  nv_l_ = model->nv;
  nq_l_ = model->nq;
  ndx_ = 4 * model->nv-12;
  nx_ = 3*(model->nv)-12 + model->nq;
  nq_ =  nq_m_ + nq_l_;
  nv_ =  nv_m_ + nv_l_;

  lb_.head(nq0) = -std::numeric_limits<Scalar>::infinity() * VectorXs::Ones(nq0);
  ub_.head(nq0) = std::numeric_limits<Scalar>::infinity() * VectorXs::Ones(nq0);
  lb_.segment(nq0, nq_l_ - nq0) = pinocchio_->lowerPositionLimit.tail(nq_l_ - nq0);
  ub_.segment(nq0, nq_l_ - nq0) = pinocchio_->upperPositionLimit.tail(nq_l_ - nq0);
  lb_.segment(nq_l_, nv_l_) = -pinocchio_->velocityLimit;
  ub_.segment(nq_l_, nv_l_) = pinocchio_->velocityLimit;
  Base::update_has_limits();
  
}

template <typename Scalar>
StateSoftMultibodyTpl<Scalar>::StateSoftMultibodyTpl() : Base(), x0_(VectorXs::Zero(0)) {}

template <typename Scalar>
StateSoftMultibodyTpl<Scalar>::~StateSoftMultibodyTpl() {}

template <typename Scalar>
typename MathBaseTpl<Scalar>::VectorXs StateSoftMultibodyTpl<Scalar>::zero() const {
  return x0_;
}

template <typename Scalar>
typename MathBaseTpl<Scalar>::VectorXs StateSoftMultibodyTpl<Scalar>::rand() const {
  VectorXs xrand = VectorXs::Random(nx_);
  xrand.head(nq_l_) = pinocchio::randomConfiguration(*pinocchio_.get());
  return xrand;
}

template <typename Scalar>
void StateSoftMultibodyTpl<Scalar>::diff(const Eigen::Ref<const VectorXs>& x0, const Eigen::Ref<const VectorXs>& x1,
                                     Eigen::Ref<VectorXs> dxout) const {
  if (static_cast<std::size_t>(x0.size()) != nx_) {
    throw_pretty("Invalid argument: "
                 << "x0 has wrong dimension (it should be " + std::to_string(nx_) + ")");
  }
  if (static_cast<std::size_t>(x1.size()) != nx_) {
    throw_pretty("Invalid argument: "
                 << "x1 has wrong dimension (it should be " + std::to_string(nx_) + ")");
  }
  if (static_cast<std::size_t>(dxout.size()) != ndx_) {
    throw_pretty("Invalid argument: "
                 << "dxout has wrong dimension (it should be " + std::to_string(ndx_) + ")");
  }

  pinocchio::difference(*pinocchio_.get(), x0.head(nq_l_), x1.head(nq_l_), dxout.head(nv_l_));
  dxout.tail(ndx_-nv_l_) = x1.tail(nx_-nq_l_) - x0.tail(nx_-nq_l_);
}

template <typename Scalar>
void StateSoftMultibodyTpl<Scalar>::integrate(const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& dx,
                                          Eigen::Ref<VectorXs> xout) const {
  if (static_cast<std::size_t>(x.size()) != nx_) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " + std::to_string(nx_) + ")");
  }
  if (static_cast<std::size_t>(dx.size()) != ndx_) {
    throw_pretty("Invalid argument: "
                 << "dx has wrong dimension (it should be " + std::to_string(ndx_) + ")");
  }
  if (static_cast<std::size_t>(xout.size()) != nx_) {
    throw_pretty("Invalid argument: "
                 << "xout has wrong dimension (it should be " + std::to_string(nx_) + ")");
  }

  pinocchio::integrate(*pinocchio_.get(), x.head(nq_l_), dx.head(nv_l_), xout.head(nq_l_));
  xout.tail(nx_-nq_l_) = x.tail(nx_-nq_l_) + dx.tail(ndx_-nv_l_);
}

template <typename Scalar>
void StateSoftMultibodyTpl<Scalar>::Jdiff(const Eigen::Ref<const VectorXs>& x0, const Eigen::Ref<const VectorXs>& x1,
                                      Eigen::Ref<MatrixXs> Jfirst, Eigen::Ref<MatrixXs> Jsecond,
                                      const Jcomponent firstsecond) const {
  assert_pretty(is_a_Jcomponent(firstsecond), ("firstsecond must be one of the Jcomponent {both, first, second}"));
  if (static_cast<std::size_t>(x0.size()) != nx_) {
    throw_pretty("Invalid argument: "
                 << "x0 has wrong dimension (it should be " + std::to_string(nx_) + ")");
  }
  if (static_cast<std::size_t>(x1.size()) != nx_) {
    throw_pretty("Invalid argument: "
                 << "x1 has wrong dimension (it should be " + std::to_string(nx_) + ")");
  }

  if (firstsecond == first) {
    if (static_cast<std::size_t>(Jfirst.rows()) != ndx_ || static_cast<std::size_t>(Jfirst.cols()) != ndx_) {
      throw_pretty("Invalid argument: "
                   << "Jfirst has wrong dimension (it should be " + std::to_string(ndx_) + "," + std::to_string(ndx_) +
                          ")");
    }

    pinocchio::dDifference(*pinocchio_.get(), x0.head(nq_l_), x1.head(nq_l_), Jfirst.topLeftCorner(nv_l_, nv_l_),
                           pinocchio::ARG0);
    Jfirst.block(nv_l_,nv_l_,nv_l_, nv_l_).diagonal().array() = (Scalar)-1;
    Jfirst.bottomRightCorner(nv_m_, nv_m_).diagonal().array() = (Scalar)-1;
    
  } else if (firstsecond == second) {
    if (static_cast<std::size_t>(Jsecond.rows()) != ndx_ || static_cast<std::size_t>(Jsecond.cols()) != ndx_) {
      throw_pretty("Invalid argument: "
                   << "Jsecond has wrong dimension (it should be " + std::to_string(ndx_) + "," +
                          std::to_string(ndx_) + ")");
    }
    pinocchio::dDifference(*pinocchio_.get(), x0.head(nq_l_), x1.head(nq_l_), Jsecond.topLeftCorner(nv_l_, nv_l_),
                           pinocchio::ARG1);
    Jsecond.block(nv_l_,nv_l_,nv_l_, nv_l_).diagonal().array() = (Scalar)1;
    Jsecond.bottomRightCorner(nv_m_, nv_m_).diagonal().array() = (Scalar)1;
  } else {  // computing both
    if (static_cast<std::size_t>(Jfirst.rows()) != ndx_ || static_cast<std::size_t>(Jfirst.cols()) != ndx_) {
      throw_pretty("Invalid argument: "
                   << "Jfirst has wrong dimension (it should be " + std::to_string(ndx_) + "," + std::to_string(ndx_) +
                          ")");
    }
    if (static_cast<std::size_t>(Jsecond.rows()) != ndx_ || static_cast<std::size_t>(Jsecond.cols()) != ndx_) {
      throw_pretty("Invalid argument: "
                   << "Jsecond has wrong dimension (it should be " + std::to_string(ndx_) + "," +
                          std::to_string(ndx_) + ")");
    }
    pinocchio::dDifference(*pinocchio_.get(), x0.head(nq_l_), x1.head(nq_l_), Jfirst.topLeftCorner(nv_l_, nv_l_),
                           pinocchio::ARG0);
    pinocchio::dDifference(*pinocchio_.get(), x0.head(nq_l_), x1.head(nq_l_), Jsecond.topLeftCorner(nv_l_, nv_l_),
                           pinocchio::ARG1);
    Jfirst.block(nv_l_,nv_l_,nv_l_, nv_l_).diagonal().array() = (Scalar)-1;
    Jfirst.bottomRightCorner(nv_m_, nv_m_).diagonal().array() = (Scalar)-1;    
    Jsecond.block(nv_l_,nv_l_,nv_l_, nv_l_).diagonal().array() = (Scalar)1;
    Jsecond.bottomRightCorner(nv_m_, nv_m_).diagonal().array() = (Scalar)1;
  }
}

template <typename Scalar>
void StateSoftMultibodyTpl<Scalar>::Jintegrate(const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& dx,
                                           Eigen::Ref<MatrixXs> Jfirst, Eigen::Ref<MatrixXs> Jsecond,
                                           const Jcomponent firstsecond, const AssignmentOp op) const {
  assert_pretty(is_a_Jcomponent(firstsecond), ("firstsecond must be one of the Jcomponent {both, first, second}"));
  assert_pretty(is_a_AssignmentOp(op), ("op must be one of the AssignmentOp {settop, addto, rmfrom}"));
  if (firstsecond == first || firstsecond == both) {
    if (static_cast<std::size_t>(Jfirst.rows()) != ndx_ || static_cast<std::size_t>(Jfirst.cols()) != ndx_) {
      throw_pretty("Invalid argument: "
                   << "Jfirst has wrong dimension (it should be " + std::to_string(ndx_) + "," + std::to_string(ndx_) +
                          ")");
    }
    switch (op) {
      case setto:
        pinocchio::dIntegrate(*pinocchio_.get(), x.head(nq_l_), dx.head(nv_l_), Jfirst.topLeftCorner(nv_l_, nv_l_),
                              pinocchio::ARG0, pinocchio::SETTO);
        Jfirst.block(nv_l_,nv_l_,nv_l_, nv_l_).diagonal().array() = (Scalar)1;
        Jfirst.block(2*nv_l_,2*nv_l_,nv_m_, nv_m_).diagonal().array() = (Scalar)1;
        Jfirst.bottomRightCorner(nv_m_, nv_m_).diagonal().array() = (Scalar)1;
        break;
      case addto:
        pinocchio::dIntegrate(*pinocchio_.get(), x.head(nq_l_), dx.head(nv_l_), Jfirst.topLeftCorner(nv_l_, nv_l_),
                              pinocchio::ARG0, pinocchio::ADDTO);
        Jfirst.block(nv_l_,nv_l_,nv_l_, nv_l_).diagonal().array() += (Scalar)1;
        Jfirst.block(2*nv_l_,2*nv_l_,nv_m_, nv_m_).diagonal().array() += (Scalar)1;
        Jfirst.bottomRightCorner(nv_m_, nv_m_).diagonal().array() += (Scalar)1;
        break;
      case rmfrom:
        pinocchio::dIntegrate(*pinocchio_.get(), x.head(nq_l_), dx.head(nv_l_), Jfirst.topLeftCorner(nv_l_, nv_l_),
                              pinocchio::ARG0, pinocchio::RMTO);
        Jfirst.block(nv_l_,nv_l_,nv_l_, nv_l_).diagonal().array() -= (Scalar)1;
        Jfirst.block(2*nv_l_,2*nv_l_,nv_m_, nv_m_).diagonal().array() -= (Scalar)1;
        Jfirst.bottomRightCorner(nv_m_, nv_m_).diagonal().array() -= (Scalar)1;
        break;
      default:
        throw_pretty("Invalid argument: allowed operators: setto, addto, rmfrom");
        break;
    }
  }
  if (firstsecond == second || firstsecond == both) {
    if (static_cast<std::size_t>(Jsecond.rows()) != ndx_ || static_cast<std::size_t>(Jsecond.cols()) != ndx_) {
      throw_pretty("Invalid argument: "
                   << "Jsecond has wrong dimension (it should be " + std::to_string(ndx_) + "," +
                          std::to_string(ndx_) + ")");
    }
    switch (op) {
      case setto:
        pinocchio::dIntegrate(*pinocchio_.get(), x.head(nq_l_), dx.head(nv_l_), Jsecond.topLeftCorner(nv_l_, nv_l_),
                              pinocchio::ARG1, pinocchio::SETTO);
        Jsecond.block(nv_l_,nv_l_,nv_l_, nv_l_).diagonal().array() = (Scalar)1;
        Jsecond.block(2*nv_l_,2*nv_l_,nv_m_, nv_m_).diagonal().array() = (Scalar)1;
        Jsecond.bottomRightCorner(nv_m_, nv_m_).diagonal().array() = (Scalar)1;
        break;
      case addto:
        pinocchio::dIntegrate(*pinocchio_.get(), x.head(nq_l_), dx.head(nv_l_), Jsecond.topLeftCorner(nv_l_, nv_l_),
                              pinocchio::ARG1, pinocchio::ADDTO);
        Jsecond.block(nv_l_,nv_l_,nv_l_, nv_l_).diagonal().array() += (Scalar)1;
        Jsecond.block(2*nv_l_,2*nv_l_,nv_m_, nv_m_).diagonal().array() += (Scalar)1;
        Jsecond.bottomRightCorner(nv_m_, nv_m_).diagonal().array() += (Scalar)1;
        break;
      case rmfrom:
        pinocchio::dIntegrate(*pinocchio_.get(), x.head(nq_l_), dx.head(nv_l_), Jsecond.topLeftCorner(nv_l_, nv_l_),
                              pinocchio::ARG1, pinocchio::RMTO);
        Jsecond.block(nv_l_,nv_l_,nv_l_, nv_l_).diagonal().array() -= (Scalar)1;
        Jsecond.block(2*nv_l_,2*nv_l_,nv_m_, nv_m_).diagonal().array() -= (Scalar)1;
        Jsecond.bottomRightCorner(nv_m_, nv_m_).diagonal().array() -= (Scalar)1;
        break;
      default:
        throw_pretty("Invalid argument: allowed operators: setto, addto, rmfrom");
        break;
    }
  }
}

template <typename Scalar>
void StateSoftMultibodyTpl<Scalar>::JintegrateTransport(const Eigen::Ref<const VectorXs>& x,
                                                    const Eigen::Ref<const VectorXs>& dx, Eigen::Ref<MatrixXs> Jin,
                                                    const Jcomponent firstsecond) const {
  assert_pretty(is_a_Jcomponent(firstsecond), ("firstsecond must be one of the Jcomponent {both, first, second}"));

  switch (firstsecond) {
    case first:
      pinocchio::dIntegrateTransport(*pinocchio_.get(), x.head(nq_l_), dx.head(nv_l_), Jin.topRows(nv_l_), pinocchio::ARG0);
      break;
    case second:
      pinocchio::dIntegrateTransport(*pinocchio_.get(), x.head(nq_l_), dx.head(nv_l_), Jin.topRows(nv_l_), pinocchio::ARG1);
      break;
    default:
      throw_pretty(
          "Invalid argument: firstsecond must be either first or second. both not supported for this operation.");
      break;
  }
}

template <typename Scalar>
std::size_t StateSoftMultibodyTpl<Scalar>::get_nv_l() const {
  return nv_l_;
}
template <typename Scalar>
std::size_t StateSoftMultibodyTpl<Scalar>::get_nv_m() const {
  return nv_m_;
}
template <typename Scalar>
std::size_t StateSoftMultibodyTpl<Scalar>::get_nq_l() const {
  return nq_l_;
}
template <typename Scalar>
std::size_t StateSoftMultibodyTpl<Scalar>::get_nq_m() const {
  return nq_m_;
}

template <typename Scalar>
const boost::shared_ptr<pinocchio::ModelTpl<Scalar> >& StateSoftMultibodyTpl<Scalar>::get_pinocchio() const {
  return pinocchio_;
}

}  // namespace crocoddyl
