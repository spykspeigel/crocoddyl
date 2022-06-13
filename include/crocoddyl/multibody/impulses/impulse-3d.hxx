///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/multibody/impulses/impulse-3d.hpp"

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics-derivatives.hpp>

namespace crocoddyl {

template <typename Scalar>
ImpulseModel3DTpl<Scalar>::ImpulseModel3DTpl(boost::shared_ptr<StateMultibody> state, const std::size_t frame)
    : Base(state, 3), frame_(frame) {}

template <typename Scalar>
ImpulseModel3DTpl<Scalar>::~ImpulseModel3DTpl() {}

template <typename Scalar>
void ImpulseModel3DTpl<Scalar>::calc(const boost::shared_ptr<ImpulseDataAbstract>& data,
                                     const Eigen::Ref<const VectorXs>&) {
  boost::shared_ptr<Data> d = boost::static_pointer_cast<Data>(data);
  const std::size_t nv_l = 18;
  pinocchio::getFrameJacobian(*state_->get_pinocchio().get(), *d->pinocchio, frame_, pinocchio::LOCAL, d->fJf.leftCols(nv_l));
    d->Jc.leftCols(nv_l) = d->fJf.leftCols(nv_l).template topRows<3>();
}

template <typename Scalar>
void ImpulseModel3DTpl<Scalar>::calcDiff(const boost::shared_ptr<ImpulseDataAbstract>& data,
                                         const Eigen::Ref<const VectorXs>&) {
  boost::shared_ptr<Data> d = boost::static_pointer_cast<Data>(data);
  const std::size_t nv_l = 18;
  const pinocchio::JointIndex joint = state_->get_pinocchio()->frames[d->frame].parent;
  pinocchio::getJointVelocityDerivatives(*state_->get_pinocchio().get(), *d->pinocchio, joint, pinocchio::LOCAL,
                                         d->v_partial_dq.leftCols(nv_l), d->v_partial_dv.leftCols(nv_l));
  d->dv0_dq.leftCols(nv_l).noalias() = d->fXj.leftCols(nv_l).template topRows<3>() * d->v_partial_dq.leftCols(nv_l);
}

template <typename Scalar>
void ImpulseModel3DTpl<Scalar>::updateForce(const boost::shared_ptr<ImpulseDataAbstract>& data,
                                            const VectorXs& force) {
  if (force.size() != 3) {
    throw_pretty("Invalid argument: "
                 << "lambda has wrong dimension (it should be 3)");
  }
  data->f = data->jMf.act(pinocchio::ForceTpl<Scalar>(force, Vector3s::Zero()));
}

template <typename Scalar>
boost::shared_ptr<ImpulseDataAbstractTpl<Scalar> > ImpulseModel3DTpl<Scalar>::createData(
    pinocchio::DataTpl<Scalar>* const data) {
  return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this, data);
}

template <typename Scalar>
void ImpulseModel3DTpl<Scalar>::print(std::ostream& os) const {
  os << "ImpulseModel3D {frame=" << state_->get_pinocchio()->frames[frame_].name << "}";
}

template <typename Scalar>
std::size_t ImpulseModel3DTpl<Scalar>::get_frame() const {
  return frame_;
}

}  // namespace crocoddyl
