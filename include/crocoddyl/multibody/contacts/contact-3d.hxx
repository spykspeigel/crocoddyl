///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

namespace crocoddyl {

template <typename Scalar>
ContactModel3DTpl<Scalar>::ContactModel3DTpl(boost::shared_ptr<StateMultibody> state, const pinocchio::FrameIndex id,
                                             const Vector3s& xref, const std::size_t nu, const Vector2s& gains)
    : Base(state, 3, nu), xref_(xref), gains_(gains) {
  id_ = id;
}

template <typename Scalar>
ContactModel3DTpl<Scalar>::ContactModel3DTpl(boost::shared_ptr<StateMultibody> state, const pinocchio::FrameIndex id,
                                             const Vector3s& xref, const Vector2s& gains)
    : Base(state, 3), xref_(xref), gains_(gains) {
  id_ = id;
}

#pragma GCC diagnostic push  // TODO: Remove once the deprecated FrameXX has been removed in a future release
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

template <typename Scalar>
ContactModel3DTpl<Scalar>::ContactModel3DTpl(boost::shared_ptr<StateMultibody> state,
                                             const FrameTranslationTpl<Scalar>& xref, const std::size_t nu,
                                             const Vector2s& gains)
    : Base(state, 3, nu), xref_(xref.translation), gains_(gains) {
  id_ = xref.id;
  std::cerr << "Deprecated: Use constructor which is not based on FrameTranslation." << std::endl;
}

template <typename Scalar>
ContactModel3DTpl<Scalar>::ContactModel3DTpl(boost::shared_ptr<StateMultibody> state,
                                             const FrameTranslationTpl<Scalar>& xref, const Vector2s& gains)
    : Base(state, 3), xref_(xref.translation), gains_(gains) {
  id_ = xref.id;
  std::cerr << "Deprecated: Use constructor which is not based on FrameTranslation." << std::endl;
}

#pragma GCC diagnostic pop

template <typename Scalar>
ContactModel3DTpl<Scalar>::~ContactModel3DTpl() {}

template <typename Scalar>
void ContactModel3DTpl<Scalar>::calc(const boost::shared_ptr<ContactDataAbstract>& data,
                                     const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());
  const std::size_t nv_l = 18;
  pinocchio::updateFramePlacement(*state_->get_pinocchio().get(), *d->pinocchio, id_);
  pinocchio::getFrameJacobian(*state_->get_pinocchio().get(), *d->pinocchio, id_, pinocchio::LOCAL, d->fJf);
  d->v = pinocchio::getFrameVelocity(*state_->get_pinocchio().get(), *d->pinocchio, id_);
  d->a = pinocchio::getFrameAcceleration(*state_->get_pinocchio().get(), *d->pinocchio, id_);

  d->Jc.leftCols(nv_l) = d->fJf.leftCols(nv_l).template topRows<3>();
  d->vw = d->v.angular();
  d->vv = d->v.linear();
  d->a0 = d->a.linear() + d->vw.cross(d->vv);

  if (gains_[0] != 0.) {
    d->a0 += gains_[0] * (d->pinocchio->oMf[id_].translation() - xref_);
  }
  if (gains_[1] != 0.) {
    d->a0 += gains_[1] * d->vv;
  }
}

template <typename Scalar>
void ContactModel3DTpl<Scalar>::calcDiff(const boost::shared_ptr<ContactDataAbstract>& data,
                                         const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());
  const pinocchio::JointIndex joint = state_->get_pinocchio()->frames[d->frame].parent;
  const std::size_t nv_l = 18;
  pinocchio::getJointAccelerationDerivatives(*state_->get_pinocchio().get(), *d->pinocchio, joint, pinocchio::LOCAL,
                                             d->v_partial_dq.leftCols(nv_l), d->a_partial_dq.leftCols(nv_l), d->a_partial_dv.leftCols(nv_l), d->a_partial_da.leftCols(nv_l));
  const std::size_t nv = state_->get_nv();
  pinocchio::skew(d->vv, d->vv_skew);
  pinocchio::skew(d->vw, d->vw_skew);
  d->fXjdv_dq.leftCols(nv_l).noalias() = d->fXj * d->v_partial_dq.leftCols(nv_l);
  d->fXjda_dq.leftCols(nv_l).noalias() = d->fXj * d->a_partial_dq.leftCols(nv_l);
  d->fXjda_dv.leftCols(nv_l).noalias() = d->fXj * d->a_partial_dv.leftCols(nv_l);
  d->da0_dx.leftCols(nv_l) = d->fXjda_dq.leftCols(nv_l).template topRows<3>();
  d->da0_dx.leftCols(nv_l).noalias() += d->vw_skew * d->fXjdv_dq.leftCols(nv_l).template topRows<3>();
  d->da0_dx.leftCols(nv_l).noalias() -= d->vv_skew * d->fXjdv_dq.leftCols(nv_l).template bottomRows<3>();
  d->da0_dx.middleCols(nv_l,nv_l) = d->fXjda_dv.leftCols(nv_l).template topRows<3>();
  d->da0_dx.middleCols(nv_l,nv_l).noalias() += d->vw_skew * d->Jc.leftCols(nv_l);
  d->da0_dx.middleCols(nv_l,nv_l).noalias() -= d->vv_skew * d->fJf.leftCols(nv_l).template bottomRows<3>();
  if (gains_[0] != 0.) {
    d->oRf = d->pinocchio->oMf[id_].rotation();
    d->da0_dx.leftCols(nv_l).noalias() += gains_[0] * d->oRf * d->Jc.leftCols(nv_l);
  }
  if (gains_[1] != 0.) {
    d->da0_dx.leftCols(nv_l).noalias() += gains_[1] * d->fXj.template topRows<3>() * d->v_partial_dq;
    d->da0_dx.middleCols(nv_l,nv_l).noalias() += gains_[1] * d->fXj.template topRows<3>() * d->a_partial_da;
  }
}

template <typename Scalar>
void ContactModel3DTpl<Scalar>::updateForce(const boost::shared_ptr<ContactDataAbstract>& data,
                                            const VectorXs& force) {
  if (force.size() != 3) {
    throw_pretty("Invalid argument: "
                 << "lambda has wrong dimension (it should be 3)");
  }
  Data* d = static_cast<Data*>(data.get());
  data->f = d->jMf.act(pinocchio::ForceTpl<Scalar>(force, Vector3s::Zero()));
}

template <typename Scalar>
boost::shared_ptr<ContactDataAbstractTpl<Scalar> > ContactModel3DTpl<Scalar>::createData(
    pinocchio::DataTpl<Scalar>* const data) {
  return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this, data);
}

template <typename Scalar>
void ContactModel3DTpl<Scalar>::print(std::ostream& os) const {
  os << "ContactModel3D {frame=" << state_->get_pinocchio()->frames[id_].name << "}";
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::Vector3s& ContactModel3DTpl<Scalar>::get_reference() const {
  return xref_;
}

#pragma GCC diagnostic push  // TODO: Remove once the deprecated FrameXX has been removed in a future release
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

template <typename Scalar>
FrameTranslationTpl<Scalar> ContactModel3DTpl<Scalar>::get_xref() const {
  return FrameTranslationTpl<Scalar>(id_, xref_);
}

#pragma GCC diagnostic pop

template <typename Scalar>
const typename MathBaseTpl<Scalar>::Vector2s& ContactModel3DTpl<Scalar>::get_gains() const {
  return gains_;
}

template <typename Scalar>
void ContactModel3DTpl<Scalar>::set_reference(const Vector3s& reference) {
  xref_ = reference;
}

}  // namespace crocoddyl
