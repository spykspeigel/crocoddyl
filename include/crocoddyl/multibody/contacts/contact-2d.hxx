///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020-2021, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

namespace crocoddyl {

template <typename Scalar>
ContactModel2DTpl<Scalar>::ContactModel2DTpl(boost::shared_ptr<StateMultibody> state, const pinocchio::FrameIndex id,
                                             const Vector2s& xref, const std::size_t nu, const Vector2s& gains)
    : Base(state, 2, nu), xref_(xref), gains_(gains) {
  id_ = id;
}

template <typename Scalar>
ContactModel2DTpl<Scalar>::ContactModel2DTpl(boost::shared_ptr<StateMultibody> state, const pinocchio::FrameIndex id,
                                             const Vector2s& xref, const Vector2s& gains)
    : Base(state, 2), xref_(xref), gains_(gains) {
  id_ = id;
}

#pragma GCC diagnostic push  // TODO: Remove once the deprecated FrameXX has been removed in a future release
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

template <typename Scalar>
ContactModel2DTpl<Scalar>::ContactModel2DTpl(boost::shared_ptr<StateMultibody> state,
                                             const FrameTranslationTpl<Scalar>& xref, const std::size_t nu,
                                             const Vector2s& gains)
    : Base(state, 2, nu), xref_(Vector2s(xref.translation[0], xref.translation[2])), gains_(gains) {
  id_ = xref.id;
  std::cerr << "Deprecated: Use constructor which is not based on FrameTranslation." << std::endl;
}

template <typename Scalar>
ContactModel2DTpl<Scalar>::ContactModel2DTpl(boost::shared_ptr<StateMultibody> state,
                                             const FrameTranslationTpl<Scalar>& xref, const Vector2s& gains)
    : Base(state, 2), xref_(Vector2s(xref.translation[0], xref.translation[2])), gains_(gains) {
  id_ = xref.id;
  std::cerr << "Deprecated: Use constructor which is not based on FrameTranslation." << std::endl;
}

#pragma GCC diagnostic pop

template <typename Scalar>
ContactModel2DTpl<Scalar>::~ContactModel2DTpl() {}

template <typename Scalar>
void ContactModel2DTpl<Scalar>::calc(const boost::shared_ptr<ContactDataAbstract>& data,
                                     const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());
  const std::size_t nv_l = 3;
  pinocchio::updateFramePlacement(*state_->get_pinocchio().get(), *d->pinocchio, id_);
  pinocchio::getFrameJacobian(*state_->get_pinocchio().get(), *d->pinocchio, id_, pinocchio::LOCAL, d->fJf.leftCols(nv_l));
  d->v = pinocchio::getFrameVelocity(*state_->get_pinocchio().get(), *d->pinocchio, id_);
  d->a = pinocchio::getFrameAcceleration(*state_->get_pinocchio().get(), *d->pinocchio, id_);
  d->Jc.row(0).leftCols(nv_l) = d->fJf.row(0).leftCols(nv_l);
  d->Jc.row(1).leftCols(nv_l) = d->fJf.row(2).leftCols(nv_l);

  d->vw = d->v.angular();
  d->vv = d->v.linear();

  d->a0[0] = d->a.linear()[0] + d->vw[1] * d->vv[2] - d->vw[2] * d->vv[1];
  d->a0[1] = d->a.linear()[2] + d->vw[0] * d->vv[1] - d->vw[1] * d->vv[0];

  if (gains_[0] != 0.) {
    d->a0[0] += gains_[0] * (d->pinocchio->oMf[id_].translation()[0] - xref_[0]);
    d->a0[1] += gains_[0] * (d->pinocchio->oMf[id_].translation()[2] - xref_[1]);
  }
  if (gains_[1] != 0.) {
    d->a0[0] += gains_[1] * d->vv[0];
    d->a0[1] += gains_[1] * d->vv[2];
  }
}

template <typename Scalar>
void ContactModel2DTpl<Scalar>::calcDiff(const boost::shared_ptr<ContactDataAbstract>& data,
                                         const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());
  const pinocchio::JointIndex joint = state_->get_pinocchio()->frames[d->frame].parent;
  const std::size_t nv_l = 3;
  pinocchio::getJointAccelerationDerivatives(*state_->get_pinocchio().get(), *d->pinocchio, joint, pinocchio::LOCAL,
                                             d->v_partial_dq.leftCols(nv_l), d->a_partial_dq.leftCols(nv_l), d->a_partial_dv.leftCols(nv_l), d->a_partial_da.leftCols(nv_l));
  const std::size_t nv = state_->get_nv();
  pinocchio::skew(d->vv, d->vv_skew);
  pinocchio::skew(d->vw, d->vw_skew);
  d->fXjdv_dq.noalias() = d->fXj * d->v_partial_dq.leftCols(nv_l);
  d->fXjda_dq.noalias() = d->fXj * d->a_partial_dq.leftCols(nv_l);
  d->fXjda_dv.noalias() = d->fXj * d->a_partial_dv.leftCols(nv_l);

  d->da0_dx.leftCols(nv_l).row(0) = d->fXjda_dq.row(0).leftCols(nv_l);
  d->da0_dx.leftCols(nv_l).row(0).noalias() += d->vw_skew.row(0) * d->fXjdv_dq.leftCols(nv_l).template topRows<3>();
  d->da0_dx.leftCols(nv_l).row(0).noalias() -= d->vv_skew.row(0) * d->fXjdv_dq.leftCols(nv_l).template bottomRows<3>();

  d->da0_dx.leftCols(nv_l).row(1) = d->fXjda_dq.row(2).leftCols(nv_l);
  d->da0_dx.leftCols(nv_l).row(1).noalias() += d->vw_skew.row(2) * d->fXjdv_dq.leftCols(nv_l).template topRows<3>();
  d->da0_dx.leftCols(nv_l).row(1).noalias() -= d->vv_skew.row(2) * d->fXjdv_dq.leftCols(nv_l).template bottomRows<3>();

  d->da0_dx.middleCols(nv_l,nv_l).row(0) = d->fXjda_dv.row(0).leftCols(nv_l);
  d->da0_dx.middleCols(nv_l,nv_l).row(0).noalias() += d->vw_skew.row(0) * d->fJf.leftCols(nv_l).template topRows<3>();
  d->da0_dx.middleCols(nv_l,nv_l).row(0).noalias() -= d->vv_skew.row(0) * d->fJf.leftCols(nv_l).template bottomRows<3>();

  d->da0_dx.middleCols(nv_l,nv_l).row(1) = d->fXjda_dv.row(2).leftCols(nv_l);
  d->da0_dx.middleCols(nv_l,nv_l).row(1).noalias() += d->vw_skew.row(2) * d->fJf.leftCols(nv_l).template topRows<3>();
  d->da0_dx.middleCols(nv_l,nv_l).row(1).noalias() -= d->vv_skew.row(2) * d->fJf.leftCols(nv_l).template bottomRows<3>();

  if (gains_[0] != 0.) {
    const Eigen::Ref<const Matrix3s> oRf = d->pinocchio->oMf[id_].rotation();
    d->oRf(0, 0) = oRf(0, 0);
    d->oRf(1, 0) = oRf(2, 0);
    d->oRf(0, 1) = oRf(0, 2);
    d->oRf(1, 1) = oRf(2, 2);
    d->da0_dx.leftCols(nv_l).noalias() += gains_[0] * d->oRf * d->Jc.leftCols(nv_l);
  }
  if (gains_[1] != 0.) {
    d->da0_dx.leftCols(nv_l).row(0).noalias() += gains_[1] * d->fXj.row(0) * d->v_partial_dq.leftCols(nv_l);
    d->da0_dx.leftCols(nv_l).row(1).noalias() += gains_[1] * d->fXj.row(2) * d->v_partial_dq.leftCols(nv_l);
    d->da0_dx.middleCols(nv_l,nv_l).row(0).noalias() += gains_[1] * d->fXj.row(0) * d->a_partial_da.leftCols(nv_l);
    d->da0_dx.middleCols(nv_l,nv_l).row(1).noalias() += gains_[1] * d->fXj.row(2) * d->a_partial_da.leftCols(nv_l);
  }
}

template <typename Scalar>
void ContactModel2DTpl<Scalar>::updateForce(const boost::shared_ptr<ContactDataAbstract>& data,
                                            const VectorXs& force) {
  if (force.size() != 2) {
    throw_pretty("Invalid argument: "
                 << "lambda has wrong dimension (it should be 2)");
  }
  Data* d = static_cast<Data*>(data.get());
  const Eigen::Ref<const Matrix3s> R = d->jMf.rotation();
  data->f.linear() = R.col(0) * force[0] + R.col(2) * force[1];
  data->f.angular() = d->jMf.translation().cross(data->f.linear());
}

template <typename Scalar>
boost::shared_ptr<ContactDataAbstractTpl<Scalar> > ContactModel2DTpl<Scalar>::createData(
    pinocchio::DataTpl<Scalar>* const data) {
  return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this, data);
}

template <typename Scalar>
void ContactModel2DTpl<Scalar>::print(std::ostream& os) const {
  os << "ContactModel2D {frame=" << state_->get_pinocchio()->frames[id_].name << "}";
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::Vector2s& ContactModel2DTpl<Scalar>::get_reference() const {
  return xref_;
}

#pragma GCC diagnostic push  // TODO: Remove once the deprecated FrameXX has been removed in a future release
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

template <typename Scalar>
FrameTranslationTpl<Scalar> ContactModel2DTpl<Scalar>::get_xref() const {
  Vector3s x(xref_[0], 0., xref_[1]);
  return FrameTranslationTpl<Scalar>(id_, x);
}

#pragma GCC diagnostic pop

template <typename Scalar>
const typename MathBaseTpl<Scalar>::Vector2s& ContactModel2DTpl<Scalar>::get_gains() const {
  return gains_;
}

template <typename Scalar>
void ContactModel2DTpl<Scalar>::set_reference(const Vector2s& reference) {
  xref_ = reference;
}

}  // namespace crocoddyl
