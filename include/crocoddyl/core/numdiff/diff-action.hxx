///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh, New York University,
// Max Planck Gesellschaft
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/core/numdiff/diff-action.hpp"

namespace crocoddyl {

template <typename Scalar>
DifferentialActionModelNumDiffTpl<Scalar>::DifferentialActionModelNumDiffTpl(boost::shared_ptr<Base> model,
                                                                             const bool with_gauss_approx)
    : Base(model->get_state(), model->get_nu(), model->get_nr()), model_(model) {
  with_gauss_approx_ = with_gauss_approx;
  disturbance_ = std::sqrt(2.0 * std::numeric_limits<Scalar>::epsilon());
  if (with_gauss_approx_ && nr_ == 1) throw_pretty("No Gauss approximation possible with nr = 1");
}

template <typename Scalar>
DifferentialActionModelNumDiffTpl<Scalar>::~DifferentialActionModelNumDiffTpl() {}

template <typename Scalar>
void DifferentialActionModelNumDiffTpl<Scalar>::calc(const boost::shared_ptr<DifferentialActionDataAbstract>& data,
                                                     const Eigen::Ref<const VectorXs>& x,
                                                     const Eigen::Ref<const VectorXs>& u) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
  }
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "u has wrong dimension (it should be " + std::to_string(nu_) + ")");
  }
  Data* d = static_cast<Data*>(data.get());
  model_->calc(d->data_0, x, u);
  data->cost = d->data_0->cost;
  data->xout = d->data_0->xout;
}

template <typename Scalar>
void DifferentialActionModelNumDiffTpl<Scalar>::calc(const boost::shared_ptr<DifferentialActionDataAbstract>& data,
                                                     const Eigen::Ref<const VectorXs>& x) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
  }
  Data* d = static_cast<Data*>(data.get());
  model_->calc(d->data_0, x);
  data->cost = d->data_0->cost;
  data->xout = d->data_0->xout;
}

template <typename Scalar>
void DifferentialActionModelNumDiffTpl<Scalar>::calcDiff(const boost::shared_ptr<DifferentialActionDataAbstract>& data,
                                                         const Eigen::Ref<const VectorXs>& x,
                                                         const Eigen::Ref<const VectorXs>& u) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
  }
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "u has wrong dimension (it should be " + std::to_string(nu_) + ")");
  }
  Data* d = static_cast<Data*>(data.get());

  const VectorXs& xn0 = d->data_0->xout;
  const Scalar c0 = d->data_0->cost;
  data->xout = d->data_0->xout;
  data->cost = d->data_0->cost;

  assertStableStateFD(x);

  // Computing the d action(x,u) / dx
  d->dx.setZero();
  for (std::size_t ix = 0; ix < state_->get_ndx(); ++ix) {
    d->dx(ix) = disturbance_;
    model_->get_state()->integrate(x, d->dx, d->xp);
    model_->calc(d->data_x[ix], d->xp, u);

    const VectorXs& xp = d->data_x[ix]->xout;
    const Scalar cp = d->data_x[ix]->cost;
    data->Fx.col(ix) = (xp - xn0) / disturbance_;

    data->Lx(ix) = (cp - c0) / disturbance_;
    d->Rx.col(ix) = (d->data_x[ix]->r - d->data_0->r) / disturbance_;
    
    // The finite difference of the partial derivatives 
    //
    //  First order derivatives
    //  Lx[i] = (L(x+ \delta x) - L(x))/disturbance                   # this formula has a higher order terms of O(\delta x)
    //  Lx[i] = (L(x+ \delta x) - L(x- \delta x)) / disturbance       # this formula has a higher order terms of O(\delta x^2)

    //  Lxx[i,i] = (Lx(x_i+\delta x_i)-2*Lx(x_i)+Lx(x_i-\delta x_i))/disturbance**2
    //  Lxx[i,j] = (Lx(x_i+\delta x_i, x_j+\delta x_j) - Lx(x_i+\delta x_i, x_j) - Lx(x_i, x_j+\delta x_j) + Lx(x_i, x_j)) /  (delta x_i *delta x_j)      #this formula has a higher order terms of O(\delta x)
    //  One can write a similar formula for the finite difference with O(delta x**2) but from computation time perspective the above formula is more efficient as we can reuse 2 terms

    // d->dx(ix) = -disturbance_;
    model_->get_state()->integrate(x, -d->dx, d->xp); 
    model_->calc(d->data_x[ix], d->xp, u);
    data->Lxx(ix,ix) = (d->data_x[ix]->cost + cp - 2*c0)/ (disturbance_*disturbance_);    
    std::cout<<"Here is the value of ix :"<<ix<<"\n";

    for (std::size_t jx = ix+1; jx < state_->get_ndx(); ++jx) {
      
      std::cout<<"Here is the value of jx :"<<jx<<"\n";

      d->dx(jx) = disturbance_;
      model_->get_state()->integrate(x, d->dx, d->xp); 
      model_->calc(d->data_x[ix], d->xp, u);
      const Scalar c_pp = d->data_x[ix]->cost;  //cost due to positive disturbance in both directions
      std::cout<<"Here is the value of dx :"<<d->dx(jx)<<"\n";

      d->dx(ix) = 0.0;
      model_->get_state()->integrate(x, d->dx, d->xp); 
      model_->calc(d->data_x[ix], d->xp, u);
      const Scalar c_zp = d->data_x[ix]->cost;  //cost due to zero disturance in 'i' and positive disturbance in 'j' direction
      std::cout<<"Here is the value of dx :"<<d->dx(jx)<<"\n";
      std::cout<<"Here is the value of c,  c_pp, c_zp :"<<cp << c_pp << c_zp<<"\n";
    
      data->Lxx(ix,jx) = (c_pp - c_zp - cp + c0)/ (disturbance_*disturbance_);
      
      data->Lxx(jx,ix) = data->Lxx(ix,jx); // can also be implemented at the end by Lxx = Lxx.T
      d->dx(ix) = disturbance_;
      d->dx(jx) = 0.0;
    }

    d->dx(ix) = 0.0;

  }

  // Computing the d action(x,u) / du
  d->du.setZero();
  for (unsigned iu = 0; iu < model_->get_nu(); ++iu) {
    d->du(iu) = disturbance_;
    model_->calc(d->data_u[iu], x, u + d->du);

    const VectorXs& xn = d->data_u[iu]->xout;
    const Scalar cp = d->data_u[iu]->cost;
    data->Fu.col(iu) = (xn - xn0) / disturbance_;

    data->Lu(iu) = (cp - c0) / disturbance_;
    d->Ru.col(iu) = (d->data_u[iu]->r - d->data_0->r) / disturbance_;

    // We can apply the same formulas for finite difference as above

    // d->du(iu) = -disturbance_; 
    model_->calc(d->data_u[iu], d->xp, u- d->du);
    data->Luu(iu,iu) = (d->data_u[iu]->cost + cp - 2*c0)/ (disturbance_*disturbance_);    
    std::cout<<"Here is the value of iu :"<<iu<<"\n";

    for (std::size_t ju = iu+1; ju < model_->get_nu(); ++ju) {
      
      std::cout<<"Here is the value of ju :"<<ju<<"\n";

      d->du(ju) = disturbance_;

      model_->calc(d->data_u[iu], d->xp, u + d->du);
      const Scalar c_pp = d->data_u[iu]->cost;  //cost due to positive disturbance in both directions
      std::cout<<"Here is the value of du :"<<d->du<<"\n";

      d->du(iu) = 0.0;

      model_->calc(d->data_u[iu], d->xp, u + d->du);
      const Scalar c_zp = d->data_u[iu]->cost;  //cost due to zero disturance in 'i' and positive disturbance in 'j' direction
      std::cout<<"Here is the value of du :"<<d->du<<"\n";
      std::cout<<"Here is the value of c,  c_pp, c_zp :"<<cp << c_pp << c_zp<<"\n";
    
      data->Luu(iu,ju) = (c_pp - c_zp - cp + c0)/ (disturbance_*disturbance_);
      
      data->Luu(ju,iu) = data->Luu(iu,ju); // can also be implemented at the end by Luu = Luu.T
      d->du(iu) = disturbance_;
      d->du(ju) = 0.0;
    }

    d->du(iu) = 0.0;

  }

  if (with_gauss_approx_) {
    // data->Lxx = d->Rx.transpose() * d->Rx;
    data->Lxu = d->Rx.transpose() * d->Ru;
    // data->Luu = d->Ru.transpose() * d->Ru;
  } else {
    // data->Lxx.setZero();
    data->Lxu.setZero();
    // data->Luu.setZero();
  }
}

template <typename Scalar>
void DifferentialActionModelNumDiffTpl<Scalar>::calcDiff(const boost::shared_ptr<DifferentialActionDataAbstract>& data,
                                                         const Eigen::Ref<const VectorXs>& x) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
  }
  Data* d = static_cast<Data*>(data.get());

  const Scalar c0 = d->data_0->cost;
  data->xout = d->data_0->xout;
  data->cost = d->data_0->cost;

  assertStableStateFD(x);

  // Computing the d action(x,u) / dx
  d->dx.setZero();
  for (std::size_t ix = 0; ix < state_->get_ndx(); ++ix) {
    d->dx(ix) = disturbance_;
    model_->get_state()->integrate(x, d->dx, d->xp);
    model_->calc(d->data_x[ix], d->xp);
    const Scalar c = d->data_x[ix]->cost;

    data->Lx(ix) = (c - c0) / disturbance_;
    d->Rx.col(ix) = (d->data_x[ix]->r - d->data_0->r) / disturbance_;
    d->dx(ix) = 0.0;
  }

  if (with_gauss_approx_) {
    data->Lxx = d->Rx.transpose() * d->Rx;
  } else {
    data->Lxx.setZero();
  }
}

template <typename Scalar>
boost::shared_ptr<DifferentialActionDataAbstractTpl<Scalar> > DifferentialActionModelNumDiffTpl<Scalar>::createData() {
  return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this);
}

template <typename Scalar>
const boost::shared_ptr<DifferentialActionModelAbstractTpl<Scalar> >&
DifferentialActionModelNumDiffTpl<Scalar>::get_model() const {
  return model_;
}

template <typename Scalar>
const Scalar DifferentialActionModelNumDiffTpl<Scalar>::get_disturbance() const {
  return disturbance_;
}

template <typename Scalar>
void DifferentialActionModelNumDiffTpl<Scalar>::set_disturbance(const Scalar disturbance) {
  if (disturbance < 0.) {
    throw_pretty("Invalid argument: "
                 << "Disturbance value is positive");
  }
  disturbance_ = disturbance;
}

template <typename Scalar>
bool DifferentialActionModelNumDiffTpl<Scalar>::get_with_gauss_approx() {
  return with_gauss_approx_;
}

template <typename Scalar>
void DifferentialActionModelNumDiffTpl<Scalar>::assertStableStateFD(const Eigen::Ref<const VectorXs>& /** x */) {
  // TODO(cmastalli): First we need to do it AMNumDiff and then to replicate it.
}

}  // namespace crocoddyl
