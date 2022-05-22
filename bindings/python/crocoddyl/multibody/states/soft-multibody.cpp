///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2022 Centro E Piaggio, Unipi
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/multibody/states/multibody.hpp"
#include "crocoddyl/multibody/states/soft-multibody.hpp"
#include "python/crocoddyl/multibody/multibody.hpp"
#include "python/crocoddyl/core/state-base.hpp"

namespace crocoddyl {
namespace python {

void exposeStateSoftMultibody() {
  bp::register_ptr_to_python<boost::shared_ptr<crocoddyl::StateSoftMultibody> >();

  bp::class_<StateSoftMultibody, bp::bases<StateMultibody> >(
      "StateSoftMultibody",
      "Multibody state defined using Pinocchio.\n\n"
      "Pinocchio defines operators for integrating or differentiating the robot's\n"
      "configuration space. And here we assume that the state is defined by the\n"
      "robot's configuration and its joint velocities (x=[q,v]). Generally speaking,\n"
      "q lies on the manifold configuration manifold (M) and v in its tangent space\n"
      "(Tx M). Additionally the Pinocchio allows us to compute analytically the\n"
      "Jacobians for the differentiate and integrate operators. Note that this code\n"
      "can be reused in any robot that is described through its Pinocchio model.",
      bp::init<boost::shared_ptr<pinocchio::Model> >(
          bp::args("self", "pinocchioModel"),
          "Initialize the multibody state given a Pinocchio model.\n\n"
          ":param pinocchioModel: pinocchio model (i.e. multibody model)")[bp::with_custodian_and_ward<1, 2>()])
      .def("zero", &StateSoftMultibody::zero, bp::args("self"),
           "Return the neutral robot configuration with zero velocity.\n\n"
           ":return neutral robot configuration with zero velocity")
      .def("rand", &StateSoftMultibody::rand, bp::args("self"),
           "Return a random reference state.\n\n"
           ":return random reference state")
      .def("diff", &StateSoftMultibody::diff_dx, bp::args("self", "x0", "x1"),
           "Operator that differentiates the two robot states.\n\n"
           "It returns the value of x1 [-] x0 operation. This operator uses the Lie\n"
           "algebra since the robot's root could lie in the SE(3) manifold.\n"
           ":param x0: current state (dim state.nx()).\n"
           ":param x1: next state (dim state.nx()).\n"
           ":return x1 - x0 value (dim state.nx()).")
      .def("integrate", &StateSoftMultibody::integrate_x, bp::args("self", "x", "dx"),
           "Operator that integrates the current robot state.\n\n"
           "It returns the value of x [+] dx operation. This operator uses the Lie\n"
           "algebra since the robot's root could lie in the SE(3) manifold.\n"
           "Futhermore there is no timestep here (i.e. dx = v*dt), note this if you're\n"
           "integrating a velocity v during an interval dt.\n"
           ":param x: current state (dim state.nx()).\n"
           ":param dx: displacement of the state (dim state.ndx()).\n"
           ":return x + dx value (dim state.nx()).")
      .def("Jdiff", &StateSoftMultibody::Jdiff_Js,
           Jdiffs(bp::args("self", "x0", "x1", "firstsecond"),
                  "Compute the partial derivatives of the diff operator.\n\n"
                  "Both Jacobian matrices are represented throught an identity matrix, with the exception\n"
                  "that the robot's root is defined as free-flying joint (SE(3)). By default, this\n"
                  "function returns the derivatives of the first and second argument (i.e.\n"
                  "firstsecond='both'). However we ask for a specific partial derivative by setting\n"
                  "firstsecond='first' or firstsecond='second'.\n"
                  ":param x0: current state (dim state.nx()).\n"
                  ":param x1: next state (dim state.nx()).\n"
                  ":param firstsecond: derivative w.r.t x0 or x1 or both\n"
                  ":return the partial derivative(s) of the diff(x0, x1) function"))
      .def("Jintegrate", &StateSoftMultibody::Jintegrate_Js,
           Jintegrates(bp::args("self", "x", "dx", "firstsecond"),
                       "Compute the partial derivatives of arithmetic addition.\n\n"
                       "Both Jacobian matrices are represented throught an identity matrix. with the exception\n"
                       "that the robot's root is defined as free-flying joint (SE(3)). By default, this\n"
                       "function returns the derivatives of the first and second argument (i.e.\n"
                       "firstsecond='both'). However we ask for a specific partial derivative by setting\n"
                       "firstsecond='first' or firstsecond='second'.\n"
                       ":param x: current state (dim state.nx()).\n"
                       ":param dx: displacement of the state (dim state.ndx()).\n"
                       ":param firstsecond: derivative w.r.t x or dx or both\n"
                       ":return the partial derivative(s) of the integrate(x, dx) function"))
      .def("JintegrateTransport", &StateSoftMultibody::JintegrateTransport,
           bp::args("self", "x", "dx", "Jin", "firstsecond"),
           "Parallel transport from integrate(x, dx) to x.\n\n"
           "This function performs the parallel transportation of an input matrix whose columns\n"
           "are expressed in the tangent space at integrate(x, dx) to the tangent space at x point\n"
           ":param x: state point (dim. state.nx).\n"
           ":param dx: velocity vector (dim state.ndx).\n"
           ":param Jin: input matrix (number of rows = state.nv).\n"
           ":param firstsecond: derivative w.r.t x or dx")
      .add_property("pinocchio",
                    bp::make_function(&StateSoftMultibody::get_pinocchio, bp::return_value_policy<bp::return_by_value>()),
                    "pinocchio model")
      .add_property("nq_m", bp::make_function(&StateSoftMultibody::get_nq_m), "dimension of configuration vector of motor side")
      .add_property("nv_m", bp::make_function(&StateSoftMultibody::get_nv_m), "dimension of velocity vector of motor side")
      .add_property("nq_l", bp::make_function(&StateSoftMultibody::get_nq_l),"dimension of the configuration of link side")
      .add_property("nv_l", bp::make_function(&StateSoftMultibody::get_nv_l), "dimension of velocity of link side ");

}

}  // namespace python
}  // namespace crocoddyl
