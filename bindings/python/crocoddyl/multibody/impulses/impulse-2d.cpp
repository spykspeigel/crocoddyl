///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/multibody/impulses/impulse-2d.hpp"
#include "python/crocoddyl/multibody/multibody.hpp"

namespace crocoddyl {
namespace python {

void exposeImpulse2D() {
  bp::register_ptr_to_python<boost::shared_ptr<ImpulseModel2D> >();

  bp::class_<ImpulseModel2D, bp::bases<ImpulseModelAbstract> >(
      "ImpulseModel2D",
      "Rigid 2D impulse model.\n\n"
      "It defines a rigid 2D impulse models (point impulse) based on acceleration-based holonomic constraints.\n"
      "The calc and calcDiff functions compute the impulse Jacobian and drift (holonomic constraint) or\n"
      "the derivatives of the holonomic constraint, respectively.",
      bp::init<boost::shared_ptr<StateMultibody>, int>(bp::args("self", "state", "frame"),
                                                       "Initialize the 3D impulse model.\n\n"
                                                       ":param state: state of the multibody system\n"
                                                       ":param frame: reference frame id"))
      .def("calc", &ImpulseModel2D::calc, bp::args("self", "data", "x"),
           "Compute the 3D impulse Jacobian and drift.\n\n"
           "The rigid impulse model throught acceleration-base holonomic constraint\n"
           "of the impulse frame placement.\n"
           ":param data: impulse data\n"
           ":param x: state point (dim. state.nx)")
      .def("calcDiff", &ImpulseModel2D::calcDiff, bp::args("self", "data", "x"),
           "Compute the derivatives of the 3D impulse holonomic constraint.\n\n"
           "The rigid impulse model throught acceleration-base holonomic constraint\n"
           "of the impulse frame placement.\n"
           "It assumes that calc has been run first.\n"
           ":param data: cost data\n"
           ":param x: state point (dim. state.nx)")
      .def("updateForce", &ImpulseModel2D::updateForce, bp::args("self", "data", "force"),
           "Convert the force into a stack of spatial forces.\n\n"
           ":param data: cost data\n"
           ":param force: force vector (dimension 3)")
      .def("createData", &ImpulseModel2D::createData, bp::with_custodian_and_ward_postcall<0, 2>(),
           bp::args("self", "data"),
           "Create the 3D impulse data.\n\n"
           "Each impulse model has its own data that needs to be allocated. This function\n"
           "returns the allocated data for a predefined cost.\n"
           ":param data: Pinocchio data\n"
           ":return impulse data.")
      .add_property("frame", bp::make_function(&ImpulseModel2D::get_frame), "reference frame id");

  bp::register_ptr_to_python<boost::shared_ptr<ImpulseData2D> >();

  bp::class_<ImpulseData2D, bp::bases<ImpulseDataAbstract> >(
      "ImpulseData2D", "Data for 2D impulse.\n\n",
      bp::init<ImpulseModel2D*, pinocchio::Data*>(
          bp::args("self", "model", "data"),
          "Create 3D impulse data.\n\n"
          ":param model: 3D impulse model\n"
          ":param data: Pinocchio data")[bp::with_custodian_and_ward<1, 2, bp::with_custodian_and_ward<1, 3> >()])
      .add_property("fXj", bp::make_getter(&ImpulseData2D::fXj, bp::return_value_policy<bp::return_by_value>()),
                    "action matrix from impulse to local frames")
      .add_property("fJf", bp::make_getter(&ImpulseData2D::fJf, bp::return_internal_reference<>()),
                    "local Jacobian of the impulse frame")
      .add_property("v_partial_dq", bp::make_getter(&ImpulseData2D::v_partial_dq, bp::return_internal_reference<>()),
                    "Jacobian of the spatial body velocity")
      .add_property("v_partial_dv", bp::make_getter(&ImpulseData2D::v_partial_dv, bp::return_internal_reference<>()),
                    "Jacobian of the spatial body velocity");
}

}  // namespace python
}  // namespace crocoddyl
