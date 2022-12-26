#include "utils.cpp"
#include "regression.hpp"
#include "log_reg.hpp"
#include "kmeans.hpp"

namespace py = pybind11;

PYBIND11_MODULE(simpleml, m)
{
    m.doc() = "simple machine learning tookits";
    
    // linear regression
    py::class_<Regression>(m, "Regression")
        .def(py::init<>())
        .def("load_data", static_cast<void (Regression::*)(const char *, const char *)>(&Regression::load_data))
        .def("load_data", static_cast<void (Regression::*)(std::vector<double> const &, std::vector<double> const &)>(&Regression::load_data))
        .def("train", &Regression::train)
        .def("test", static_cast<std::vector<double> (Regression::*)()>(&Regression::test))
        .def("test", static_cast<std::vector<double> (Regression::*)(std::vector<double> const &)>(&Regression::test))
        .def_property_readonly("weight", &Regression::get_weight);
    
    py::class_<Log_regression>(m, "Log_regression")
        .def(py::init<>())
        .def("load_data", static_cast<void (Log_regression::*)(const char *, const char *, const int)>(&Log_regression::load_data))
        .def("load_data", static_cast<void (Log_regression::*)(std::vector<double> const &, std::vector<double> const &, const int)>(&Log_regression::load_data))
        .def("train", &Log_regression::train)
        .def("test", static_cast<std::vector<int> (Log_regression::*)()>(&Log_regression::test))
        .def("test", static_cast<std::vector<int> (Log_regression::*)(std::vector<double> const &)>(&Log_regression::test))
        .def_property_readonly("weight", &Log_regression::get_weight);
    
    py::class_<Kmeans>(m, "Kmeans")
        .def(py::init<>())
        .def("load_data", static_cast<void (Kmeans::*)(const char *, const int)>(&Kmeans::load_data))
        .def("load_data", static_cast<void (Kmeans::*)(std::vector<double> const &, const int, const int)>(&Kmeans::load_data))
        .def("train", &Kmeans::train)
        .def("test", static_cast<std::vector<int> (Kmeans::*)(std::vector<double> const &)>(&Kmeans::test))
        .def_property_readonly("center", &Kmeans::get_center)
        .def_property_readonly("classes", &Kmeans::get_classes);
        
}