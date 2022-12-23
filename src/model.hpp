#pragma once
#include "utils.hpp"

class Model
{
public:
    std::vector<double> data;
    std::vector<double> target;
    int n_rows, n_cols;
    std::vector<double> weight;

    bool trained = false;
    bool loaded = false;

    // virtual void train() = 0;
    // virtual void test() = 0;

    // std::vector<double> get_weight() const {return weight;}
};

