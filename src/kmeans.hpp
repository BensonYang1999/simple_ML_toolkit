#pragma once
#include "utils.hpp"
#include "model.hpp"

// #include <cstdlib>
#include <random>
#include <ctime>
#include <cmath>
#include <set>

class Kmeans : public Model
{
public:
    std::vector<std::vector<double>> center;
    std::vector<int> classes;
    int K;
    double total_distance;

    void load_data(const char *file_data, const int nk)
    {
        std::ifstream ifd(file_data, std::ifstream::in);
        if (!ifd)
        {
            std::cout << "Reading file failed!\n";
            return;
        }
        std::string line, substr;
        std::vector<std::vector<double>> temp_data;
        int pos;

        // data input
        while (getline(ifd, line))
        {
            // std::cout << line << std::endl;
            std::vector<double> temp;
            while ((pos = line.find(',')) != std::string::npos)
            {
                substr = line.substr(0, pos);
                // std::cout << substr << " ";
                temp.push_back(std::stod(substr));
                line.erase(0, pos + 1);
            }
            temp.push_back(std::stod(line));
            temp_data.push_back(temp);
        }
        // data 2D to 1D && normalization
        for (auto i : temp_data) // n_rows * n_cols
        {
            for (auto j : i)
            {
                data.push_back(j);
            }
        }
        // for (auto x : data)
        //     std::cout << x << " ";
        // std::cout << std::endl;

        n_rows = temp_data.size();
        n_cols = temp_data[0].size();
        K = nk;
        // std::cout << "data size: " << n_rows << " * " << n_cols << std::endl;
        // std::cout << "groups: " << K << std::endl;

        classes.resize(n_rows);

        // std::cout << "Loading data success!" << std::endl;
        loaded = true;
    }

    void load_data(const std::vector<double> &x, const int nrows, const int nk)
    {
        data = x;
        n_rows = nrows;
        n_cols = (int)(x.size() / nrows);
        K = nk;
        classes.resize(n_rows);
        // std::cout << "Loading data success!" << std::endl;
        loaded = true;
    }

    void train(const int iterations)
    {
        if (!loaded)
        {
            std::cout << "Data is not loaded!\n";
            return;
        }

        // random generate center points
        std::set<int> point_idx;
        std::vector<double> point;
        point.resize(n_cols);
        srand((unsigned)time(0));
        for (int i = 0; i < K; i++)
        {
            int idx = rand() % n_rows;
            // int idx = rdn(generator);
            // int idx = (n_rows / (K + 1)) * (i + 1);
            // std::cout << idx << " ";
            while (point_idx.count(idx) != 0)
                idx = rand() % n_rows;
            point_idx.insert(idx);
            for (int j = 0; j < n_cols; j++)
                point[j] = data[idx * n_cols + j];
            center.push_back(point);
        }
        // test center
        // std::cout << "Random center before training" << std::endl;
        // for (auto xx : center)
        // {
        //     for (auto yy : xx)
        //         std::cout << std::setw(10) << yy * 255 << " ";
        //     std::cout << std::endl;
        // }
        double old_total_dist = 1e100;

        for (int iter = 0; iter < iterations; iter++)
        {
            // find nearest center(group)
            // std::cout << "go into iteration" << std::endl;
            std::vector<std::vector<double>> distance(n_rows, std::vector<double>(K, 0));
            // double distance[n_rows][K] = {0.0}; // segmentation fault
            // std::cout << "distance generate success" << std::endl;
            // int classes[n_rows];
            double temp_dist = 1e100;
            total_distance = 0.0;
            for (int i = 0; i < n_rows; i++)
            {
                temp_dist = 1e100;
                for (int k = 0; k < K; k++)
                {
                    for (int j = 0; j < n_cols; j++)
                    {
                        distance[i][k] += pow(data[i * n_cols + j] - center[k][j], 2.0);
                    }
                    if (distance[i][k] < temp_dist)
                    {
                        temp_dist = distance[i][k];
                        classes[i] = k;
                    }
                }
                total_distance += temp_dist;
            }

            if (total_distance < old_total_dist)
                old_total_dist = total_distance;
            else if (total_distance - old_total_dist < 1)
            {
                std::cout << "Early stop in iteration " << iter << std::endl;
                break;
            }
            // std::cout << "find nearest center" << std::endl;

            // find new center of each group
            int class_count[K] = {0};
            for (int i = 0; i < K; i++)
                std::fill(center[i].begin(), center[i].end(), 0);
            for (int i = 0; i < n_rows; i++)
            {
                for (int j = 0; j < n_cols; j++)
                    center[classes[i]][j] += data[i * n_cols + j];
                class_count[classes[i]] += 1;
            }
            for (int k = 0; k < K; k++)
                for (int j = 0; j < n_cols; j++)
                    center[k][j] /= class_count[k];
        }

        // test center
        // std::cout << "Center after training" << std::endl;
        // for (auto xx : center)
        // {
        //     for (auto yy : xx)
        //         std::cout << std::setw(10) << yy * 255 << " ";
        //     std::cout << std::endl;
        // }

        // std::cout << "Training success!" << std::endl;
        trained = true;
    }

    std::vector<std::vector<double>> get_center() const { return center; }
    std::vector<int> get_classes() const { return classes; }
};
