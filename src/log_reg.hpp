#pragma once
#include "model.hpp"
#include <algorithm>

class Log_regression : public Model
{
public:
    // std::vector<double> data;
    // std::vector<double> target;
    // int n_rows, n_cols, n_cls;
    // std::vector<double> weight;
    int n_cls;

    void load_data(const char *file_data, const char *file_target, const int ncls)
    {
        std::ifstream ifd(file_data, std::ifstream::in);
        std::ifstream ift(file_target, std::ifstream::in);
        if (!ifd || !ift)
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
        n_cls = ncls;
        // std::cout << "data size: " << n_rows << " * " << n_cols << std::endl;

        // target input
        while (getline(ift, line)) // n_rows * n_cls
        {
            // std::cout << line << std::endl;
            double tar_temp = std::stod(line);
            for (int i = 0; i < ncls; i++)
            {
                if (i == (int)tar_temp)
                    target.push_back(1.0);
                else
                    target.push_back(0.0);
            }
            // target.push_back(std::stod(line));
        }
        // for (auto x : target)
        //     std::cout << x << " ";
        // std::cout << std::endl;

        // target transpose
        std::vector<double> target_temp;
        target_temp.assign(target.begin(), target.end());
        int temp_idx = 0;
        for (int i = 0; i < n_cls; i++) // n_cls * n_rows
        {
            for (int j = 0; j < n_rows; j++)
            {
                target[temp_idx++] = target_temp[j * n_cls + i];
            }
        }
        // for (auto x : target)
        //     std::cout << x << " ";
        // std::cout << std::endl;

        // for (auto x : data) std::cout<< x << " ";
        // std::cout << "Loading data success!" << std::endl;
        loaded = true;
    }

    void load_data(const std::vector<double> &x, const std::vector<double> &y, const int ncls)
    {
        data = x;
        target = y;
        n_rows = y.size();
        n_cols = (int)(x.size() / y.size());
        n_cls = ncls;
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

        // for (int i = 0; i < n_cls; i++)
        //     for (int j = 0; j < n_cols; j++)
        //         weight.push_back(0.0); // n_cls * n_cols

        weight.assign(n_cls * n_cols, 0.0); // n_cls * n_cols

        for (int i = 0; i < iterations; i++)
        {
            std::vector<double> train_a(n_cls * n_rows, 0.0);
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n_cls, n_rows, n_cols, 1.0,
                        weight.data(), n_cols,
                        data.data(), n_cols, 0.0,
                        train_a.data(), n_rows);
            // std::cout << "testpoint 1\n";
            // for (auto x : train_a)
            //     std::cout << x << " ";
            // std::cout << std::endl;

            std::vector<double> a_exp(n_cls * n_rows);
            for (int i = 0; i < train_a.size(); i++)
                a_exp[i] = exp(train_a[i]);

            std::vector<double> train_y = a_exp; // n_cls * n_rows
            for (int i = 0; i < n_rows; i++)
            {
                double exp_sum = 0.0;
                for (int j = 0; j < n_cls; j++)
                    exp_sum += a_exp[j * n_rows + i];
                // std::cout << exp_sum << " ";
                for (int j = 0; j < n_cls; j++)
                    train_y[j * n_rows + i] /= exp_sum;
            }
            // std::cout << "\n\n";
            // std::cout << "testpoint 2\n";
            // for (auto x : train_y)
            //     std::cout << x << " ";
            // std::cout << std::endl;

            std::vector<double> diff(n_cls * n_rows); // n_cls * n_rows
            // for (int i = 0; i < train_y.size(); i++)
            //     diff[i] = train_y[i] - target[i];
            std::transform(train_y.begin(), train_y.end(), target.begin(), diff.begin(), std::minus<double>());

            std::vector<double> gradient(n_cls * n_cols, 0.0);
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n_cls, n_cols, n_rows, 1.0,
                        diff.data(), n_rows,
                        data.data(), n_cols, 0.0,
                        gradient.data(), n_cols);
            // std::cout << "testpoint 3\n";
            // for (auto x : gradient)
            //     std::cout << x << " ";
            // std::cout << std::endl;

            for (int i = 0; i < weight.size(); i++)
                weight[i] -= 0.0005 * gradient[i];

            // for (auto x : weight)
            //     std::cout << x << " ";
            // std::cout << std::endl;
        }
        // for (auto x : weight)
        //     std::cout << x << " ";
        // std::cout << std::endl;
        // std::cout << "Training success!" << std::endl;
        trained = true;
    }

    std::vector<int> test()
    {
        if (!trained)
        {
            std::cout << "Model is not trained!\n";
            return std::vector<int>();
        }

        std::vector<int> pred = test_core(data);

        // std::cout << "Testing success!" << std::endl;
        return pred;
    }

    std::vector<int> test(const std::vector<double> &test_data)
    {
        if (!trained)
        {
            std::cout << "Model is not trained!\n";
            return std::vector<int>();
        }
        if (test_data.size() % n_cols != 0)
        {
            std::cout << "incorrect data matrix size!!\n";
            return std::vector<int>();
        }

        std::vector<int> pred = test_core(test_data);

        // std::cout << "Testing success!" << std::endl;
        return pred;
    }

    std::vector<int>test_core(const std::vector<double> &test_data)
    {
        int n_rows_test = (int)(test_data.size() / n_cols);
        std::vector<double> test_a(n_cls * n_rows_test);
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n_cls, n_rows_test, n_cols, 1.0,
                    weight.data(), n_cols,
                    test_data.data(), n_cols, 0.0,
                    test_a.data(), n_rows_test);

        std::vector<double> test_a_exp; // n_cls * n_rows_test
        for (auto x : test_a)
            test_a_exp.push_back(exp(x));

        std::vector<double> test_y = test_a_exp; // n_cls * n_rows_test
        for (int i = 0; i < n_rows_test; i++)
        {
            double exp_sum = 0.0;
            for (int j = 0; j < n_cls; j++)
                exp_sum += test_a_exp[j * n_rows_test + i];
            for (int j = 0; j < n_cls; j++)
                test_y[j * n_rows_test + i] /= exp_sum;
        }

        std::vector<int> pred;
        double pred_temp;
        int idx_temp;
        for (int i = 0; i < n_rows_test; i++) // n_cls * n_rows_test
        {
            pred_temp = 0.0;
            idx_temp = 0;
            for (int j = 0; j < n_cls; j++)
            {
                if (test_y[j * n_rows_test + i] > pred_temp)
                {
                    idx_temp = j;
                    pred_temp = test_y[j * n_rows_test + i];
                }
            }
            pred.push_back(idx_temp);
        }
        return pred;
    }

    std::vector<double> get_weight() const { return weight; }
};
