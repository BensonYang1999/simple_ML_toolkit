#pragma once
#include "model.hpp"

class Regression : public Model
{
public:
    // std::vector<double> data;
    // std::vector<double> target;
    // int n_rows, n_cols;
    // std::vector<double> weight;

    void load_data(const char *file_data, const char *file_target)
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
        while (getline(ift, line))
        {
            // std::cout << line << std::endl;
            target.push_back(std::stod(line));
        }
        // for (auto x : target) {
        //     std::cout << x << " ";
        // }
        // std::cout << std::endl;

        // 2D to 1D
        n_rows = temp_data.size();
        n_cols = temp_data[0].size() + 1;
        // std::cout << "data size: " << n_rows << " * " << n_cols << std::endl;

        for (auto i : temp_data)
        {
            data.push_back(1.0);
            for (auto j : i)
            {
                data.push_back(j);
            }
        }

        // for (auto x : data) std::cout<< x << " ";
        // std::cout << data.size() << std::endl;
        std::cout << "Loading data success!" << std::endl;
        loaded = true;
    }

    void load_data(const std::vector<double> &x, const std::vector<double> &y)
    {
        data = x;
        target = y;
        n_rows = y.size();
        n_cols = (int)(x.size() / y.size());
        std::cout << "Loading data success!" << std::endl;
        loaded = true;
    }

    void train()
    {
        if (!loaded)
        {
            std::cout << "Data is not loaded!\n";
            return;
        }

        std::vector<double> phi(n_cols * n_cols, 0.0); // n_cols * n_cols

        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, n_cols, n_cols, n_rows, 1.0,
                    data.data(), n_cols,
                    data.data(), n_cols, 0.0,
                    phi.data(), n_cols);

        // std::cout << "phi" << std::endl;
        // for (auto x : phi)
        //     std::cout << x << " ";
        // std::cout << std::endl;

        // SVD method
        // std::vector<double> phi_inv(n_cols * n_cols, 0.0);
        // std::vector<double> s(n_cols, 0.0);
        // std::vector<double> u(n_cols * n_cols, 0.0);
        // std::vector<double> vt(n_cols * n_cols, 0.0);
        // std::vector<double> superb(n_cols, 0.0);
        // LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'A', 'A',
        //                n_cols, n_cols, phi.data(), n_cols,
        //                s.data(), u.data(), n_cols, vt.data(), n_cols,
        //                superb.data());
        // // for (auto x : s)
        // //     std::cout << x << " ";
        // // std::cout << std::endl;
        // // std::cout << std::endl;

        // for (int i = 0; i < s.size(); i++)
        // {
        //     if (s[i] != 0.0 && s[i] > 0.01)
        //         s[i] = 1 / s[i];
        //     else
        //         s[i] = 0;
        // }
        // std::vector<double> s_diag(n_cols * n_cols, 0.0);
        // for (int i = 0; i < n_cols; i++)
        //     s_diag[i * i] = s[i];
        // std::vector<double> phi_inv_temp(n_cols * n_cols, 0.0);
        // cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n_cols, n_cols, n_cols, 1.0,
        //             s_diag.data(), n_cols,
        //             u.data(), n_cols, 0.0,
        //             phi_inv_temp.data(), n_cols);
        // cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, n_cols, n_cols, n_cols, 1.0,
        //             vt.data(), n_cols,
        //             phi_inv_temp.data(), n_cols, 0.0,
        //             phi_inv.data(), n_cols);

        // LAPACK matrix inverse
        std::vector<double> phi_inv = phi;
        int ipiv[n_cols];
        LAPACKE_dgetrf(LAPACK_ROW_MAJOR, n_cols, n_cols, phi_inv.data(), n_cols, ipiv);
        LAPACKE_dgetri(LAPACK_ROW_MAJOR, n_cols, phi_inv.data(), n_cols, ipiv);

        // std::cout << "phi_inv" << std::endl;
        // for (auto x : phi_inv)
        //     std::cout << x << " ";
        // std::cout << std::endl;

        std::vector<double> phi_s3(n_cols * n_rows, 0.0); // n_cols * n_rows
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n_cols, n_rows, n_cols, 1.0,
                    phi_inv.data(), n_cols,
                    data.data(), n_cols, 0.0,
                    phi_s3.data(), n_rows);
        // for (auto x : phi_s3)
        //     std::cout << x << " ";
        // std::cout << std::endl;

        // weght(n_cols * 1, 0.0); // n_cols * 1
        // for (int t = 0; t < n_cols; t++)
        //     weight.push_back(0.0); // n_cols * 1
        weight.assign(n_cols, 0.0); // n_cols * 1
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n_cols, 1, n_rows, 1.0,
                    phi_s3.data(), n_rows,
                    target.data(), 1, 0.0,
                    weight.data(), 1);
        // std::cout << "weight: ";
        // for (auto x : weight)
        //     std::cout << x << " ";
        // std::cout << std::endl;

        std::cout << "Training success!" << std::endl;
        trained = true;
    }

    std::vector<double> test()
    {
        if (!trained)
        {
            std::cout << "Model is not trained!\n";
            return std::vector<double>();
        }
        std::vector<double> test_y(target.size(), 0.0);
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n_rows, 1, n_cols, 1.0,
                    data.data(), n_cols,
                    weight.data(), 1, 0.0,
                    test_y.data(), 1);
        // std::cout << "testing result: ";
        // for (auto x : test_y)
        //     std::cout << x << " ";
        // std::cout << std::endl;
        std::cout << "Testing success!" << std::endl;
        return test_y;
    }

    std::vector<double> test(const std::vector<double> &test_data)
    {
        if (!trained)
        {
            std::cout << "Model is not trained!\n";
            return std::vector<double>();
        }
        if (test_data.size() % n_cols != 0)
        {
            std::cout << "incorrect data matrix size!!\n";
            return std::vector<double>();
        }
        std::vector<double> test_y(target.size(), 0.0);
        int rows = test_data.size() / n_cols;
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rows, 1, n_cols, 1.0,
                    test_data.data(), n_cols,
                    weight.data(), 1, 0.0,
                    test_y.data(), 1);
        // std::cout << "testing result: ";
        // for (auto x : test_y)
        //     std::cout << x << " ";
        // std::cout << std::endl;
        std::cout << "Testing success!" << std::endl;
        return test_y;
    }

    std::vector<double> get_weight() const { return weight; }
};
