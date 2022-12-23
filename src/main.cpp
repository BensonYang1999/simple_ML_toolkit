#include "regression.hpp"
#include "log_reg.hpp"
#include "kmeans.hpp"
#include <chrono>
using namespace std;
using namespace chrono;

int main()
{
    // regression
    // Regression test;
    // test.load_data("test_data/x.csv", "test_data/y.csv");

    // log_reg
    // Log_regression test;
    // test.load_data("test_data/x_train.csv", "test_data/t_train.csv", 3);

    // kmeans
    Kmeans test;
    test.load_data("test_data/cat.csv", 3);

    // cout << "Reading Success\n";
    auto start = system_clock::now();
    // test.train();  // regression && log_reg
    test.train(100); // kmeans
    auto end = system_clock::now();
    // cout << "Training Success\n";
    auto duration = duration_cast<nanoseconds>(end - start);
    cout << "Duration: ";
    cout << setprecision(10) << double(duration.count()) * nanoseconds::period::num / nanoseconds::period::den;
    cout << " seconds\n";

    // regression && log_reg
    // std::vector<double> weight = test.get_weight();
    // std::cout << "weight: ";
    // for (auto x : weight)
    //     std::cout << x << " ";
    // std::cout << std::endl;
    // kmeans
    std::vector<std::vector<double>> center = test.get_center();
    std::vector<int> classes = test.get_classes();
    cout << "-------------- center --------------\n";
    for (auto xx : center)
    {
        for (auto yy : xx)
            cout << setw(10) << yy * 255 << " ";
        cout << endl;
    }

    // regression
    // std::vector<double> pred = test.test();
    // log_reg
    // std::vector<int> pred = test.test();

    // std::cout << "predict: ";
    // for (auto x : pred)
    //     std::cout << x << " ";
    // std::cout << std::endl;
}