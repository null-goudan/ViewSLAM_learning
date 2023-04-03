#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace Eigen;

// y = exp( a*x*x + b*x + c)

int main(int argc, char **argv){

    double ar = 1.0, br = 2.0, cr = 1.0;    // the real Parameter
    double ae = 2.0, be = -1.0, ce = 5.0;   // the estimatical Parameter
    int N = 100;                            // the numbers of Data
    double w_sigma = 2.0;                   // the Noise
    double inv_sigma = 1.0 / w_sigma;
    cv::RNG rng;

    vector<double> x_data, y_data; // data
    for(int i = 0; i<N; i++){   
        double x = i / 100.0;
        x_data.push_back(x);
        y_data.push_back(exp(ar * x * x + br * x + cr) + rng.gaussian(w_sigma * w_sigma));
    }

    // start iteration of Gussion
    int iterations = 100; // the numbers of iterations
    double cost = 0, lastCost = 0; // this iteration's cost and the last iteration's cost

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    for(int iter = 0; iter < iterations; ++iter){
        Matrix3d H = Matrix3d::Zero();  // Hassion matrix = J^T W^{-1}   J in Newton
        Vector3d b = Vector3d::Zero();  // bias
        cost = 0; // the cost of this iteration 

        for (int i = 0; i<N; ++i){
            double xi = x_data[i], yi = y_data[i];
            double error = yi - exp(ae* xi * xi + be *xi + ce);

            Vector3d J;  // Jacobian Matrix
            J[0] = -xi * xi * exp(ae* xi * xi + be *xi + ce);      // de/da
            J[1] = -xi * exp(ae* xi * xi + be *xi + ce);           // de/db
            J[2] = -exp(ae* xi * xi + be *xi + ce);                // de/dc

            H += inv_sigma * inv_sigma * J * J.transpose();
            b += -inv_sigma * inv_sigma * error * J;

            cost += error * error;
        }
        // solve the linear function: Hx = b;
        Vector3d dx = H.ldlt().solve(b);
        if(isnan(dx[0])){
            cout << "result is nan" << endl;
            break;
        }

        if(iter > 0 && cost >= lastCost){
            cout << "cost: " << cost << ">= last cost: " << lastCost << ", break" << endl;
            break;
        }

        ae += dx[0];
        be += dx[1];
        ce += dx[2];

        lastCost = cost;

        cout << "total cost: " << cost <<",\t\tupdate: " << dx.transpose() 
        << ",\t\testimate params: " << ae << "," << be <<"," << ce << "," << endl;
    }

    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);

    cout << "solve time cost = " << time_used.count() << " seconds. " << endl;
    cout << "estimated abc = " << ae << ", " << be << ", " << ce << endl;
    return 0;
}
