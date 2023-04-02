#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>


using namespace Eigen;
using namespace std;


int main(){
    // Ax=b
    MatrixXd A = MatrixXd::Random(100, 100);
    VectorXd b = VectorXd::Random(100);

    VectorXd x_qr = A.householderQr().solve(b);
    VectorXd x_cholesky = A.llt().solve(b);

    cout << "QR分解求解x:" << endl << x_qr << endl;
    cout << "Cholesky分解求解x:" << endl << x_cholesky << endl;

    return 0;
}