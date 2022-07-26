//
// Created by goudan on 7/19/22.
//
#include <iostream>
#include <cmath>

using namespace std;

#include <Eigen/Core>
#include <Eigen/Geometry>

using namespace Eigen;

int main(int argc, char** argv){
    Matrix3d rotation_matrix = Matrix3d::Identity();
    AngleAxisd rotation_vector(M_PI / 4, Vector3d(0, 0, 1));
    cout.precision(3);
    cout<<"rotation matrix =\n"<<rotation_vector.matrix() <<endl;

    rotation_matrix = rotation_vector.toRotationMatrix();

    cout<<"rotation matrix =\n"<<rotation_matrix<<endl;

    // AngleAxis transform
    Vector3d v(1, 0, 0);
    Vector3d v_rotated = rotation_vector * v;
    cout << "(1, 0, 0) after rotation (by angle axis) = " <<v_rotated.transpose()<<endl;
    v_rotated = rotation_matrix * v;
    cout << "(1, 0, 0) after rotation (by matrix) = " << v_rotated.transpose()<<endl;

    // Euler
    Vector3d euler_angles = rotation_matrix.eulerAngles(2, 1, 0); // ZYX
    cout<< "yaw pitch roll = "<<euler_angles.transpose()<<endl;

    // ou transpose
    Isometry3d T = Isometry3d::Identity();
    T.rotate(rotation_vector);
    T.pretranslate(Vector3d(1, 3, 4));
    cout<< "Transform matrix = \n"<<T.matrix()<<endl;

    // Transform use matrix
    Vector3d v_transformed = T*v;
    cout<< "v_transformed = " << v_transformed.transpose()<<endl;

    // Quaternion
    Quaterniond q = Quaterniond(rotation_vector);
    cout << "quaternion from rotation vector = "<<q.coeffs().transpose()<<endl;
    q = Quaterniond(rotation_matrix);
    cout << "quaternion form rotation matrix = "<<q.coeffs().transpose()<<endl;

    // use override mul
    v_rotated = q * v; // in math : q * v * q^{-1}
    cout << "(1, 0, 0) after rotation = "<<v_rotated.transpose() << endl;

    // use vector
    cout<< "should be equal to " << (q * Quaterniond(0, 1, 0, 0) * q.inverse()).coeffs().transpose()<<endl;

    return 0;
}