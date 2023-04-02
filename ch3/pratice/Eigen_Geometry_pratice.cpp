#include <iostream>
#include <Eigen/Core>
#include <Eigen/Geometry>

using namespace std;
using namespace Eigen;

int main(){
    Quaterniond q1(0.55, 0.3, 0.2, 0.2);
    Vector3d t1(0.7, 1.1, 0.2);

    Quaterniond q2(-0.1, 0.3, -0.7, 0.2);
    Vector3d t2(-0.1, 0.4, 0.8);

    Vector3d p1(0.5, -0.1, 0.2);

    // transfer p1 to world
    Matrix3d R1 = q1.toRotationMatrix();
    Matrix4d T1 = Matrix4d::Identity();
    T1.block<3,3>(0,0) = R1;
    T1.block<3,1>(0,3) = t1;
    Vector4d p1_h = Vector4d(p1(0), p1(1), p1(2), 1);
    Vector4d p1_w = T1*p1_h;
    
    cout<<"p1 in world: "<<p1_w.block<3,1>(0,0)<<endl;

    // transfer p1 in world to 2nd  
    Matrix3d R2 = q2.toRotationMatrix();
    Matrix4d T2 = Matrix4d::Identity();
    T1.block<3,3>(0,0) = R2;
    T1.block<3,1>(0,3) = t2;
    Vector4d p2_2_h = T2.inverse()*p1_w;

    cout << "小萝卜一号坐标系下的点p1：" << endl << p1 << endl;
    cout << "小萝卜二号坐标系下的点p1_2：" << endl << p2_2_h.block<3,1>(0,0) << endl;
    return 0;
}