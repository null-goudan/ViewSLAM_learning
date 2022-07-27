//
// Created by goudan on 7/26/22.
//
#include <iostream>
#include <chrono>
using namespace std;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>

int main(int argc, char **argv){
    cv::Mat image;
    image = cv::imread(argv[1]);

    if(image.data == nullptr){
        cerr<<"file "<<argv[1]<<" not exist!"<<endl;
        return 0;
    }

    cout << "image weight: "<<image.cols<<" height: "<<image.rows<<endl
        <<" channels: "<<image.channels()<<endl;



    if(image.type() != CV_8UC1 && image.type() != CV_8UC3){
        cout<<"please input a Color image or Gray image"<<endl;
        return 0;
    }

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    for(size_t y = 0;y<image.rows;++y){
        auto * row_ptr = image.ptr<unsigned char>(y);
        for(size_t x = 0;x<image.cols;++x){
            unsigned char* data_ptr = &row_ptr[x * image.channels()];
            for(int c = 0; c< image.channels();++c){
                unsigned char data = data_ptr[c];
            }
        }
    }

    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "transfer used "<<time_used.count()<<" s."<<endl;
    cv::imshow("image", image);
    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}