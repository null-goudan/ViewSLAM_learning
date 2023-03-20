#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

using namespace std;
using namespace cv;

void find_feature_matches(const Mat &img_1, const Mat &img_2,
                          std::vector<KeyPoint> &keypoints_1,
                          std::vector<KeyPoint> &keypoints_2,
                          std::vector<DMatch> &matches);

void pose_estimation_2d2d(  vector<KeyPoint> keypoints_1,
                            vector<KeyPoint> keypoints_2,
                            vector<DMatch> matches,
                            Mat &R, Mat &t);

Point2d pixel2cam(const Point2d &p, const Mat &K);

int main(int argc, char** argv){
    if(argc != 3)
    {
        cout<<"usage featrue_extraction img1 img2"<<endl;
        return 1;
    }
    // load image
    Mat img_1 = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    Mat img_2 = imread(argv[2], CV_LOAD_IMAGE_COLOR);
    assert(img_1.data!=nullptr && img_2.data!= nullptr);

    vector<KeyPoint> keypoints_1, keypoints_2;
    vector<DMatch> matches;
    find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
    cout << "Totally, find " << matches.size() << " groups matches." << endl;

    // estimate the movement between of two pictures
    Mat R, t;
    pose_estimation_2d2d(keypoints_1, keypoints_2, matches, R, t);

    // validation: E=t^R*scale 
    Mat t_x =
    (Mat_<double>(3, 3) << 0, -t.at<double>(2, 0), t.at<double>(1, 0),
    t.at<double>(2, 0), 0, -t.at<double>(0, 0),
    -t.at<double>(1, 0), t.at<double>(0, 0), 0);

    cout << "t^R=" << endl << t_x * R << endl;

    // Ep ipo lar Const raint
    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    for (DMatch m: matches) {
        Point2d pt1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
        Mat y1 = (Mat_<double>(3, 1) << pt1.x, pt1.y, 1);
        Point2d pt2 = pixel2cam(keypoints_2[m.trainIdx].pt, K);
        Mat y2 = (Mat_<double>(3, 1) << pt2.x, pt2.y, 1);
        Mat d = y2.t() * t_x * R * y1;
        cout << "epipolar constraint = " << d << endl;
    }
    return 0;
}

Point2d pixel2cam(const Point2d &p, const Mat &K) {
  return Point2d
    (
      (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
      (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
    );
}

void pose_estimation_2d2d(  vector<KeyPoint> keypoints_1,
                            vector<KeyPoint> keypoints_2,
                            vector<DMatch> matches,
                            Mat &R, Mat &t)
{
    // Camera internal parameters
    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

    // convert the matches to vector<Point2f>   
    std::vector<Point2f> points1;
    std::vector<Point2f> points2;

    for(int i=0;i<matches.size(); ++i){
        points1.push_back(keypoints_1[matches[i].queryIdx].pt);
        points2.push_back(keypoints_2[matches[i].trainIdx].pt);
    }

    // caculate fundamental matrix
    Mat fundamental_matrix;
    fundamental_matrix = findFundamentalMat(points1, points2, CV_FM_8POINT); // The API of OpenCV which use to find fundamental matrix by 8-Points-Alogithm
    cout << "fundamental_matrix is" << endl << fundamental_matrix << endl;

    // caculate essential matrix
    Point2d pricipal_point(325.1, 249.7); // // Camera light center (Calibration value)
    double focal_length = 521; //Camera focal length
    Mat essential_matrix;
    essential_matrix = findEssentialMat(points1, points2, focal_length, pricipal_point); //The API of OpenCV which use to find Essential matrix

    // Recover rotation and translation information from the essential matrix
    recoverPose(essential_matrix, points1, points2, R, t, focal_length, pricipal_point);
    cout << "R is " << endl << R << endl;
    cout << "t is " << endl << t << endl;
}

// ORB
void find_feature_matches(const Mat &img_1, const Mat &img_2,
                          std::vector<KeyPoint> &keypoints_1,
                          std::vector<KeyPoint> &keypoints_2,
                          std::vector<DMatch> &matches)
{
    // initialization 
    Mat descriptors_1, descriptors_2;
    Ptr<FeatureDetector> detector = ORB::create(); // featuredector of ORB - FAST 
    // Ptr<>: the automatic ptr in opencv use the template technology 
    Ptr<DescriptorExtractor> descriptor = ORB::create(); // miao shu zi 
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming"); // Hamming match method

    // Firstly: Detect Oriented Fast 
    // chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    detector->detect(img_1, keypoints_1);
    detector->detect(img_2, keypoints_2);

    // Secondly: compute the BRIEF descriptors using Oriented Fast detectors
    descriptor->compute(img_1, keypoints_1, descriptors_1);
    descriptor->compute(img_2, keypoints_2, descriptors_2);

    Mat outimg1;
    drawKeypoints(img_1, keypoints_1, outimg1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
    imshow("ORB features", outimg1);

    // match
    // vector<DMatch> matches;
    matcher->match(descriptors_1, descriptors_2, matches);

    // filter   to find the minimun distance and maximum distance
    auto min_max = minmax_element(matches.begin(), matches.end(),
                                [](const DMatch &m1, const DMatch &m2) { return m1.distance < m2.distance; });
    double min_dist = min_max.first->distance;
    double max_dist = min_max.second->distance;

    printf("-- Max dist : %f \n", max_dist);
    printf("-- Min dist : %f \n", min_dist);

    std::vector<DMatch> good_matches;
    for (int i = 0; i < descriptors_1.rows; i++) {
        if (matches[i].distance <= max(2 * min_dist, 30.0)) {
        good_matches.push_back(matches[i]);
        }
    }
}
