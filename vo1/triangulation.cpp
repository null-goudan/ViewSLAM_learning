#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void find_feature_matches(  const Mat &img_1, const Mat &img_2,
                            std::vector<KeyPoint> &keypoints_1,
                            std::vector<KeyPoint> &keypoints_2,
                            std::vector<DMatch> &matches);

void pose_estimation_2d2d(  vector<KeyPoint> keypoints_1,
                            vector<KeyPoint> keypoints_2,
                            vector<DMatch> matches,
                            Mat &R, Mat &t);

void triangulation( const vector<KeyPoint> &keypoint_1,
                    const vector<KeyPoint> &keypoint_2,
                    const std::vector<DMatch> &matches,
                    const Mat& R, const Mat& t,
                    vector <Point3d>& points);

// convert function
Point2d pixel2cam(const Point2d &p, const Mat &K);

// used for draw the depth information
inline cv::Scalar get_color(float depth) {
  float up_th = 50, low_th = 10, th_range = up_th - low_th;
  if (depth > up_th) depth = up_th;
  if (depth < low_th) depth = low_th;
  return cv::Scalar(255 * depth / th_range, 0, 255 * (1 - depth / th_range));
}

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

    //-- triangulation
    vector<Point3d> points;
    triangulation(keypoints_1, keypoints_2, matches, R, t, points);

    //-- valid 
    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    Mat img1_plot = img_1.clone();
    Mat img2_plot = img_2.clone();
    for (int i = 0; i < matches.size(); i++) {
        float depth1 = points[i].z;
        cout << "depth: " << depth1 << endl;
        Point2d pt1_cam = pixel2cam(keypoints_1[matches[i].queryIdx].pt, K);
        cv::circle(img1_plot, keypoints_1[matches[i].queryIdx].pt, 2, get_color(depth1), 2);

        Mat pt2_trans = R * (Mat_<double>(3, 1) << points[i].x, points[i].y, points[i].z) + t;
        float depth2 = pt2_trans.at<double>(2, 0);
        cv::circle(img2_plot, keypoints_2[matches[i].trainIdx].pt, 2, get_color(depth2), 2);
    }
    cv::imshow("img 1", img1_plot);
    cv::imshow("img 2", img2_plot);
    cv::waitKey();

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

void triangulation( const vector<KeyPoint> &keypoint_1,
                    const vector<KeyPoint> &keypoint_2,
                    const std::vector<DMatch> &matches,
                    const Mat& R, const Mat& t,
                    vector <Point3d>& points)
{
    // The coordinates of the first camera
    Mat T1 = (Mat_<float>(3,4) <<
                1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0);
    
    // The coordinates of the second camera
    Mat T2 = (Mat_<float>(3,4) <<
        R.at<double>(0,0), R.at<double>(0,1), R.at<double>(0,2), t.at<double>(0,0),
        R.at<double>(1,0), R.at<double>(1,1), R.at<double>(1,2), t.at<double>(1,0),
        R.at<double>(2,0), R.at<double>(2,1), R.at<double>(2,2), t.at<double>(2,0));
    
    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

    vector<Point2f> point_1, point_2;
    // get the pixels' location of camara 
    for(DMatch m: matches){
        // contert points to pixel of camara
        point_1.push_back(pixel2cam(keypoint_1[m.queryIdx].pt, K));
        point_2.push_back(pixel2cam(keypoint_2[m.queryIdx].pt, K));
    }

    // triangulate
    Mat points_4d;
    cv::triangulatePoints(T1, T2, point_1, point_2, points_4d);

    // convert to Non-homogeneous coordinates
    for(int i = 0; i < points_4d.cols; ++i){
        Mat x = points_4d.col(i);
        x /= x.at<float>(3,0);
        Point3d p(
            x.at<float>(0, 0),
            x.at<float>(1, 0),
            x.at<float>(2, 0)
        );
        points.push_back(p);
    }
}
