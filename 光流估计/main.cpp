#include <QtCore/QCoreApplication>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <opencv2/video/tracking.hpp>


#ifdef _DEBUG
#pragma comment (lib,"opencv_world430d")
#else
#pragma comment (lib,"opencv_world430")
#endif 

using namespace std;
using namespace cv;

Point2f point;
bool addRemovePt = false;
static void onMouse(int event, int x, int y, int /*flags*/, void* /*param*/)
{
    if (event == EVENT_LBUTTONDOWN)
    {
        point = Point2f((float)x, (float)y);
        addRemovePt = true;
    }
}

int main(int argc, char* argv[])
{
    QCoreApplication a(argc, argv);

    auto cap = VideoCapture("../image/test.avi");
    Mat frame, grayImg, tempImg, dstImg;
    cap.read(tempImg);
    cvtColor(tempImg, grayImg, COLOR_BGR2GRAY);

    vector<Point2f> points[2], goodNew, goodOld;
    goodFeaturesToTrack(grayImg, points[0], 100, 0.3, 7);

    Mat mask = Mat::zeros(tempImg.size(), tempImg.type());
    vector<uchar> status;
    vector<float> err;
    while (cap.read(frame))
    {
        cvtColor(frame, dstImg, COLOR_BGR2GRAY);
        goodNew.clear();
        goodOld.clear();
        calcOpticalFlowPyrLK(grayImg, dstImg, points[0], points[1], status, err);
        for (int i = 0; i < points[1].size(); i++)
        {
            if (status[i] == 1)
            {
                goodNew.push_back(points[1][i]);
                goodOld.push_back(points[0][i]);
            }
        }
        for (auto& j : goodOld)
        {
            circle(frame, j, 5, { 0,0,255 }, -1);
        }
        imshow("111", frame);
        waitKey(170);


        std::swap(points[1], points[0]);
        cv::swap(grayImg, dstImg);

    }

    /*
    //VideoCapture cap;
    //TermCriteria termcrit(TermCriteria::COUNT | TermCriteria::EPS, 20, 0.03);
    //Size subPixWinSize(10, 10), winSize(31, 31);
    //const int MAX_COUNT = 500;
    //bool needToInit = false;
    //bool nightMode = false;

    //cap.open("../image/test.avi");
    //if (!cap.isOpened())
    //{
    //    cout << "Could not initialize capturing...\n";
    //    return 0;
    //}
    //namedWindow("LK Demo", 1);
    //setMouseCallback("LK Demo", onMouse, 0);
    //Mat gray, prevGray, image, frame;
    //vector<Point2f> points[2];
    //for (;;)
    //{
    //    cap >> frame;
    //    if (frame.empty())
    //        break;
    //    frame.copyTo(image);
    //    cvtColor(image, gray, COLOR_BGR2GRAY);
    //    if (nightMode)
    //        image = Scalar::all(0);
    //    if (needToInit)
    //    {
    //        // automatic initialization
    //        goodFeaturesToTrack(gray, points[1], MAX_COUNT, 0.01, 10, Mat(), 3, 3, 0, 0.04);
    //        cornerSubPix(gray, points[1], subPixWinSize, Size(-1, -1), termcrit);
    //        addRemovePt = false;
    //    }
    //    else if (!points[0].empty())
    //    {
    //        vector<uchar> status;
    //        vector<float> err;
    //        if (prevGray.empty())
    //            gray.copyTo(prevGray);
    //        calcOpticalFlowPyrLK(prevGray, gray, points[0], points[1], status, err, winSize,
    //            3, termcrit, 0, 0.001);
    //        size_t i, k;
    //        for (i = k = 0; i < points[1].size(); i++)
    //        {
    //            if (addRemovePt)
    //            {
    //                if (norm(point - points[1][i]) <= 5)
    //                {
    //                    addRemovePt = false;
    //                    continue;
    //                }
    //            }
    //            if (!status[i])
    //                continue;
    //            points[1][k++] = points[1][i];
    //            circle(image, points[1][i], 3, Scalar(0, 255, 0), -1, 8);
    //        }
    //        points[1].resize(k);
    //    }
    //    if (addRemovePt && points[1].size() < (size_t)MAX_COUNT)
    //    {
    //        vector<Point2f> tmp;
    //        tmp.push_back(point);
    //        cornerSubPix(gray, tmp, winSize, Size(-1, -1), termcrit);
    //        points[1].push_back(tmp[0]);
    //        addRemovePt = false;
    //    }
    //    needToInit = false;
    //    imshow("LK Demo", image);
    //    char c = (char)waitKey(10);
    //    if (c == 27)
    //        break;
    //    switch (c)
    //    {
    //    case 'r':
    //        needToInit = true;
    //        break;
    //    case 'c':
    //        points[0].clear();
    //        points[1].clear();
    //        break;
    //    case 'n':
    //        nightMode = !nightMode;
    //        break;
    //    }
    //    std::swap(points[1], points[0]);
    //    cv::swap(prevGray, gray);
    //}

    */

    return a.exec();
}
