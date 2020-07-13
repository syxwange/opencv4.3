#include <QtCore/QCoreApplication>
#include <iostream>
#include<opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <qstring.h>
#include <opencv2/xfeatures2d.hpp>

using namespace cv;
using namespace std;

#ifdef _DEBUG
#pragma comment (lib,"opencv_world430d")
#else
#pragma comment (lib,"opencv_world430")
#endif 

int threhod1{}, threhod2{};
Mat gImg;
void cbCanny(int pos, void* userdata)
{
	Mat canny;
	Canny(gImg, canny, threhod1, threhod2);
	imshow("test", canny);
}

int main(int argc, char* argv[])
{
	QCoreApplication a(argc, argv);

	Mat src, gray, temp, edges, dst;
	gImg = imread("../image/sudoku.png", IMREAD_GRAYSCALE);
	
	namedWindow("test");
	createTrackbar("threhod1", "test", &threhod1, 255, cbCanny);
	createTrackbar("threhod2", "test", &threhod2, 255, cbCanny);
	cbCanny(0, 0);

	return a.exec();
}