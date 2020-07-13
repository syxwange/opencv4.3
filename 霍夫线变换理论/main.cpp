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


int main(int argc, char* argv[])
{
	QCoreApplication a(argc, argv);

	Mat src, gray, temp, edges, dst;
	src = imread("../image/sudoku.png");
	cvtColor(src, gray, COLOR_BGR2GRAY);
	Canny(gray, edges, 50, 150);

	vector<Vec2f> lines;
	vector<Vec4i> liness;

	HoughLines(edges, lines, 1, CV_PI / 180, 200);
	HoughLinesP(edges, liness, 1, CV_PI / 180, 200, 100, 3);
	for (auto& i : liness)
	{

		line(src, { i[0],i[1] }, { i[2],i[3] }, Scalar(0, 0, 255), 3, LINE_AA);
	}

	return a.exec();
}
