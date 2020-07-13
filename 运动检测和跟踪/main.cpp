#include <QtCore/QCoreApplication>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

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

	Mat frame1, frame2, diff, dstImg, grayImg;
	auto cap = VideoCapture("../image/test.avi");
	cap >> frame1 >> frame2;
	Mat kernel = getStructuringElement(MORPH_RECT, { 9,9 });
	vector<vector<Point>> cons;
	do
	{
		absdiff(frame1, frame2, diff);
		cvtColor(diff, diff, COLOR_BGR2GRAY);
		GaussianBlur(diff, diff, { 5,5 }, 0);
		threshold(diff, diff, 50, 255, THRESH_BINARY);
		dilate(diff, grayImg, kernel);
		findContours(grayImg, cons, RETR_TREE, CHAIN_APPROX_SIMPLE);
		for (auto& i : cons)
		{
			auto rect = boundingRect(i);
			if (contourArea(i) < 900)
				continue;
			rectangle(frame1, rect, { 0,0,255 }, 2);

		}

		imshow("test", frame1);
		frame2.copyTo(frame1);

		waitKey(3);
	} while (cap.read(frame2));

	return a.exec();
}
