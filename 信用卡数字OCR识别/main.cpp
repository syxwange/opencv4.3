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

const string testFile = "../image/111.png";
const string cardFileName = "../image/credit_card_05.png";
const string numFileName = "../image/ocr_a_reference.png";

////找到信用卡0-9中每个数字的图片，用来做模板匹配
void findNumber(const Mat& thresholdCard, vector<Mat>& numbers)
{

	vector<vector<Point>>  numCons;

	//找到轮廓
	//morphologyEx(thresholdCard, thresholdCard, MORPH_DILATE, getStructuringElement(MORPH_RECT, { 3,3 }));
	findContours(thresholdCard, numCons, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	//numberPic.copyTo(temp);
	//drawContours(temp, numCons, -1, Scalar(0, 0, 255), 3);

	////////////找到轮廓的最小外接矩形，并根据矩形的第一个点的X坐标排序，这样把0-9个数字找出来。
	vector<Rect> rects;
	for (auto& i : numCons)
	{
		//找到最小外接矩形
		rects.push_back(boundingRect(i));
	}
	//对最小外接矩形第一个点的X坐标排序
	std::sort(rects.begin(), rects.end(), [](Rect a, Rect b) {return a.x < b.x; });
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	////////////根据外接矩形，找到0-9每个数字的图片
	for (auto& i : rects)
	{
		Mat temp;
		resize(thresholdCard(Rect(i.x, i.y, i.width, i.height)), temp, Size(57, 88));
		numbers.push_back(temp);
	}
	/////////////////////////////////////////////////////////
}

int main(int argc, char* argv[])
{
	QCoreApplication a(argc, argv);

	vector<Mat> numberPics;
	Mat numSrc = imread(numFileName, IMREAD_GRAYSCALE);
	threshold(numSrc, numSrc, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);
	findNumber(numSrc, numberPics);


	Mat cardPic = imread(cardFileName);
	int j = (double)300 / cardPic.size().width * cardPic.size().height;
	resize(cardPic, cardPic, { 300, j });
	Mat gray, temp, dst, sobel, thres, tophat;

	cvtColor(cardPic, gray, COLOR_BGR2GRAY);

	morphologyEx(gray, tophat, MORPH_TOPHAT, getStructuringElement(MORPH_RECT, { 9,3 }));

	Sobel(tophat, sobel, -1, 1, 0);
	morphologyEx(sobel, temp, MORPH_CLOSE, getStructuringElement(MORPH_RECT, { 13,5 }));
	threshold(temp, temp, 0, 255, THRESH_BINARY | THRESH_OTSU);
	vector<vector<Point>> cons;
	findContours(temp, cons, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	vector<Rect> rects;
	vector<Mat> numMats;
	for (auto& i : cons)
	{
		auto tectTemp = boundingRect(i);
		double d = (double)tectTemp.width / tectTemp.height;
		rectangle(cardPic, tectTemp, { 255,255,0 }, 2);
		if (d > 2.4 && d < 3.8 && tectTemp.width>40 && tectTemp.width < 60)
		{
			rects.push_back(tectTemp);
			rectangle(cardPic, tectTemp, { 0,255,0 }, 2);
		}
	}
	sort(rects.begin(), rects.end(), [](Rect a, Rect b) {return a.x < b.x; });
	vector<Mat> cardNum;
	for (auto& i : rects)
	{
		Mat numTemp = gray({ i.x - 2,i.y - 2,i.width + 4,i.height + 4 });
		threshold(numTemp, numTemp, 0, 255, THRESH_BINARY | THRESH_OTSU);
		findNumber(numTemp, cardNum);

	}


	vector<int> numbers;
	for (const auto& n : cardNum)
	{
		vector<double> scores;
		for (const auto& i : numberPics)
		{
			Mat res;
			double min, max;
			matchTemplate(n, i, res, TM_CCORR_NORMED);
			minMaxLoc(res, &min, &max);
			scores.push_back(max);
		}
		auto ret = std::distance(scores.begin(), std::max_element(scores.begin(), scores.end()));

		if (scores[ret] > 0.78)
		{
			numbers.push_back(ret);

		}
	}
	for (auto& i : numbers)
		cout << i;

	return a.exec();
}
