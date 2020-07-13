#include <QtCore/QCoreApplication>
#include <iostream>
#include<opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <qstring.h>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/calib3d.hpp>


using namespace cv;
using namespace std;

#ifdef _DEBUG
#pragma comment (lib,"opencv_world430d")
#else
#pragma comment (lib,"opencv_world430")
#endif 



const string leftFile = "../image/left_01.png";
const string rightFile = "../image/right_01.png";


int main(int argc, char* argv[])
{
	QCoreApplication a(argc, argv);



	auto leftImg = imread(leftFile);
	auto rightImg = imread(rightFile);



	//采用SIFT方法找特征点
	auto sift = xfeatures2d::SIFT::create();
	vector<KeyPoint> keyPointLeft, keyPointRight;
	Mat decpLeft, decpRight, outputImg, tempImg;

	sift->detectAndCompute(leftImg, {}, keyPointLeft, decpLeft);
	sift->detectAndCompute(rightImg, {}, keyPointRight, decpRight);

	//采用BF配对，用knnMatch方法配对
	auto bf = BFMatcher::create();
	vector<vector<DMatch>> knnMatchs;
	vector<DMatch> matchs;
	bf->knnMatch(decpLeft, decpRight, knnMatchs, 2);

	//找到好的配对特征点，系数0.4-0.7比较好
	for (auto& i : knnMatchs)
	{
		if (i[0].distance > 0.4 * i[1].distance) continue;
		matchs.push_back(i[0]);
	}
	if (matchs.size() < 4)
		return -1;

	vector< Point2f> leftPoint, rightPoint;
	for (int i = 0; i < matchs.size(); i++)
	{

		leftPoint.push_back(keyPointLeft[matchs[i].queryIdx].pt);
		rightPoint.push_back(keyPointRight[matchs[i].trainIdx].pt);
	}

	//得到变换矩阵。注意两个点集合rightPoint, leftPoint的先后顺序，一定要注意，错了warpPerspective会出问题。或都warpPerspective的flag参数选16反射才可以。
	auto H = findHomography(rightPoint, leftPoint, RANSAC, 4);

	//得到图片的宽度，因为findHomography得到H矩阵是3X3的所以点增加一个为1的配对项，向量从2列变成3列，好用来和矩阵相乘
	double dSrc[]{ rightImg.cols, 0, 1 };//右上角   这是拿右上角，也可拿右下脚，取最大值
	double dDes[3];//变换后的坐标值
	Mat vSrc{ 3, 1, CV_64FC1, dSrc };  //列向量
	Mat vDes{ 3, 1, CV_64FC1,dDes };  //列向量    
	vDes = H * vSrc;
	int nWidth = dDes[0] / dDes[2];

	//画特征点配对好的图
	drawMatches(leftImg, keyPointLeft, rightImg, keyPointRight, matchs, outputImg);

	//变换右边的图片。findHomography参数搞反了，要用以下这个，注意参数flag为16
	//warpPerspective(rightImg, tempImg, H, cv::Size(leftImg.size().width+ rightImg.size().width,leftImg.size().height),16); 	
	warpPerspective(rightImg, tempImg, H, cv::Size(nWidth, leftImg.rows));

	//把左边图copy到tempImg的ROI区域
	leftImg.copyTo(tempImg(Rect(0, 0, leftImg.cols, leftImg.rows)));

	imshow("tempImg", tempImg);
	imshow("outputImg", outputImg);

	return a.exec();
}
