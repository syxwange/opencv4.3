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



	//����SIFT������������
	auto sift = xfeatures2d::SIFT::create();
	vector<KeyPoint> keyPointLeft, keyPointRight;
	Mat decpLeft, decpRight, outputImg, tempImg;

	sift->detectAndCompute(leftImg, {}, keyPointLeft, decpLeft);
	sift->detectAndCompute(rightImg, {}, keyPointRight, decpRight);

	//����BF��ԣ���knnMatch�������
	auto bf = BFMatcher::create();
	vector<vector<DMatch>> knnMatchs;
	vector<DMatch> matchs;
	bf->knnMatch(decpLeft, decpRight, knnMatchs, 2);

	//�ҵ��õ���������㣬ϵ��0.4-0.7�ȽϺ�
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

	//�õ��任����ע�������㼯��rightPoint, leftPoint���Ⱥ�˳��һ��Ҫע�⣬����warpPerspective������⡣��warpPerspective��flag����ѡ16����ſ��ԡ�
	auto H = findHomography(rightPoint, leftPoint, RANSAC, 4);

	//�õ�ͼƬ�Ŀ�ȣ���ΪfindHomography�õ�H������3X3�����Ե�����һ��Ϊ1������������2�б��3�У��������;������
	double dSrc[]{ rightImg.cols, 0, 1 };//���Ͻ�   ���������Ͻǣ�Ҳ�������½ţ�ȡ���ֵ
	double dDes[3];//�任�������ֵ
	Mat vSrc{ 3, 1, CV_64FC1, dSrc };  //������
	Mat vDes{ 3, 1, CV_64FC1,dDes };  //������    
	vDes = H * vSrc;
	int nWidth = dDes[0] / dDes[2];

	//����������Ժõ�ͼ
	drawMatches(leftImg, keyPointLeft, rightImg, keyPointRight, matchs, outputImg);

	//�任�ұߵ�ͼƬ��findHomography�����㷴�ˣ�Ҫ�����������ע�����flagΪ16
	//warpPerspective(rightImg, tempImg, H, cv::Size(leftImg.size().width+ rightImg.size().width,leftImg.size().height),16); 	
	warpPerspective(rightImg, tempImg, H, cv::Size(nWidth, leftImg.rows));

	//�����ͼcopy��tempImg��ROI����
	leftImg.copyTo(tempImg(Rect(0, 0, leftImg.cols, leftImg.rows)));

	imshow("tempImg", tempImg);
	imshow("outputImg", outputImg);

	return a.exec();
}
