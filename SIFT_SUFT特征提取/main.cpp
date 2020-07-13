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



const string testFile = "../image/left_01.png";
const string sceneFile = "../image/right_01.png";


int main(int argc, char* argv[])
{
	QCoreApplication a(argc, argv);
	Mat box = imread(testFile);
	Mat scene = imread(sceneFile);
	//提取特征点方法
	//SIFT
	auto sift = cv::xfeatures2d::SIFT::create();

	//特征点
	std::vector<cv::KeyPoint> keyBox, keyScene;
	//提取特征点   由于下步detectAndCompute提取特征点并计算特征描述子，故注释了
	//sift->detect(box, keyBox);
	//sift->detect(scene, keyScene);

	//画特征点
	//cv::Mat keyBoxImg,keySceneImg;
	//drawKeypoints(box, keyBox, keyBoxImg);
	//drawKeypoints(scene, keyScene, keySceneImg);

	//特征点匹配
	cv::Mat despBox, despScene;
	//提取特征点并计算特征描述子
	sift->detectAndCompute(box, cv::Mat(), keyBox, despBox);
	sift->detectAndCompute(scene, cv::Mat(), keyScene, despScene);


	//结构 ： DMatch: 属性
	//int queryIdx –>是样本图像的特征点描述符（descriptor）的下标，同时也是描述符对应特征点（keypoint)的下标。原图，拿来做配对的图
	//int trainIdx –> 是测试图像的特征点描述符的下标，同样也是相应的特征点的下标。一般包含了原图，是比原图场景大，来找到原图的大场景图
	//int imgIdx –>当样本是多张图像的话有用。
	//float distance –>代表这一对匹配的特征点描述符（本质是向量）的欧氏距离，数值越小也就说明两个特征点越相像。
	std::vector<cv::DMatch> matches, bfKnn, bf;

	//如果采用flannBased方法 那么 desp通过orb的到的类型不同需要先转换类型///////////这个速度快////
	if (despBox.type() != CV_32F || despScene.type() != CV_32F)
	{
		despBox.convertTo(despBox, CV_32F);
		despScene.convertTo(despScene, CV_32F);
	}
	auto  matcher = cv::DescriptorMatcher::create("FlannBased");
	matcher->match(despBox, despScene, matches);
	cv::Mat imageOutput, bfOutput, bfKnnOutput;

	/*
	//计算特征点距离的最大值
	double maxDist = 0;
	for (int i = 0; i < despL.rows; i++)
	{
		double dist = matches[i].distance;
		if (dist > maxDist)
			maxDist = dist;
	}

	//挑选好的匹配点
	std::vector< cv::DMatch > good_matches;
	for (int i = 0; i < despL.rows; i++)
	{
		if (matches[i].distance < 0.5 * maxDist)
		{
			good_matches.push_back(matches[i]);
		}
	}
	*/
	//以distance从小到大排序
	sort(matches.begin(), matches.end(), [](cv::DMatch ii, cv::DMatch jj) { return ii.distance < jj.distance; });
	//画前30个匹配点
	cv::drawMatches(box, keyBox, scene, keyScene, vector<cv::DMatch>(matches.begin(), matches.begin() + 30), imageOutput);

	imshow("imageOutput", imageOutput);

	//BFMatcher方法//////////////////////////////////////////////////////////////////////////////////////
	auto bfMatcher = BFMatcher::create(4, true);
	auto bfMach = BFMatcher::create();
	//////用knnMatch方法随机抽样一致法得到配对点
	vector<vector<DMatch>> knn_matches;
	bfMach->knnMatch(despBox, despScene, knn_matches, 10);
	for (size_t r = 0; r < knn_matches.size(); ++r)
	{
		if (knn_matches[r][0].distance > 0.5 * knn_matches[r][1].distance) continue;
		bfKnn.push_back(knn_matches[r][0]);
	}
	cv::drawMatches(box, keyBox, scene, keyScene, bfKnn, bfKnnOutput);
	cout << bfKnn.size();
	imshow("bfKnnOutput", bfKnnOutput);
	////////用match得到配对点////////////////////////////////////////
	bfMatcher->match(despBox, despScene, bf);
	//以distance从小到大排序
	sort(bf.begin(), bf.end(), [](cv::DMatch ii, cv::DMatch jj) { return ii.distance < jj.distance; });
	//画前10个匹配点
	cv::drawMatches(box, keyBox, scene, keyScene, vector<cv::DMatch>(bf.begin(), bf.begin() + 10), bfOutput);
	imshow("bfOutput", bfOutput);
	return a.exec();
}
