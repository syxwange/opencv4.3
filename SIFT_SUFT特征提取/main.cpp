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
	//��ȡ�����㷽��
	//SIFT
	auto sift = cv::xfeatures2d::SIFT::create();

	//������
	std::vector<cv::KeyPoint> keyBox, keyScene;
	//��ȡ������   �����²�detectAndCompute��ȡ�����㲢�������������ӣ���ע����
	//sift->detect(box, keyBox);
	//sift->detect(scene, keyScene);

	//��������
	//cv::Mat keyBoxImg,keySceneImg;
	//drawKeypoints(box, keyBox, keyBoxImg);
	//drawKeypoints(scene, keyScene, keySceneImg);

	//������ƥ��
	cv::Mat despBox, despScene;
	//��ȡ�����㲢��������������
	sift->detectAndCompute(box, cv::Mat(), keyBox, despBox);
	sift->detectAndCompute(scene, cv::Mat(), keyScene, despScene);


	//�ṹ �� DMatch: ����
	//int queryIdx �C>������ͼ�����������������descriptor�����±꣬ͬʱҲ����������Ӧ�����㣨keypoint)���±ꡣԭͼ����������Ե�ͼ
	//int trainIdx �C> �ǲ���ͼ������������������±꣬ͬ��Ҳ����Ӧ����������±ꡣһ�������ԭͼ���Ǳ�ԭͼ���������ҵ�ԭͼ�Ĵ󳡾�ͼ
	//int imgIdx �C>�������Ƕ���ͼ��Ļ����á�
	//float distance �C>������һ��ƥ�������������������������������ŷ�Ͼ��룬��ֵԽСҲ��˵������������Խ����
	std::vector<cv::DMatch> matches, bfKnn, bf;

	//�������flannBased���� ��ô despͨ��orb�ĵ������Ͳ�ͬ��Ҫ��ת������///////////����ٶȿ�////
	if (despBox.type() != CV_32F || despScene.type() != CV_32F)
	{
		despBox.convertTo(despBox, CV_32F);
		despScene.convertTo(despScene, CV_32F);
	}
	auto  matcher = cv::DescriptorMatcher::create("FlannBased");
	matcher->match(despBox, despScene, matches);
	cv::Mat imageOutput, bfOutput, bfKnnOutput;

	/*
	//�����������������ֵ
	double maxDist = 0;
	for (int i = 0; i < despL.rows; i++)
	{
		double dist = matches[i].distance;
		if (dist > maxDist)
			maxDist = dist;
	}

	//��ѡ�õ�ƥ���
	std::vector< cv::DMatch > good_matches;
	for (int i = 0; i < despL.rows; i++)
	{
		if (matches[i].distance < 0.5 * maxDist)
		{
			good_matches.push_back(matches[i]);
		}
	}
	*/
	//��distance��С��������
	sort(matches.begin(), matches.end(), [](cv::DMatch ii, cv::DMatch jj) { return ii.distance < jj.distance; });
	//��ǰ30��ƥ���
	cv::drawMatches(box, keyBox, scene, keyScene, vector<cv::DMatch>(matches.begin(), matches.begin() + 30), imageOutput);

	imshow("imageOutput", imageOutput);

	//BFMatcher����//////////////////////////////////////////////////////////////////////////////////////
	auto bfMatcher = BFMatcher::create(4, true);
	auto bfMach = BFMatcher::create();
	//////��knnMatch�����������һ�·��õ���Ե�
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
	////////��match�õ���Ե�////////////////////////////////////////
	bfMatcher->match(despBox, despScene, bf);
	//��distance��С��������
	sort(bf.begin(), bf.end(), [](cv::DMatch ii, cv::DMatch jj) { return ii.distance < jj.distance; });
	//��ǰ10��ƥ���
	cv::drawMatches(box, keyBox, scene, keyScene, vector<cv::DMatch>(bf.begin(), bf.begin() + 10), bfOutput);
	imshow("bfOutput", bfOutput);
	return a.exec();
}
