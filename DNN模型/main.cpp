#include <QtCore/QCoreApplication>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <qdir.h>
#include <iostream>
#include <opencv2/dnn.hpp>
#include <qfile.h>

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

	vector<string>  names1;
	QFile file("../image/synset_words.txt");
	file.open(QIODevice::ReadOnly);
	while (!file.atEnd())
	{
		string temp = file.readLine().toStdString();
		names1.push_back(temp);
	}
	file.close();

	auto net = dnn::readNetFromCaffe("../image/bvlc_googlenet.prototxt", "../image/bvlc_googlenet.caffemodel");
	QDir imgDir("../image/dnnImages");
	auto files = imgDir.entryList(QDir::Files);
	Mat srcImg = imread(("../image/dnnImages/" + files[5]).toStdString());

	resize(srcImg, srcImg, { 224,224 });

	auto blob = dnn::blobFromImage(srcImg, 1, { 224,224 }, { 104,117,123 });
	net.setInput(blob);
	vector<float> ret = net.forward();
	int maxPosition = max_element(ret.begin(), ret.end()) - ret.begin();
	cout << ret[maxPosition] << "            " << names1[maxPosition];

	return a.exec();
}
