
#include <QtCore/QCoreApplication>
#include <iostream>
#include<opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;

#ifdef _DEBUG
#pragma comment (lib,"opencv_world430d")
#else
#pragma comment (lib,"opencv_world430")
#endif 

const string srcFile = "../image/test_01.png";
const string rightFile = "../image/test_04.png";


int main(int argc, char* argv[])
{
    //假如习题有5道，答案分别为a,b,c,d,e，
    int v[]{ 0,2,2,3,4 };
    //识别后的结果 ，为1就是对，为0错误
    int ans[5]{};
    QCoreApplication a(argc, argv);
    Mat srcImg = imread(rightFile);

    Mat grayImg, tempImg, desImg, gaussImg;

    srcImg.copyTo(grayImg);
    cvtColor(grayImg, grayImg, COLOR_BGR2GRAY);
    GaussianBlur(grayImg, gaussImg, { 5,5 }, 0);

    Canny(gaussImg, gaussImg, 100, 200);
    vector<vector<Point>> contours;
    vector<Point2f> docCnt;

#pragma region //找到矩形图像的四个顶点//
    findContours(gaussImg, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    //drawContours(srcImg, contours, -1, { 0,0,255 });
    //得到轮廓近似多边形  ,如不是四边形就是不对
    approxPolyDP(contours[0], docCnt, 0.02 * arcLength(contours[0], true), true);
    //如不是四边形就是不对
    assert(docCnt.size() == 4);
#pragma endregion

#pragma region //图像进行变形//
    //简单定义外形变换后的宽高
    int height = docCnt[1].y - docCnt[0].y;
    int  width = docCnt[2].x - docCnt[1].x;
    //定义变形后矩形的四个顶点
    vector<Point2f> trnCnt{ Point2f(0,0),Point2f(0,height),Point2f(width,height),Point2f(width,0) };
    auto H = getPerspectiveTransform(docCnt, trnCnt);
    warpPerspective(grayImg, grayImg, H, { width,height });
#pragma endregion

#pragma region //通过轮廓找到答题卡每个题的圆形，并排好序放进titles中//
    contours.clear();
    threshold(grayImg, grayImg, 0, 255, THRESH_OTSU | THRESH_BINARY_INV);
    findContours(grayImg, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    vector<Rect>  rects;

    for (auto& i : contours)
    {
        Rect2f rect = boundingRect(i);
        //因是圆形所以外接矩形长宽比接近1，这个图中的题的外接矩形的宽度》30
        if (rect.width / rect.height < 1.1 && rect.width / rect.height>0.9 && rect.width > 30)
            rects.push_back(rect);
    }
    srcImg.copyTo(tempImg);
    warpPerspective(tempImg, tempImg, H, { width,height });
    for (auto& i : rects)
        rectangle(tempImg, i, { 0,0,255 });
    drawContours(tempImg, contours, -1, { 255,0,0 }, -1);
    std::sort(rects.begin(), rects.end(),[]( Rect& a,  Rect& b) {return a.y < b.y; });
   
    vector<vector<Rect>> titles;
    for (int i = 0; i < rects.size() / 5; i++)
    {
        vector<Rect> temp{ rects.begin() + i * 5, rects.begin() + (i + 1) * 5 };
        sort(temp.begin(), temp.end(), [](const Rect& a, const Rect& b) {return a.x < b.x; });
        titles.push_back(temp);
    }
#pragma endregion
    for (int i = 0; i < titles.size(); i++)
    {
        bool right = false;
        for (int j = 0; j < titles[i].size(); j++)
        {
            Mat::zeros(titles[i][j].size(), CV_8UC1);
            //以外接矩形ROI。非0值的像素数量>600就是涂写过的答题卡
            if (countNonZero(grayImg(titles[i][j])) > 600 && v[i] == j)
            {
                cout << "NO:" << i + 1 << "     aswer:" << j << "  is right" << endl;
                right = true;
                ans[i] = 1;
            }
        }
        if (!right)
            cout << "NO:" << i + 1 << "     aswer is error" << endl;
    }
    for (auto& i : ans)
        cout << i;
    return a.exec();
}
