
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
    //����ϰ����5�����𰸷ֱ�Ϊa,b,c,d,e��
    int v[]{ 0,2,2,3,4 };
    //ʶ���Ľ�� ��Ϊ1���Ƕԣ�Ϊ0����
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

#pragma region //�ҵ�����ͼ����ĸ�����//
    findContours(gaussImg, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    //drawContours(srcImg, contours, -1, { 0,0,255 });
    //�õ��������ƶ����  ,�粻���ı��ξ��ǲ���
    approxPolyDP(contours[0], docCnt, 0.02 * arcLength(contours[0], true), true);
    //�粻���ı��ξ��ǲ���
    assert(docCnt.size() == 4);
#pragma endregion

#pragma region //ͼ����б���//
    //�򵥶������α任��Ŀ��
    int height = docCnt[1].y - docCnt[0].y;
    int  width = docCnt[2].x - docCnt[1].x;
    //������κ���ε��ĸ�����
    vector<Point2f> trnCnt{ Point2f(0,0),Point2f(0,height),Point2f(width,height),Point2f(width,0) };
    auto H = getPerspectiveTransform(docCnt, trnCnt);
    warpPerspective(grayImg, grayImg, H, { width,height });
#pragma endregion

#pragma region //ͨ�������ҵ����⿨ÿ�����Բ�Σ����ź���Ž�titles��//
    contours.clear();
    threshold(grayImg, grayImg, 0, 255, THRESH_OTSU | THRESH_BINARY_INV);
    findContours(grayImg, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    vector<Rect>  rects;

    for (auto& i : contours)
    {
        Rect2f rect = boundingRect(i);
        //����Բ��������Ӿ��γ���Ƚӽ�1�����ͼ�е������Ӿ��εĿ�ȡ�30
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
            //����Ӿ���ROI����0ֵ����������>600����Ϳд���Ĵ��⿨
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
