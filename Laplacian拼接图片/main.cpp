
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



class LaplacianBlending {
private:
    Mat leftImg;
    Mat rightImg;
    Mat blendMask;

    //Laplacian Pyramids
    vector<Mat> leftLapPyr, rightLapPyr, resultLapPyr;
    Mat leftHighestLevel, rightHighestLevel, resultHighestLevel;
    //maskΪ��ͨ������������
    vector<Mat> maskGaussianPyramid;

    int levels;


    void buildGaussianPyramid()
    {
        //����������Ϊÿһ�����ģ
        assert(leftLapPyr.size() > 0);

        maskGaussianPyramid.clear();
        Mat currentImg;
        cvtColor(blendMask, currentImg, COLOR_GRAY2BGR);
        //����mask��������ÿһ��ͼ��
        maskGaussianPyramid.push_back(currentImg); //0-level

        currentImg = blendMask;
        for (int l = 1; l < levels + 1; l++) {
            Mat _down;
            if (leftLapPyr.size() > l)
                pyrDown(currentImg, _down, leftLapPyr[l].size());
            else
                pyrDown(currentImg, _down, leftHighestLevel.size()); //lowest level

            Mat down;
            cvtColor(_down, down, COLOR_GRAY2BGR);
            //add color blend mask into mask Pyramid
            maskGaussianPyramid.push_back(down);
            currentImg = _down;
        }
    }

    void buildLaplacianPyramid(const Mat& img, vector<Mat>& lapPyr, Mat& HighestLevel)
    {
        lapPyr.clear();
        Mat currentImg = img;
        for (int l = 0; l < levels; l++) {
            Mat down, up, temp;
            pyrDown(currentImg, down);
            pyrUp(down, up, currentImg.size());
            Mat lap = currentImg - up;
            lapPyr.push_back(lap);
            currentImg = down;
        }
        currentImg.copyTo(HighestLevel);
    }


    void blendLapPyrs()
    {
        //���ÿ���������ֱ����������ͼLaplacian�任ƴ�ɵ�ͼ��resultLapPyr
        resultHighestLevel = leftHighestLevel.mul(maskGaussianPyramid.back()) + rightHighestLevel.mul(Scalar(1.0, 1.0, 1.0) - maskGaussianPyramid.back());
        for (int l = 0; l < levels; l++)
        {
            Mat A = leftLapPyr[l].mul(maskGaussianPyramid[l]);
            Mat antiMask = Scalar(1.0, 1.0, 1.0) - maskGaussianPyramid[l];
            Mat B = rightLapPyr[l].mul(antiMask);
            Mat blendedLevel = A + B;

            resultLapPyr.push_back(blendedLevel);
        }
    }

public:
    LaplacianBlending(const Mat& _left, const Mat& _right, const Mat& _blendMask, int _levels) ://construct function, used in LaplacianBlending lb(l,r,m,4);
        leftImg(_left), rightImg(_right), blendMask(_blendMask), levels(_levels)
    {
        assert(_left.size() == _right.size());
        assert(_left.size() == _blendMask.size());
        //����������˹�������͸�˹������
        buildLaplacianPyramid(leftImg, leftLapPyr, leftHighestLevel);
        buildLaplacianPyramid(rightImg, rightLapPyr, rightHighestLevel);
        buildGaussianPyramid();
        //ÿ�������ͼ��ϲ�Ϊһ��
        blendLapPyrs();
    };

    Mat blend()
    {
        //�ؽ�������˹������
         //������laplacianͼ��ƴ�ɵ�resultLapPyr��������ÿһ��
        //���ϵ��²�ֵ�Ŵ���в���ӣ�����blendͼ����
        Mat currentImg = resultHighestLevel;
        for (int l = levels - 1; l >= 0; l--)
        {
            Mat test;
            Mat up;
            pyrUp(currentImg, up, resultLapPyr[l].size());
            currentImg = up + resultLapPyr[l];

            currentImg.convertTo(test, CV_8UC3);
        }
        return currentImg;
    }
};



int main() {
    Mat leftImg = imread("../image/apple.png");
    Mat rightImg = imread("../image/orange.png");

    int hight = leftImg.rows - 1;
    int width = leftImg.cols;
    leftImg = leftImg(Range(0, hight), Range::all());

    Mat leftImg32f, rightImg32f, temp;
    leftImg.convertTo(leftImg32f, CV_32F);
    rightImg.convertTo(rightImg32f, CV_32F);

    //�������ڻ�ϵ���Ĥ���������м���л��
    Mat mask = Mat::zeros(hight, width, CV_32FC1);
    mask(Range::all(), Range(0, mask.cols * 0.5)) = 1.0;


    LaplacianBlending laplaceBlend(leftImg32f, rightImg32f, mask, 5);
    Mat blendImg = laplaceBlend.blend();

    blendImg.convertTo(blendImg, CV_8UC3);

    imshow("left", leftImg);
    imshow("right", rightImg);
    imshow("blended", blendImg);

    waitKey(0);
    return 0;
}