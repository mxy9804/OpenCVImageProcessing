#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2\imgproc\types_c.h>
using namespace cv;
using namespace std;

// 直方图拉伸函数
// grayImage - 要拉伸的单通道灰度图像
// hist - grayImage的直方图
// minValue - 忽略像数个数小于此值的灰度级
void histStretch(Mat& grayImage, const Mat& hist, int minValue)
{
    CV_Assert(!grayImage.empty() && grayImage.channels() == 1 && grayImage.depth() == CV_8U);
    CV_Assert(!hist.empty() && hist.rows == 256 && hist.cols == 1 && hist.depth() == CV_32F);
    CV_Assert(minValue >= 0);

    // 求左边界
    uchar grayMin = 0;
    for (int i = 0; i < hist.rows; ++i)
    {
        if (hist.at<float>(i, 0) > minValue)
        {
            grayMin = static_cast<uchar>(i);
            break;
        }
    }

    // 求右边界
    uchar grayMax = 0;
    for (int i = hist.rows - 1; i >= 0; --i)
    {
        if (hist.at<float>(i, 0) > minValue)
        {
            grayMax = static_cast<uchar>(i);
            break;
        }
    }

    if (grayMin >= grayMax)
    {
        return;
    }

    const int w = grayImage.cols;
    const int h = grayImage.rows;
    for (int y = 0; y < h; ++y)
    {
        uchar* imageData = grayImage.ptr<uchar>(y);
        for (int x = 0; x < w; ++x)
        {
            if (imageData[x] < grayMin)
            {
                imageData[x] = 0;
            }
            else if (imageData[x] > grayMax)
            {
                imageData[x] = 255;
            }
            else
            {
                imageData[x] = static_cast<uchar>(round((imageData[x] - grayMin) * 255.0 / (grayMax - grayMin)));
            }
        }
    }
}
