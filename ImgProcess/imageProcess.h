#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2\imgproc\types_c.h>

typedef unsigned char BYTE;
using namespace cv;
using namespace std;

typedef struct {
    int	width;
    int	height;
    int bits_per_pixel;//默认为32
    int bits_per_channel;//默认为8
    int channel_count;//默认为4
    BYTE pixel[];
} Bitmap;


/*
    使用说明：
    Bitmap格式转Mat函数
    BitmapConverseMat()
    参数1：const Bitmap bitImg是Bitmap结构体；
    参数2：Mat matImg是转换成的Mat对象；
*/
void BitmapConverseMat(const Bitmap bitImg,Mat matImg){
    
    //根据Bitmap对象的宽和高创建一个matImgRGB三通道对象：
    matImg = Mat::zeros(bitImg.height, bitImg.width, CV_8UC3);
    
    int rows = bitImg.height;
    int cols = bitImg.width;
    int index=0;//将bitImg像素下标置为0；
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
           
            matImg.at<Vec3b>(row, col)[2] = bitImg.pixel[index];    //Mat对象：B、G、R<——>Bitmap对象：R、G、B、A
            index++;
            matImg.at<Vec3b>(row, col)[1] = bitImg.pixel[index];
            index++;
            matImg.at<Vec3b>(row, col)[0] = bitImg.pixel[index];
            index += 2;//自动跳过A通道赋值。
             
        }
    }


}

/*
    使用说明：
    Mat格式转Bitmap函数
    MatConverseBitmap()
    参数1：const Mat matImg是Mat类对象；
    参数2：Bitmap bitImg是转换成的Bitmap结构体；
*/
void MatConverseBitmap(const Mat matImg, Bitmap bitImg) {
    bitImg.width = matImg.cols;
    bitImg.height = matImg.rows;
    int index = 0;
    int rows = bitImg.height;
    int cols = bitImg.width;
    //将bitImg像素下标置为0；
    int index = 0;
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {

            bitImg.pixel[index] = matImg.at<Vec3b>(row, col)[2];   //Mat对象：B、G、R<——>Bitmap对象：R、G、B、A
            index++;
            bitImg.pixel[index] = matImg.at<Vec3b>(row, col)[1];  
            index++;
            bitImg.pixel[index] = matImg.at<Vec3b>(row, col)[0];
            index += 2;//自动跳过A通道赋值。
        }
    }
   
}


// 使用Rect绘制直方图
void drawHist_Rect(const Mat& hist, Mat& canvas, const Scalar& color)
{
    CV_Assert(!hist.empty() && hist.cols == 1);
    CV_Assert(hist.depth() == CV_32F && hist.channels() == 1);
    CV_Assert(!canvas.empty() && canvas.cols >= hist.rows);

    const int width = canvas.cols;
    const int height = canvas.rows;

    // 获取最大值
    double dMax = 0.0;
    minMaxLoc(hist, nullptr, &dMax);

    // 计算直线的宽度
    float thickness = float(width) / float(hist.rows);

    // 绘制直方图
    for (int i = 1; i < hist.rows; ++i)
    {
        double h = hist.at<float>(i, 0) / dMax * 0.9 * height; // 最高显示为画布的90%
        cv::rectangle(canvas,
            cv::Point(static_cast<int>((i - 1) * thickness), height),
            cv::Point(static_cast<int>(i * thickness), static_cast<int>(height - h)),
            color,
            static_cast<int>(thickness));
    }
}

// 直方图拉伸
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






//对单通道图像进行均衡化：
void equalize(const Mat& src,Mat dst ) {
    CV_Assert(!src.empty()&&src.channels() == 1);
    equalizeHist(src, dst);
}

//可以分离R、G、B三个通道图像的函数,并二极化输出：
/*
使用说明：
第一个参数src    是要进行RGB分离的图像；
第二个参数dst_r  是分离出的R通道图像；
第三个参数dst_g  是分离出的G通道图像；
第四个参数dst_b  是分离出的B通道图像；
第五个参数r_thres是对R通道图像进行二极化后的图像；
第六个参数g_thres是对G通道图像进行二极化后的图像；
第七个参数b_thres是对B通道图像进行二极化后的图像；
第八个参数type   是对图像进行二极化的类型选项，默认为0，一般二极化；
*/
void split(const Mat& src, Mat dst_r, Mat dst_g, Mat dst_b, Mat r_thres, Mat g_thres, Mat b_thres, int type = 0) {
    
    CV_Assert(!src.empty() && src.channels() == 3);
        
        //自定义阈值：
        const int thres = 127;
    	
    	
    	//设置阈值类型选项(默认为阈值二极化）
    	int number = 0;//0,1,2,3,4,5,6,7
    	int type = THRESH_BINARY;
    	switch (number) {
    	case 0:type = THRESH_BINARY; break;         //阈值二极化(大于阈值的部分被置为255，小于部分被置为0)；
    	case 1:type = THRESH_BINARY_INV; break;     //阈值反二极化(大于阈值部分被置为0，小于部分被置为255)；
    	case 2:type = THRESH_TRUNC; break;          //大于阈值部分被置为threshold，小于部分保持原样;
    	case 3:type = THRESH_TOZERO; break;         //小于阈值部分被置为0，大于部分保持不变;
    	case 4:type = THRESH_TOZERO_INV; break;     //大于阈值部分被置为0，小于部分保持不变 ;
    	case 5:type = THRESH_MASK; break;
    	case 6:type = THRESH_OTSU; break;           //系统根据图像自动选择最优阈值，并进行二极化；
    	case 7:type = THRESH_TRIANGLE; break;       //系统根据图像自动选择最优阈值，并进行二极化；
    
    	}
        Mat dst;
    	cvtColor(src, dst, CV_BGR2GRAY);
    	dst_b.create(dst.size(), dst.type());
    	dst_g.create(dst.size(), dst.type());
    	dst_r.create(dst.size(), dst.type());
    	int rows = src.rows;
    	int cols = src.cols;
    	int channel = src.channels();
    	for (int row = 0; row < rows; row++) {
    		for (int col = 0; col < cols; col++) {
    			
    				dst_b.at<uchar>(row,col) = src.at<Vec3b>(row, col)[0];
    				dst_g.at<uchar>(row,col) = src.at<Vec3b>(row, col)[1];
    				dst_r.at<uchar>(row,col) = src.at<Vec3b>(row, col)[2];
    				
    				
    		}
    	}
    	
    	Mat r_thres, g_thres, b_thres;
    	threshold(dst_b,b_thres, thres, 255, type);
    	threshold(dst_g,g_thres, thres, 255, type);
    	threshold(dst_r,r_thres, thres, 255, type);
    
    	
}



int main(int argc, char** argv) {
    int channels[1] = { 0 };
    int histSize = 256;
    float range[2] = { 0, 256 };
    const float* ranges[1] = { range };

	Mat src, dst_gray,dst;
	src = imread("C:\\Users\\40962\\Pictures\\img.jpg");
	if (src.empty()) {
		printf("can not load image...");
		return -1;
	}
	imshow("原图：", src);
	cvtColor(src, dst_gray, CV_BGR2GRAY);
	imshow("灰度图：", dst_gray);
	//输入图像：必须是8bit单通道图像！
	equalizeHist(dst_gray, dst);
	imshow("直方图均衡衡后：", dst);

    //绘制灰度图直方图：
    Mat grayHist;
    Mat histCanvas_gray(400, 512, CV_8UC3, Scalar::all(255));
    calcHist(&dst_gray, 1, channels, Mat(), grayHist, 1, &histSize, ranges);
    drawHist_Rect(grayHist, histCanvas_gray, Scalar(255, 0, 0));
    imshow("原灰度图直方图：", histCanvas_gray);
   

    Mat dstHist;
    Mat histCanvas_dst(400, 512, CV_8UC3, Scalar::all(255));
    calcHist(&dst, 1, channels, Mat(), dstHist, 1, &histSize, ranges);
    drawHist_Rect(dstHist, histCanvas_dst, Scalar(255, 0, 0));
    imshow("均衡操作后图片的直方图：", histCanvas_dst);

	waitKey(0);
	return 0;
}

//#include <opencv2/opencv.hpp>
//#include <iostream>
//#include <math.h>
//#include <opencv2\imgproc\types_c.h>
//
//using namespace cv;
//int main(int argc, char** argv) {
//	Mat src, dst;
//	src = imread("C:\\Users\\40962\\Pictures\\img.jpg");
//	if (!src.data) {
//		printf("could not load image...\n");
//		return -1;
//	}
//
//	cvtColor(src, src, CV_BGR2GRAY);
//	equalizeHist(src, dst);
//	
//
//	imshow("原图", src);
//	imshow("均衡化后图片：", dst);
//
//	waitKey(0);
//	return 0;
//}

/*
	设计一个能单独输出R、G、B三个通道图像的函数,并二极化输出：
*/
//#include<opencv2/opencv.hpp>
//#include<iostream>
//#include <opencv2\imgproc\types_c.h>
//
//using namespace cv;
//using namespace std;
//int main(int argc, char** argv) {
//	Mat src, dst_b, dst_g, dst_r, dst;
//	src = imread("C:\\Users\\40962\\Pictures\\高中军训.jpg");
//	if (src.empty()) {
//		printf("can not load image...");
//		return -1;
//	}
//
//	//自定义阈值：
//	const int thres = 127;
//	
//	
//	//设置阈值类型选项(默认为阈值二极化）
//	int number = 0;//0,1,2,3,45,6,7
//	int type = THRESH_BINARY;
//	switch (number) {
//	case 0:type = cv::THRESH_BINARY; break;
//	case 1:type = cv::THRESH_BINARY_INV; break;
//	case 2:type = cv::THRESH_TRUNC; break;
//	case 3:type = cv::THRESH_TOZERO; break;
//	case 4:type = cv::THRESH_TOZERO_INV; break;
//	case 5:type = cv::THRESH_MASK; break;
//	case 6:type = cv::THRESH_OTSU; break;
//	case 7:type = cv::THRESH_TRIANGLE; break;
//
//	}
//
//	imshow("原图：", src);
//	cvtColor(src, dst, CV_BGR2GRAY);
//	imshow("灰度图：", dst);
//	dst_b.create(dst.size(), dst.type());
//	dst_g.create(dst.size(), dst.type());
//	dst_r.create(dst.size(), dst.type());
//	//输出测试：
//	/*
//	printf("灰度图通道数：%d\n",dst.channels());
//	printf("灰度图宽度：%d\n", dst.cols);
//	printf("灰度图高度：%d\n", dst.rows);
//
//	printf("分离后B通道图通道数：%d\n", dst_b.channels());
//	printf("分离后B通道图宽度：%d\n", dst_b.cols);
//	printf("分离后B通道图高度：%d\n", dst_b.rows);
//	*/
//	int rows = src.rows;
//	int cols = src.cols;
//	int channel = src.channels();
//	for (int row = 0; row < rows; row++) {
//		for (int col = 0; col < cols; col++) {
//			
//				dst_b.at<uchar>(row,col) = src.at<Vec3b>(row, col)[0];
//				dst_g.at<uchar>(row,col) = src.at<Vec3b>(row, col)[1];
//				dst_r.at<uchar>(row,col) = src.at<Vec3b>(row, col)[2];
//				
//				//输出打印测试：
//				/*
//				printf("B:%d,G:%d,R:%d\n",src.at<Vec3b>(row,col)[0], src.at<Vec3b>(row, col)[1], src.at<Vec3b>(row, col)[2]);
//				printf("B：%d\n",dst_b.at<uchar>(row,col));
//				printf("G：%d\n",dst_g.at<uchar>(row,col));
//				printf("R：%d\n",dst_r.at<uchar>(row,col));
//				*/
//		}
//	}
//	imshow("R通道：", dst_r);
//	imshow("G通道：", dst_g);
//	imshow("B通道：", dst_b);
//	Mat r_thres, g_thres, b_thres;
//	threshold(dst_b,b_thres, thres, 255, type);
//	threshold(dst_g,g_thres, thres, 255, type);
//	threshold(dst_r,r_thres, thres, 255, type);
//
//	imshow("R通道二极化：", r_thres);
//	imshow("G通道二极化：", g_thres);
//	imshow("B通道二极化：", b_thres);
//	waitKey(0);
//	return 0;
//}






// 灰度直方图拉伸

//int main()
//{
//    // 读入图像，此时是3通道的RGB图像
//    Mat image = imread("C:\\Users\\40962\\Pictures\\MambaForever.jpg");
//    if (image.empty())
//    {
//        return -1;
//    }
//
//    // 转换为单通道的灰度图
//    Mat grayImage;
//    cvtColor(image, grayImage, COLOR_BGR2GRAY);
//
//    // 计算直方图并绘制
//    Mat hist;
//    Mat histCanvas(400, 512, CV_8UC3, Scalar(255, 255, 255));
//    int channels[1] = { 0 };
//    int histSize = 256;
//    float range[2] = { 0, 256 };
//    const float* ranges[1] = { range };
//    calcHist(&grayImage, 1, channels, Mat(), hist, 1, &histSize, ranges);
//    drawHist_Rect(hist, histCanvas, cv::Scalar(255, 0, 0));
//
//    // 显示原始灰度图像及其直方图
//    imshow("灰度图像：", grayImage);
//    imshow("灰度图的直方图", histCanvas);
//
//
//
//    // 直方图拉伸
//    Mat grayImageStretched = grayImage.clone();
//    histStretch(grayImageStretched, hist, 20);
//
//    // 计算直方图并绘制
//    Mat histStretched;
//    Mat histCanvasStretched(400, 512, CV_8UC3, Scalar(255, 255, 255));
//    calcHist(&grayImageStretched, 1, channels, Mat(), histStretched, 1, &histSize, ranges);
//    drawHist_Rect(histStretched, histCanvasStretched, Scalar(255, 0, 0));
//
//    // 显示拉伸后的灰度图像及其直方图
//    imshow("拉伸后的灰度图：", grayImageStretched);
//    imshow("拉伸后灰度图的直方图：", histCanvasStretched);
//
//    //绘制对灰度图进行矩阵的掩膜操作后的图片：
//    Mat dst;
//    Mat Kernel = (Mat_<char>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
//    filter2D(grayImage, dst, grayImage.depth(), Kernel);
//    
//    //绘制掩膜操作后图片的直方图：
//    Mat dstHist;
//    Mat histCanvas_dst(400, 512, CV_8UC3, Scalar::all(255));
//    calcHist(&dst, 1, channels, Mat(), dstHist, 1, &histSize, ranges);
//    drawHist_Rect(dstHist, histCanvas_dst, Scalar(255,0,0));
//    imshow("掩膜操作后的图片：", dst);
//    imshow("掩膜操作后图片的直方图：", histCanvas_dst);
//
//    imwrite("C:\\Users\\40962\\Pictures\\原图.jpg", grayImage);
//    imwrite("C:\\Users\\40962\\Pictures\\直方图拉伸.jpg", grayImageStretched);
//    imwrite("C:\\Users\\40962\\Pictures\\掩膜操作.jpg", dst);
//
//    cv::waitKey(0);
//    return 0;
//}
