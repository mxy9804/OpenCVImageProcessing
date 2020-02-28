
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2\imgproc\types_c.h>
using namespace cv;
using namespace std;


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

//测试通过样例：（注：图片路径需要更换成本地图片路径）
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
