/*
The Ground Model example
*/
#include <unistd.h>
#define GetCurrentDir getcwd
#include<iostream>

#include <iostream>
#include <cmath>
#include <string>
#include <vector>
#include <boost/format.hpp>
#include <typeinfo>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "sophus/se3.h"
#include "sophus/so3.h"
#include <g2o/core/base_binary_edge.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include "initKDT.h"
#include "surroundView.h"

using namespace std; 
// using namespace cv;


std::string GetCurrentWorkingDir( void ) {
  char buff[FILENAME_MAX];
  GetCurrentDir( buff, FILENAME_MAX );
  std::string current_working_dir(buff);
  return current_working_dir;
}

int main(int argc, char** argv)
{
	cout<<"------------------Initialize camera pose----------------"<<endl;
	Sophus::SE3 T_FG;
	Sophus::SE3 T_LG;
	Sophus::SE3 T_BG;
	Sophus::SE3 T_RG;
	initializePose(T_FG, T_LG, T_BG, T_RG);
	cout<<T_FG.matrix()<<endl;

	cout<<"---------------Initialize K-------------------------------"<<endl;
	Eigen::Matrix3d K_F;
	Eigen::Matrix3d K_L;
	Eigen::Matrix3d K_B;
	Eigen::Matrix3d K_R;
	initializeK(K_F, K_L, K_B, K_R);
	cout<<K_F<<endl;
	
	cout<<"--------------------Initialize D--------------------------"<<endl;
	Eigen::Vector4d D_F;
	Eigen::Vector4d D_L;
	Eigen::Vector4d D_B; 
	Eigen::Vector4d D_R;
	initializeD(D_F, D_L, D_B, D_R);
	cout<<D_F<<endl;
	
	cout<<"--------------------Load images--------------------------"<<endl;
	int img_index = 596;

	cout << GetCurrentWorkingDir() << std::endl;
	boost::format img_path_template("../../test_cases/ALL/%06d ");
	cout<<"Reading "<<(img_path_template%img_index).str()<<"F.jpg"<<endl;
	cv::Mat img_F = cv::imread((img_path_template%img_index).str()+ "F.jpg");
	cout<<"Reading "<<(img_path_template%img_index).str()<<"L.jpg"<<endl;
	cv::Mat img_L = cv::imread((img_path_template%img_index).str()+ "L.jpg");
	cout<<"Reading "<<(img_path_template%img_index).str()<<"B.jpg"<<endl;
	cv::Mat img_B = cv::imread((img_path_template%img_index).str()+ "B.jpg");
	cout<<"Reading "<<(img_path_template%img_index).str()<<"R.jpg"<<endl;
	cv::Mat img_R = cv::imread((img_path_template%img_index).str()+ "R.jpg");
	cout<<img_F.size<<endl;

	cout<<"--------------------Init K_G--------------------------"<<endl;
	int rows = 1000;
	int cols = 1000;
	double dX = 0.1;
	double dY = 0.1;
	double fx =  1/dX;
	double fy = -1/dY;
	cv::Mat K_G = cv::Mat::zeros(3,3,CV_64FC1);
	K_G.at<double>(0,0) = fx;
	K_G.at<double>(1,1) = fy;
	K_G.at<double>(0,2) = cols/2;
	K_G.at<double>(1,2) = rows/2;
	K_G.at<double>(2,2) =   1.0;
	cout<<K_G<<endl;
	
	cout<<"--------------------Add noise to T matrix--------------------------"<<endl;
	Eigen::Matrix<double,6,1>  V6;
	V6<<0.01, -0.01, 0.01, -0.01, 0.01, -0.01;
	T_FG = Sophus::SE3::exp(T_FG.log()+V6);
	T_LG = Sophus::SE3::exp(T_LG.log()+V6);
	T_BG = Sophus::SE3::exp(T_BG.log()+V6);
	T_RG = Sophus::SE3::exp(T_RG.log()+V6);
	
	Sophus::SE3 T_GF = T_FG.inverse();
	Sophus::SE3 T_GL = T_LG.inverse();
	Sophus::SE3 T_GB = T_BG.inverse();
	Sophus::SE3 T_GR = T_RG.inverse();
	
	cout<<"--------------------Project images on the ground--------------------------"<<endl;
	cv::Mat img_GF = project_on_ground(img_F,T_FG,K_F,D_F,K_G,rows,cols);
	cv::Mat img_GL = project_on_ground(img_L,T_LG,K_L,D_L,K_G,rows,cols);
	cv::Mat img_GB = project_on_ground(img_B,T_BG,K_B,D_B,K_G,rows,cols);
	cv::Mat img_GR = project_on_ground(img_R,T_RG,K_R,D_R,K_G,rows,cols);
	
	cout<<"--------------------Stitch the surround view image--------------------------"<<endl;
	cv::Mat img_G = generate_surround_view(img_GF,img_GL,img_GB,img_GR,rows,cols);
	
	int overlap_x,overlap_y,overlap_w,overlap_h;
	overlap_x = 650;
	overlap_y = 0;
	overlap_w = 350;
	overlap_h = 350;
	cv::Rect rectFR(overlap_x,overlap_y,overlap_w,overlap_h);
	cv::Rect rectFL(overlap_x,overlap_y,overlap_w,overlap_h);
	
	// update T_FG
	Eigen::Matrix<double,6,1> rhophi_F,rhophi_R;
	double lr = 1e-8;
	int iter_times = 30;
	
	for(int i=0;i<iter_times;i++)
	{
		cout<<"iter "<<i<<": "<<endl;
		
		cv::namedWindow("img_GF",0);
		cv::imshow("img_GF",img_GF(rectFR));
		cv::namedWindow("img_GL",0);
		cv::imshow("img_GL",img_GR(rectFR));
		
		adjust_pose(img_GF,img_GR,rhophi_F,rhophi_R,K_G,
				overlap_x, overlap_y, overlap_w,overlap_h,lr);
		cout<<"rhophi_F: "<<endl<<rhophi_F<<endl;
		cout<<"rhophi_R: "<<endl<<rhophi_R<<endl;
		
		T_GF = Sophus::SE3::exp(rhophi_F).inverse()*T_GF;
		T_FG = T_GF.inverse();
		
		cout<<T_FG<<endl;
		cout<<T_RG<<endl;
		
		img_GF = project_on_ground(img_F,T_FG,K_F,D_F,K_G,rows,cols);
		img_GR = project_on_ground(img_R,T_RG,K_R,D_R,K_G,rows,cols);
		
		cv::waitKey(0);
	}
	
	T_GF = Sophus::SE3::exp(T_GF.log() - rhophi_F);
	T_FG = T_GF.inverse();
	
	// project img_GF with new T_FG
	img_GF = project_on_ground(img_F,T_FG,K_F,D_F,K_G,rows,cols);
	
	// Show new diff between img_GF and img_GL
	cv::Mat img_GF_gray,img_GL_gray;
	cv::cvtColor(img_GF(rectFL),img_GF_gray,cv::COLOR_BGR2GRAY);
	cv::medianBlur(img_GF_gray,img_GF_gray,5);
	img_GF_gray.convertTo(img_GF_gray, CV_64FC1);
	cv::cvtColor(img_GL(rectFL),img_GL_gray,cv::COLOR_BGR2GRAY);
	cv::medianBlur(img_GL_gray,img_GL_gray,5);
	img_GL_gray.convertTo(img_GL_gray, CV_64FC1);
	
	cv::Mat diff_FL,diff_FL_norm,diff_FL_color;
	cv::subtract(img_GF_gray,img_GL_gray,diff_FL,cv::noArray(),CV_64FC1);
	double coef = cv::mean(img_GF_gray).val[0]/cv::mean(img_GL_gray).val[0] ;
	cv::subtract(img_GF_gray,coef*img_GL_gray,diff_FL,cv::noArray(),CV_64FC1);
	cv::normalize(diff_FL,diff_FL_norm,0,255,cv::NORM_MINMAX,CV_8U);
	cv::applyColorMap(diff_FL_norm,diff_FL_color, cv::COLORMAP_JET);
	cv::namedWindow("diff_FL2",0);
	cv::imshow("diff_FL2",diff_FL_color);
	
	img_GF_gray.convertTo(img_GF_gray, CV_8U);
	cv::namedWindow("img_GF2",0);
	cv::imshow("img_GF2",img_GF_gray);
	
	cv::waitKey(0);
	cv::destroyAllWindows();

	return 0;
}