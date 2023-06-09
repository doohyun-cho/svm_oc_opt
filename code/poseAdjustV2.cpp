/*
The Ground-Camera Model example
*/

#include <iostream>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <boost/format.hpp>
#include <typeinfo>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "sophus/se3.h"
#include "sophus/so3.h"
#include <g2o/core/base_binary_edge.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include "initKDT.h"
#include "surroundView.h"

using namespace std; 



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
	// test cases' index: 596, 1377, 1401, 1403, 1430
	int img_index = 596;
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
// 	double fx =  1/dX;
// 	double fy = -1/dY;
	cv::Mat K_G = cv::Mat::zeros(3,3,CV_64FC1);
	K_G.at<double>(0,0) = 1/dX;
	K_G.at<double>(1,1) = -1/dY;
	K_G.at<double>(0,2) = cols/2;
	K_G.at<double>(1,2) = rows/2;
	K_G.at<double>(2,2) =   1.0;
	cout<<K_G<<endl;
	
	cout<<"--------------------Add noise to T matrix--------------------------"<<endl;
	Eigen::Matrix<double,6,1>  V6;
	V6<<0.01, -0.01, 0.01, -0.01, 0.01, -0.01;
	T_LG = Sophus::SE3::exp(T_LG.log()+V6*2);
	V6<<0.01, 0.01, -0.01, 0.01, -0.01, -0.01;
	T_RG = Sophus::SE3::exp(T_RG.log()+V6*2);
	
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
	
	cv::imwrite("before.jpg",img_G);
	cv::namedWindow("img_G");
	cv::imshow("img_G",img_G);
	cv::waitKey(0);
	
	//--------------------------- ROI_F ----------------------------
	int ROI_FL_x = 100;
	int ROI_FL_y = 100;
	int ROI_FL_w = 200;
	int ROI_FL_h = 200;
	
	int ROI_FR_x = 700;
	int ROI_FR_y = 100;
	int ROI_FR_w = 200;
	int ROI_FR_h = 200;
	
	
	//--------------------------- ROI_L ----------------------------
	int ROI_LF_x = 100;
	int ROI_LF_y = 100;
	int ROI_LF_w = 200;
	int ROI_LF_h = 200;
	
	int ROI_LB_x = 100;
	int ROI_LB_y = 750;
	int ROI_LB_w = 200;
	int ROI_LB_h = 200;
	
	
	//--------------------------- ROI_B ----------------------------
	int ROI_BL_x = 100;
	int ROI_BL_y = 700;
	int ROI_BL_w = 200;
	int ROI_BL_h = 200;
	
	int ROI_BR_x = 700;
	int ROI_BR_y = 700;
	int ROI_BR_w = 200;
	int ROI_BR_h = 200;
	
	
	//--------------------------- ROI_R ----------------------------
	int ROI_RF_x = 700;
	int ROI_RF_y = 100;
	int ROI_RF_w = 200;
	int ROI_RF_h = 200;
	
	int ROI_RB_x = 750;
	int ROI_RB_y = 750;
	int ROI_RB_w = 200;
	int ROI_RB_h = 200;
	
	cv::Rect ROI_FL(200,0,150,150);
	cv::Rect ROI_LB(250,750,150,150);
	cv::Rect ROI_BR(600,750,150,150);
	cv::Rect ROI_RF(650,0,150,150);
	

	
	img_F = cv::imread((img_path_template%img_index).str()+ "F.jpg");
	img_L = cv::imread((img_path_template%img_index).str()+ "L.jpg");
	img_B = cv::imread((img_path_template%img_index).str()+ "B.jpg");
	img_R = cv::imread((img_path_template%img_index).str()+ "R.jpg");
	
	img_GF = project_on_ground(img_F,T_FG,K_F,D_F,K_G,rows,cols);
	img_GL = project_on_ground(img_L,T_LG,K_L,D_L,K_G,rows,cols);
	img_GB = project_on_ground(img_B,T_BG,K_B,D_B,K_G,rows,cols);
	img_GR = project_on_ground(img_R,T_RG,K_R,D_R,K_G,rows,cols);
	
	img_G = generate_surround_view(img_GF,img_GL,img_GB,img_GR,rows,cols);

	double rate = 5e-11;
	double decay = 0.95;
	double threshold = 70000;
	int iter_max = 50;
	
	for(int iter=0; iter<iter_max; iter++)
	{ 
		boost::format img_G_template("img/img_G_iter%02d.png");
		cout<<"Writing "<<(img_G_template%iter).str()<<endl;
		cv::imwrite((img_G_template%iter).str(),img_G);
		
		boost::format img_G_RF_template("img/img_G_RF_iter%02d.png");
		cout<<"Writing "<<(img_G_RF_template%iter).str()<<endl;
		cv::imwrite((img_G_RF_template%iter).str(),img_G(ROI_RF));
		
		boost::format img_G_BR_template("img/img_G_BR_iter%02d.png");
		cout<<"Writing "<<(img_G_BR_template%iter).str()<<endl;
		cv::imwrite((img_G_BR_template%iter).str(),img_G(ROI_BR));
		
		boost::format img_G_LB_template("img/img_G_LB_iter%02d.png");
		cout<<"Writing "<<(img_G_LB_template%iter).str()<<endl;
		cv::imwrite((img_G_LB_template%iter).str(),img_G(ROI_LB));
		
		boost::format img_G_FL_template("img/img_G_FL_iter%02d.png");
		cout<<"Writing "<<(img_G_FL_template%iter).str()<<endl;
		cv::imwrite((img_G_FL_template%iter).str(),img_G(ROI_FL));
		
		cv::Mat img_GF_gray, img_GL_gray, img_GB_gray, img_GR_gray;
		cv::cvtColor(img_GF,img_GF_gray,cv::COLOR_BGR2GRAY);
		cv::cvtColor(img_GL,img_GL_gray,cv::COLOR_BGR2GRAY);
		cv::cvtColor(img_GB,img_GB_gray,cv::COLOR_BGR2GRAY);
		cv::cvtColor(img_GR,img_GR_gray,cv::COLOR_BGR2GRAY);
		
		img_GF_gray.convertTo(img_GF_gray, CV_64FC1);
		img_GL_gray.convertTo(img_GL_gray, CV_64FC1);
		img_GB_gray.convertTo(img_GB_gray, CV_64FC1);
		img_GR_gray.convertTo(img_GR_gray, CV_64FC1);
		
		double coef;
		double thr = 16;
		
		cv::Mat diff_RF;
		coef = cv::mean(img_GR_gray(ROI_RF)).val[0]/cv::mean(img_GF_gray(ROI_RF)).val[0];
		cv::subtract(img_GR_gray(ROI_RF),coef*img_GF_gray(ROI_RF),diff_RF,cv::noArray(),CV_64FC1);
		cv::Mat diff_RF2, diff_RF_8U, diff_RF_color;
		cv::exp(-diff_RF/24,diff_RF2);
		diff_RF2 = 255/(1+diff_RF2);
		diff_RF2.convertTo(diff_RF_8U,CV_8U);
		cv::applyColorMap(diff_RF_8U,diff_RF_color, cv::COLORMAP_JET);
		cv::namedWindow("diff_RF",0);
		cv::imshow("diff_RF",diff_RF_color);
		boost::format diff_RF_template("img/diff_RF_iter%02d.png");
		cout<<"Writing "<<(diff_RF_template%iter).str()<<endl;
		cv::imwrite((diff_RF_template%iter).str(),diff_RF_color);

		
		cv::Mat diff_BR;
		coef = cv::mean(img_GB_gray(ROI_BR)).val[0]/cv::mean(img_GR_gray(ROI_BR)).val[0];
		cv::subtract(img_GB_gray(ROI_BR),coef*img_GR_gray(ROI_BR),diff_BR,cv::noArray(),CV_64FC1);
		cv::Mat diff_BR2, diff_BR_8U, diff_BR_color;
		cv::exp(-diff_BR/24,diff_BR2);
		diff_BR2 = 255/(1+diff_BR2);
		diff_BR2.convertTo(diff_BR_8U,CV_8U);
		cv::applyColorMap(diff_BR_8U,diff_BR_color, cv::COLORMAP_JET);
		cv::namedWindow("diff_BR",0);
		cv::imshow("diff_BR",diff_BR_color);
		boost::format diff_BR_template("img/diff_BR_iter%02d.png");
		cout<<"Writing "<<(diff_BR_template%iter).str()<<endl;
		cv::imwrite((diff_BR_template%iter).str(),diff_BR_color);
		
		
		cv::Mat diff_LB;
		coef = cv::mean(img_GL_gray(ROI_LB)).val[0]/cv::mean(img_GB_gray(ROI_LB)).val[0];
		cv::subtract(img_GL_gray(ROI_LB),coef*img_GB_gray(ROI_LB),diff_LB,cv::noArray(),CV_64FC1);
		cv::Mat diff_LB2, diff_LB_8U, diff_LB_color;
		cv::exp(-diff_LB/24,diff_LB2);
		diff_LB2 = 255/(1+diff_LB2);
		diff_LB2.convertTo(diff_LB_8U,CV_8U);
		cv::applyColorMap(diff_LB_8U,diff_LB_color, cv::COLORMAP_JET);
		cv::namedWindow("diff_LB",0);
		cv::imshow("diff_LB",diff_LB_color);
		boost::format diff_LB_template("img/diff_LB_iter%02d.png");
		cout<<"Writing "<<(diff_LB_template%iter).str()<<endl;
		cv::imwrite((diff_LB_template%iter).str(),diff_LB_color);
		
		
		cv::Mat diff_FL;
		coef = cv::mean(img_GF_gray(ROI_FL)).val[0]/cv::mean(img_GL_gray(ROI_FL)).val[0];
		cv::subtract(img_GF_gray(ROI_FL),coef*img_GL_gray(ROI_FL),diff_FL,cv::noArray(),CV_64FC1);
		cv::Mat diff_FL2, diff_FL_8U, diff_FL_color;
		cv::exp(-diff_FL/24,diff_FL2);
		diff_FL2 = 255/(1+diff_FL2);
		diff_FL2.convertTo(diff_FL_8U,CV_8U);
		cv::applyColorMap(diff_FL_8U,diff_FL_color, cv::COLORMAP_JET);
		cv::namedWindow("diff_FL",0);
		cv::imshow("diff_FL",diff_FL_color);
		boost::format diff_FL_template("img/diff_FL_iter%02d.png");
		cout<<"Writing "<<(diff_FL_template%iter).str()<<endl;
		cv::imwrite((diff_FL_template%iter).str(),diff_FL_color);
		
		cout<<"--------------------Calculate rhophi_L--------------------------"<<endl;
		Sophus::SE3 rhophi_L_LB_SE3;
		adjust_pose_V2(img_L,img_B,"L","B",
					rhophi_L_LB_SE3,
					K_L, K_B, 
					D_L, D_B,
					T_LG, T_BG, 
					K_G,
					ROI_LB_x,ROI_LB_y,ROI_LB_w,ROI_LB_h,
					threshold, rate
					);
		
		Sophus::SE3 rhophi_L_LF_SE3;
		adjust_pose_V2(img_L,img_F,"L","F",
					rhophi_L_LF_SE3,
					K_L, K_F, 
					D_L, D_F,
					T_LG, T_FG, 
					K_G,
					ROI_LF_x,ROI_LF_y,ROI_LF_w,ROI_LF_h,
					threshold, rate
					);
		
		T_LG = Sophus::SE3::exp(T_LG.log() - rhophi_L_LB_SE3.log() - rhophi_L_LF_SE3.log());
		T_GL = T_LG.inverse();
		cout<<rhophi_L_LB_SE3.log() + rhophi_L_LF_SE3.log()<<endl<<endl;
		
		cout<<"--------------------Calculate rhophi_R--------------------------"<<endl;
		Sophus::SE3 rhophi_R_RF_SE3;
		adjust_pose_V2(img_R,img_F,"R","F",
					rhophi_R_RF_SE3,
					K_R, K_F, 
					D_R, D_F,
					T_RG, T_FG, 
					K_G,
					ROI_RF_x,ROI_RF_y,ROI_RF_w,ROI_RF_h,
					threshold, rate
					);
		
		Sophus::SE3 rhophi_R_RB_SE3;
		adjust_pose_V2(img_R,img_B,"R","B",
					rhophi_R_RB_SE3,
					K_R, K_B, 
					D_R, D_B,
					T_RG, T_BG, 
					K_G,
					ROI_RB_x,ROI_RB_y,ROI_RB_w,ROI_RB_h,
					threshold, rate
					);
		
		T_RG = Sophus::SE3::exp(T_RG.log() - rhophi_R_RF_SE3.log() - rhophi_R_RB_SE3.log());
		T_GR = T_RG.inverse();
		cout<<rhophi_R_RF_SE3.log() + rhophi_R_RB_SE3.log()<<endl<<endl;
		
		
		cout<<"--------------------Decay rate--------------------------"<<endl;
		cout<<rate<<endl;
		rate *= decay;
		
		cout<<"--------------------Generate new surroundview image--------------------------"<<endl;
		img_GF = project_on_ground(img_F,T_FG,K_F,D_F,K_G,rows,cols);
		img_GL = project_on_ground(img_L,T_LG,K_L,D_L,K_G,rows,cols);
		img_GB = project_on_ground(img_B,T_BG,K_B,D_B,K_G,rows,cols);
		img_GR = project_on_ground(img_R,T_RG,K_R,D_R,K_G,rows,cols);
		
		img_G = generate_surround_view(img_GF,img_GL,img_GB,img_GR,rows,cols);
		
		cv::namedWindow("img_G");
		cv::imshow("img_G",img_G);
		
		// cv::waitKey(0); 
		// cv::imwrite("after.jpg",img_G);
	} 
	cv::imwrite("after.jpg",img_G);
	
	cv::waitKey(0);
	cv::destroyAllWindows();

	return 0;
}