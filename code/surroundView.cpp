#include <iostream>
#include <string>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include "sophus/se3.h"
#include "sophus/so3.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
using namespace std;

cv::Mat eigen2mat(Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> A)
{
	cv::Mat B;
	cv::eigen2cv(A,B);
	
	return B;
}

void imshow_64F(cv::Mat img,string img_name)
{
	cv::Mat img_norm,img_color;
	cv::normalize(img,img_norm,0,255,cv::NORM_MINMAX,CV_8U);
	cv::applyColorMap(img_norm,img_color, cv::COLORMAP_JET);
	cv::namedWindow(img_name,0);
	cv::imshow(img_name,img_color);
	
	return;
}

void imshow_64F_gray(cv::Mat img,string img_name)
{
	cv::Mat img_8U;
	img.convertTo(img_8U,CV_8U);
	cv::namedWindow(img_name,0);
	cv::imshow(img_name,img_8U);
	
	return;
}

cv::Mat bilinear_interpolation(cv::Mat img,cv::Mat pix_table,int rows, int cols)
{
	cv::Mat img_G(rows,cols,CV_8UC3);
	// f is float
	cv::Mat img_G_f(rows,cols,CV_64FC3);
	cv::Mat img_f;
	img.convertTo(img_f, CV_64FC3);
	float x;
	float y;
	int x_floor;
	int y_floor;
	int x_ceil;
	int y_ceil;
	cv::Vec3d ul;
	cv::Vec3d ur;
	cv::Vec3d dl;
	cv::Vec3d dr;
	cv::Vec3d pix;
	
	
	for(int i=0;i<rows;i++)
	{
		for(int j=0;j<cols;j++)
		{
			x = pix_table.at<cv::Vec2d>(i,j)[1];
			y = pix_table.at<cv::Vec2d>(i,j)[0];
			if(x<0 || y<0 || (y>=img.cols-1) || (x>=img.rows-1))
			{
				img_G_f.at<cv::Vec3d>(i,j) = cv::Vec3d(0,0,0);
			}
			else{
				x_floor = int(floor(x));
				x_ceil = x_floor+1;
				y_floor = int(floor(y));
				y_ceil = y_floor+1;
				
				ul = img_f.at<cv::Vec3d>(x_floor,y_floor);
				ur = img_f.at<cv::Vec3d>(x_ceil,y_floor);
				dl = img_f.at<cv::Vec3d>(x_floor,y_ceil);
				dr = img_f.at<cv::Vec3d>(x_ceil,y_ceil);
// 				pix = (ul*(x-x_floor)+ur*(x_ceil-x))*(y-y_floor)
// 					 +(dl*(x-x_floor)+dr*(x_ceil-x))*(y_ceil-y);
				pix = (ur*(x-x_floor)+ul*(x_ceil-x))*(y_ceil-y)
					 +(dr*(x-x_floor)+dl*(x_ceil-x))*(y-y_floor);
				if(pix(0)>=0   && pix(1)>=0   && pix(2)>=0 && 
				   pix(0)<=255 && pix(1)<=255 && pix(2)<=255)
				{
					img_G_f.at<cv::Vec3d>(j,i) = pix;
				}
			}
		}
	}
	
	img_G_f.convertTo(img_G, CV_8UC3);
// 	cout<<img_G_f(cv::Rect(0,0,5,1))<<endl<<endl;
// 	cout<<img_G(cv::Rect(0,0,5,1))<<endl<<endl;
	
	return img_G;
// 	return img_G_f;
}

cv::Mat project_on_ground(cv::Mat img, Sophus::SE3 T_CG,
						  Eigen::Matrix3d K_C,Eigen::Vector4d D_C,
						  cv::Mat K_G,int rows, int cols)
{
// 	cout<<"--------------------Init p_G and P_G------------------------"<<endl;
	cv::Mat p_G = cv::Mat::ones(3,rows*cols,CV_64FC1);
	for(int i=0;i<rows;i++)
	{
		for(int j=0;j<cols;j++)
		{
			p_G.at<double>(0,cols*i+j) = j;
			p_G.at<double>(1,cols*i+j) = i;
		}
	}
	
	cv::Mat P_G = cv::Mat::ones(4,rows*cols,CV_64FC1);
	P_G(cv::Rect(0,0,rows*cols,3)) = K_G.inv()*p_G;
	P_G(cv::Rect(0,2,rows*cols,1)) = 0;
	
// 	cout<<"--------------------Init P_GF------------------------"<<endl;
// 	cout<<"P_GC: "<<endl;
// 	cout<<P_G(cv::Rect(0,0,100,4))<<endl<<endl;
	cv::Mat P_GC = cv::Mat::zeros(4,rows*cols,CV_64FC1);
	cv::Mat T_CG_(4,4,CV_64FC1);
	cv::eigen2cv(T_CG.matrix(),T_CG_);
	P_GC =  T_CG_ * P_G;
// 	cout<<"P_GC: "<<endl;
// 	cout<<P_GC(cv::Rect(0,0,100,4))<<endl<<endl;
	
// 	cout<<"--------------------Init P_GF1------------------------"<<endl;
	cv::Mat P_GC1 = cv::Mat::zeros(1,rows*cols,CV_64FC2);
	vector<cv::Mat> channels(2);
	cv::split(P_GC1, channels);
	channels[0] = P_GC(cv::Rect(0,0,rows*cols,1))/P_GC(cv::Rect(0,2,rows*cols,1));
	channels[1] = P_GC(cv::Rect(0,1,rows*cols,1))/P_GC(cv::Rect(0,2,rows*cols,1));
	cv::merge(channels, P_GC1);
// 	cout<<"P_GC1: "<<endl;
// 	cout<<P_GC1(cv::Rect(0,0,5,1))<<endl<<endl;
	
// 	cout<<"--------------------Init p_GF------------------------"<<endl;
// 	cout<<K_C.cols()<<endl;
	cv::Mat p_GC = cv::Mat::zeros(1,rows*cols,CV_64FC2);
// 	cout<<eigen2mat(K_C)<<endl;
	vector<double> D_C_{D_C(0,0),D_C(1,0),D_C(2,0),D_C(3,0)};
	cv::fisheye::distortPoints(P_GC1,p_GC,eigen2mat(K_C),D_C_);
// 	cout<<"p_GC: "<<endl;
	p_GC.reshape(rows,cols);
	cv::Mat p_GC_table = p_GC.reshape(0,rows);
	vector<cv::Mat> p_GC_table_channels(2);
	cv::split(p_GC_table, p_GC_table_channels);
	
	cv::Mat p_GC_table_32F;
	p_GC_table.convertTo(p_GC_table_32F,CV_32FC2);
	
	cv::Mat img_GC;
	cv::remap(img,img_GC,p_GC_table_32F,cv::Mat(),cv::INTER_LINEAR);
// 	img_GC = bilinear_interpolation(img,p_GC_table,rows,cols);
// 	cout<<img_GC.size<<endl;
	
	return img_GC;
}


cv::Mat generate_surround_view(cv::Mat img_GF, cv::Mat img_GL, 
							   cv::Mat img_GB, cv::Mat img_GR, 
							   int rows, int cols)
{
	cv::Mat img_G(rows,cols,CV_8UC3);
	for(int i=0;i<rows;i++)
	{
		for(int j=0;j<cols;j++)
		{
			if(i>2*j-500)
			{
				if(i>-2*j+1500)
				{
					img_G.at<cv::Vec3b>(i,j) = img_GB.at<cv::Vec3b>(i,j);
				}
				else
				{
					img_G.at<cv::Vec3b>(i,j) = img_GL.at<cv::Vec3b>(i,j);
				}
				
			}
			else
			{
				if(i>-2*j+1500)
				{
					img_G.at<cv::Vec3b>(i,j) = img_GR.at<cv::Vec3b>(i,j);
				}
				else
				{
					img_G.at<cv::Vec3b>(i,j) = img_GF.at<cv::Vec3b>(i,j);
				}
			}
			
		}
	}
	
	for(int i=300;i<700;i++)
	{
		for(int j=400;j<600;j++)
		{
			img_G.at<cv::Vec3b>(i,j) = cv::Vec3b(0,0,0);
		}
	}
	
	return img_G;
}

void adjust_pose(cv::Mat img_GA, cv::Mat img_GB,
				 Eigen::Matrix<double,6,1>& rhophi_A,
				 Eigen::Matrix<double,6,1>& rhophi_B,
				 cv::Mat K_G,
				 int overlap_x, int overlap_y, int overlap_w, int overlap_h,
				 double lr)
{
	cv::Rect rectAB(overlap_x,overlap_y ,overlap_w,overlap_h);
	double fx = K_G.at<double>(0,0);
	double fy = K_G.at<double>(1,1);
	double cols = K_G.at<double>(0,2) * 2;
	double rows = K_G.at<double>(1,2) * 2;
	
	cv::Mat img_GA_gray;
	cv::cvtColor(img_GA(rectAB),img_GA_gray,cv::COLOR_BGR2GRAY);
	cv::medianBlur(img_GA_gray,img_GA_gray,5);
	img_GA_gray.convertTo(img_GA_gray, CV_64FC1);
	
	cv::Mat img_GB_gray;
	cv::cvtColor(img_GB(rectAB),img_GB_gray,cv::COLOR_BGR2GRAY);
	cv::medianBlur(img_GB_gray,img_GB_gray,5);
	img_GB_gray.convertTo(img_GB_gray, CV_64FC1);
 
	cv::Mat diff_AB;
	double coef = cv::mean(img_GA_gray).val[0]/cv::mean(img_GB_gray).val[0] ;
	cv::subtract(img_GA_gray,coef*img_GB_gray,diff_AB,cv::noArray(),CV_64FC1);
	cv::Mat diff_BA = -diff_AB;
	
	cv::Mat diff_AB_norm,diff_AB_color;
	cv::normalize(diff_AB,diff_AB_norm,0,255,cv::NORM_MINMAX,CV_8U);
	cv::applyColorMap(diff_AB_norm,diff_AB_color, cv::COLORMAP_JET);
	cv::namedWindow("diff_AB",0);
	cv::imshow("diff_AB",diff_AB_color);
	
	cv::Mat grad_GA_x,grad_GA_y;
	cv::Scharr(img_GA_gray, grad_GA_x, CV_64FC1, 1, 0, 1, 0, cv::BORDER_DEFAULT);
	cv::Scharr(img_GA_gray, grad_GA_y, CV_64FC1, 0, 1, 1, 0, cv::BORDER_DEFAULT);
	
	cv::Mat grad_GA_xy;
	cv::Mat pow_GA_x,pow_GA_y;
	cv::pow(grad_GA_x,2,pow_GA_x);
	cv::pow(grad_GA_y,2,pow_GA_y);
	cv::sqrt(pow_GA_x+pow_GA_y,grad_GA_xy);
// 	cout<<grad_GA_xy<<endl;
	
	cv::Mat grad_GB_x,grad_GB_y;
	cv::Scharr(img_GB_gray, grad_GB_x, CV_64FC1, 1, 0, 1, 0, cv::BORDER_DEFAULT);
	cv::Scharr(img_GB_gray, grad_GB_y, CV_64FC1, 0, 1, 1, 0, cv::BORDER_DEFAULT);
	
	cv::Mat grad_GB_xy;
	cv::Mat pow_GB_x,pow_GB_y;
	cv::pow(grad_GB_x,2,pow_GB_x);
	cv::pow(grad_GB_y,2,pow_GB_y);
	cv::sqrt(pow_GB_x+pow_GB_y,grad_GB_xy);
// 	cout<<grad_GB_xy<<endl;
	
	cv::Mat multi_GA_x,multi_GA_y;
	cv::multiply(diff_AB, grad_GA_x, multi_GA_x,1,CV_64FC1);
	cv::multiply(diff_AB, grad_GA_y, multi_GA_y,1,CV_64FC1);
	
	cv::Mat multi_GA_x_norm,multi_GA_x_color;
	cv::normalize(multi_GA_x,multi_GA_x_norm,0,255,cv::NORM_MINMAX,CV_8U);
	cv::applyColorMap(multi_GA_x_norm,multi_GA_x_color, cv::COLORMAP_JET);
	cv::namedWindow("multi_GA_x",0);
	cv::imshow("multi_GA_x",multi_GA_x_color);
	
	cv::Mat multi_GA_y_norm,multi_GA_y_color;
	cv::normalize(multi_GA_y,multi_GA_y_norm,0,255,cv::NORM_MINMAX,CV_8U);
	cv::applyColorMap(multi_GA_y_norm,multi_GA_y_color, cv::COLORMAP_JET);
	cv::namedWindow("multi_GA_y",0);
	cv::imshow("multi_GA_y",multi_GA_y_color);
	
	cv::Mat multi_GB_x,multi_GB_y;
	cv::multiply(diff_BA, grad_GB_x, multi_GB_x,1,CV_64FC1);
	cv::multiply(diff_BA, grad_GB_y, multi_GB_y,1,CV_64FC1);
	

	Eigen::Matrix<double,6,1>  delta_A,delta_B;
	rhophi_A<<0,0,0,0,0,0;
	rhophi_B<<0,0,0,0,0,0;
	double X,Y,Z;
	Z = 0;
	double mx_A,my_A,mx_B,my_B;
	int count_A = 0;
	int count_B = 0;
	for(int y=0;y<overlap_h;y++)
	{
		for(int x=0;x<overlap_w;x++)
		{
			X =  (x+overlap_x-cols/2)/fx;
			Y =  (y+overlap_y-rows/2)/fy;
			if(x%10==0 && y%10==0)
			{
				cout<<"("<<X<<","<<Y<<"), ";
			}
			
			// calculate update rhophi for camera A
			if(abs(grad_GA_xy.at<double>(y,x))>100)
			{
				count_A++;
				mx_A = multi_GA_x.at<double>(y,x);
				my_A = multi_GA_y.at<double>(y,x);
				delta_A<<mx_A*fx,
						 my_A*fy,
						 0,
						 0,
						 0,
						 mx_A*fx*Y - my_A*fy*X;
// 				delta_A<<0,
// 						 0,
// 						 mx_A*fx*Y - my_A*fy*X,
// 						 mx_A*fx,
// 						 my_A*fy,
// 						 0;
				rhophi_A += delta_A;
			}
			
			
			// calculate update rhophi for camera B
			if(abs(grad_GB_xy.at<double>(y,x))>100)
			{
				count_B++;
				mx_B = multi_GB_x.at<double>(y,x);
				my_B = multi_GB_y.at<double>(y,x);
				delta_B<<mx_B*fx,
						 my_B*fy,
						 0,
						 0,
						 0,
						 mx_B*fx*Y - my_B*fy*X;
// 				delta_B<<0,
// 						 0,
// 						 mx_B*fx*Y - my_B*fy*X,
// 						 mx_B*fx,
// 						 my_B*fy,
// 						 0;
				rhophi_B += delta_B;
			}
		}
		
		if(y%10==0)
		{
			cout<<endl;
		}
	}
	
	cout<<"count_A: "<<count_A<<endl;
	cout<<"count_B: "<<count_B<<endl;
	
	rhophi_A = lr * (rhophi_A/count_A);
	rhophi_B = lr * (rhophi_B/count_B);
	
	return;
}

cv::Mat ground2cam(int x,int y, cv::Mat K_G, Sophus::SE3 T_CG, Eigen::Matrix3d K_C)
{
	cv::Mat p_G = cv::Mat::ones(3,1,CV_64FC1);
	p_G.at<double>(0,0) = x;
	p_G.at<double>(1,0) = y;
// 	cout<<p_G<<endl;
	cv::Mat P_G = cv::Mat::ones(4,1,CV_64FC1);
	P_G(cv::Rect(0,0,1,3)) = K_G.inv()*p_G;
	P_G.at<double>(0,2) = 0;
// 	cout<<P_G<<endl;
	cv::Mat P_C = eigen2mat(T_CG.matrix())*P_G;
// 	cout<<P_C<<endl;
	cv::Mat P_C_1 = P_C(cv::Rect(0,0,1,3))/P_C.at<double>(0,2);
// 	cout<<P_C_1<<endl;
	cv::Mat p_C = eigen2mat(K_C)*P_C_1;
// 	cout<<p_C<<endl;
	
	return p_C;
}

void calc_ROI_AB( 
				 cv::Mat& img_A, cv::Mat& img_B,
				 cv::Mat& ROI_on_A, cv::Mat& ROI_on_A_from_B, cv::Mat& P_As,
				 Eigen::Matrix3d K_A, Eigen::Matrix3d K_B, 
				 Eigen::Vector4d D_A, Eigen::Vector4d D_B,
				 Sophus::SE3 T_AG, Sophus::SE3 T_BG, 
				 cv::Mat K_G,
				 int ROI_x,int ROI_y,int ROI_w,int ROI_h
				)
{
	Sophus::SE3 T_GA = T_AG.inverse();
	Sophus::SE3 T_GB = T_BG.inverse();
	cv::Mat p_A_ul = ground2cam(ROI_x,       ROI_y,
								K_G, T_AG, K_A);
	cv::Mat p_A_ur = ground2cam(ROI_x+ROI_w, ROI_y,
								K_G, T_AG, K_A);
	cv::Mat p_A_dl = ground2cam(ROI_x,       ROI_y+ROI_h,
								K_G, T_AG, K_A);
	cv::Mat p_A_dr = ground2cam(ROI_x+ROI_w, ROI_y+ROI_h,
								K_G, T_AG, K_A);
	
// 	cout<<"--------------------determin ROI on camera A--------------------------"<<endl;
	vector<double> p_x{p_A_ul.at<double>(0,0),p_A_ur.at<double>(0,0),p_A_dl.at<double>(0,0),p_A_dr.at<double>(0,0)};
	vector<double> p_y{p_A_ul.at<double>(1,0),p_A_ur.at<double>(1,0),p_A_dl.at<double>(1,0),p_A_dr.at<double>(1,0)};
	int x_max = int(*max_element(p_x.begin(),p_x.end()));
	int x_min = int(*min_element(p_x.begin(),p_x.end()));
	int y_max = int(*max_element(p_y.begin(),p_y.end()));
	int y_min = int(*min_element(p_y.begin(),p_y.end()));
	int x_width  = x_max-x_min;
	int y_height = y_max-y_min;
	cv::Rect rect_AB_on_A(x_min, y_min, x_width, y_height);
	
// 	cout<<"--------------------cut off ROA on camera A--------------------------"<<endl;
	cv::Size sizeROI(x_width,y_height);
	cv::Mat K_A_mat = eigen2mat(K_A);
	cv::Mat K_A_mat_ROI = K_A_mat.clone();
	K_A_mat_ROI.at<double>(0,2) = K_A_mat_ROI.at<double>(0,2) - x_min;
	K_A_mat_ROI.at<double>(1,2) = K_A_mat_ROI.at<double>(1,2) - y_min;
	cv::fisheye::undistortImage(img_A,ROI_on_A,
								K_A_mat,eigen2mat(D_A),
								K_A_mat_ROI,sizeROI);
	
// 	cout<<"--------------------project ROI from camera B to camera A --------------------------"<<endl;
	double a7 = T_GA.matrix()(2,0);
	double a8 = T_GA.matrix()(2,1);
	double a9 = T_GA.matrix()(2,2);
	double t3 = T_GA.matrix()(2,3);
	cv::Mat p_A, p_A_1, P_A, P_G_A, P_B_A, P_B_A_1;
	P_As = cv::Mat::zeros(y_height,x_width,CV_64FC3);
	cv::Mat P_B_A_1s = cv::Mat::zeros(y_height,x_width,CV_64FC2);
	p_A = cv::Mat::ones(3,1,CV_64FC1);
	for(int i=0;i<x_width;i++)
	{
		for(int j=0;j<y_height;j++)
		{
			// each pixel on ROI_on_A
			p_A.at<double>(0,0) = i;
			p_A.at<double>(1,0) = j;
			p_A_1 = K_A_mat_ROI.inv()*p_A;
			// calculate depth and coodinate of the pixel on camera A
			double Z_A = -t3/(a7*p_A_1.at<double>(0,0) + a8*p_A_1.at<double>(1,0) + a9);
			P_A = cv::Mat::ones(4,1,CV_64FC1);
			P_A(cv::Rect(0,0,1,3)) = Z_A*p_A_1;
			P_As.at<cv::Vec3d>(j,i) = cv::Vec3d(P_A.at<double>(0,0),
												P_A.at<double>(1,0),
												P_A.at<double>(2,0));
			// calculate the normalized coodinate of the pixel on camera B
			P_G_A = eigen2mat(T_GA.matrix())*P_A;
			P_B_A = eigen2mat(T_BG.matrix())*P_G_A;
			P_B_A_1 = P_B_A(cv::Rect(0,0,1,2))/P_B_A.at<double>(2,0);
			P_B_A_1s.at<cv::Vec2d>(j,i) = cv::Vec2d(P_B_A_1.at<double>(0,0),
													P_B_A_1.at<double>(1,0));
		}
	}
	// calculate pixel coodinate of ROI_on_A on camera B, namely the project map from B to A
	cv::Mat p_B_As;
	cv::fisheye::distortPoints(P_B_A_1s,p_B_As,eigen2mat(K_B),eigen2mat(D_B));
	cv::Mat p_B_As_32F;
	p_B_As.convertTo(p_B_As_32F,CV_32FC2);
	
	cv::remap(img_B,ROI_on_A_from_B,p_B_As_32F,cv::Mat(),cv::INTER_LINEAR);
	
	return;
}

void adjust_pose_V2(
					cv::Mat img_A, cv::Mat img_B,std::string A,std::string B,
					Sophus::SE3& rhophi_A_SE3,
					Eigen::Matrix3d K_A, Eigen::Matrix3d K_B, 
					Eigen::Vector4d D_A, Eigen::Vector4d D_B,
					Sophus::SE3 T_AG, Sophus::SE3 T_BG, 
					cv::Mat K_G,
					int ROI_x,int ROI_y,int ROI_w,int ROI_h,
					double threshold,double rate
)
{
	cv::Mat ROI_on_A;
	cv::Mat ROI_on_A_from_B;
	cv::Mat P_As;
	
	calc_ROI_AB(img_A, img_B,
				ROI_on_A, ROI_on_A_from_B, P_As,
				K_A, K_B, 
				D_A,D_B,
				T_AG,T_BG, 
				K_G,
				ROI_x, ROI_y, ROI_w,ROI_h);
	
	cv::Mat ROI_on_A_gray;
	cv::Mat ROI_on_A_gray_blur;
	cv::Mat ROI_on_A_gray_64F;
	cv::cvtColor(ROI_on_A,ROI_on_A_gray,cv::COLOR_BGR2GRAY);
	cv::medianBlur(ROI_on_A_gray,ROI_on_A_gray_blur,5);
	ROI_on_A_gray_blur.convertTo(ROI_on_A_gray_64F, CV_64FC1);
	
	cv::Mat ROI_on_A_from_B_gray;
	cv::Mat ROI_on_A_from_B_gray_blur;
	cv::Mat ROI_on_A_from_B_gray_64F;
	cv::cvtColor(ROI_on_A_from_B,ROI_on_A_from_B_gray,cv::COLOR_BGR2GRAY);
	cv::medianBlur(ROI_on_A_from_B_gray,ROI_on_A_from_B_gray_blur,5);
	ROI_on_A_from_B_gray_blur.convertTo(ROI_on_A_from_B_gray_64F, CV_64FC1);
	
	cv::Mat diff_AB;
	double coef = cv::mean(ROI_on_A_gray_64F).val[0]/cv::mean(ROI_on_A_from_B_gray_64F).val[0];
	cv::subtract(ROI_on_A_gray_64F,coef*ROI_on_A_from_B_gray_64F,diff_AB,cv::noArray(),CV_64FC1);
// 	imshow_64F(diff_AB,"diff_"+A+B);
// 	cv::Mat diff_AB2, diff_AB_8U, diff_AB_color;
// // 	cout<<diff_AB(cv::Rect(0,0,10,10))<<endl;
// 	cv::exp(-diff_AB/24,diff_AB2);
// // 	cout<<diff_AB2(cv::Rect(0,0,10,10))<<endl;
// 	diff_AB2 = 255/(1+diff_AB2);
// // 	cout<<diff_AB2(cv::Rect(0,0,10,10))<<endl;
// 	diff_AB2.convertTo(diff_AB_8U,CV_8U);
// 	cv::applyColorMap(diff_AB_8U,diff_AB_color, cv::COLORMAP_JET);
// 	cv::namedWindow("diff_"+A+B,0);
// 	cv::imshow("diff_"+A+B,diff_AB_color);
	
	cv::Mat grad_GA_x,grad_GA_y;
// 	cv::Scharr(ROI_on_A_gray_64F, grad_GA_x, CV_64FC1, 1, 0, 1, 0, cv::BORDER_DEFAULT);
// 	cv::Scharr(ROI_on_A_gray_64F, grad_GA_y, CV_64FC1, 0, 1, 1, 0, cv::BORDER_DEFAULT);
	cv::Sobel(ROI_on_A_gray_64F, grad_GA_x, CV_64FC1, 1, 0, 7);
	cv::Sobel(ROI_on_A_gray_64F, grad_GA_y, CV_64FC1, 0, 1, 7);
// 	cv::Mat grad_GA_xy;
// 	cv::Mat pow_grad_GA_x,pow_grad_GA_y;
// 	cv::pow(grad_GA_x,2,pow_grad_GA_x);
// 	cv::pow(grad_GA_y,2,pow_grad_GA_y);
// 	cv::sqrt(pow_grad_GA_x+pow_grad_GA_y,grad_GA_xy);
	
	cv::Mat multi_GA_x,multi_GA_y;
	cv::multiply(diff_AB, grad_GA_x, multi_GA_x,1,CV_64FC1);
	cv::multiply(diff_AB, grad_GA_y, multi_GA_y,1,CV_64FC1);
	cv::Mat multi_GA_xy;
	cv::Mat pow_multi_GA_x,pow_multi_GA_y;
	cv::pow(multi_GA_x,2,pow_multi_GA_x);
	cv::pow(multi_GA_y,2,pow_multi_GA_y);
	cv::sqrt(pow_multi_GA_x+pow_multi_GA_y,multi_GA_xy);
	
	double fx = K_A(0,0);
	double fy = K_A(1,1);
	double X,Y,Z;
	Eigen::Matrix<double,1,2> J_error_xy;
	Eigen::Matrix<double,2,6> J_xy_ksai;
	Eigen::Matrix<double,1,6> rhophi,J_error_ksai;
	Eigen::Matrix<double,6,6> H;
	Eigen::Matrix<double,6,1> g;
	rhophi<<0,0,0,0,0,0;
	int count = 0;
	
	for(int j=0; j<ROI_on_A.rows; j++)
	{
		for(int i=0; i<ROI_on_A.cols; i++)
		{
			Z = P_As.at<cv::Vec3d>(j,i).val[2];
			if(multi_GA_xy.at<double>(j,i)>threshold && Z>0.1)
			{
				X = P_As.at<cv::Vec3d>(j,i).val[0];
				Y = P_As.at<cv::Vec3d>(j,i).val[1];
				Z = P_As.at<cv::Vec3d>(j,i).val[2];
				
				J_error_xy(0,0) = multi_GA_x.at<double>(j,i);
				J_error_xy(0,1) = multi_GA_y.at<double>(j,i);
				
				
				J_xy_ksai(0,0) = fx/Z;
				J_xy_ksai(0,1) = 0;
				J_xy_ksai(0,2) = -(fx*X)/(Z*Z);
				J_xy_ksai(0,3) = -(fx*X*Y)/(Z*Z);
				J_xy_ksai(0,4) = fx+(fx*X*X)/(Z*Z);
				J_xy_ksai(0,5) = -(fx*Y)/Z;
				
				
				J_xy_ksai(1,0) = 0;
				J_xy_ksai(1,1) = fy/Z;
				J_xy_ksai(1,2) = -(fy*Y)/(Z*Z);
				J_xy_ksai(1,3) = -fy-(fy*Y*Y)/(Z*Z);
				J_xy_ksai(1,4) = (fy*X*Y)/(Z*Z);
				J_xy_ksai(1,5) = (fy*X)/Z;
				
// 				J_error_ksai =J_error_xy * J_xy_ksai * 1e-10;
// 				
// 				H +=  J_error_ksai.transpose() * J_error_ksai;
// 				g += -J_error_ksai.transpose() * diff_AB.at<double>(j,i);
// 				
				
				rhophi += J_error_xy*J_xy_ksai;
				
				
				count++;
				ROI_on_A_gray.at<char>(j,i) = 0;
			}
		}
	}

	
	
	cout<<"=========================="<<endl<<endl;
	
	if(count>0)
	{
// 		H /= count;
// 		g /= count;
// 		cout<<g<<endl<<endl;
// 		cout<<H<<endl<<endl;
// 		cout<<H.inverse()<<endl<<endl;
// 		rhophi = H.inverse()*g;
		rhophi /= count;
	}
	rhophi *= rate;
	cout<<"rhophi: "<<rhophi<<endl;
	
	cout<<"=========================="<<endl<<endl;
	
	
// 	imshow_64F_gray(ROI_on_A_gray,"ROI_"+A+B+"_on_"+A);
// 	imshow_64F_gray(ROI_on_A_from_B_gray,"ROI_"+A+B+"_on_"+A+"_from_"+B);
	
	rhophi_A_SE3 = Sophus::SE3::exp(rhophi);
	
	return;
}


