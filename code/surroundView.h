#ifndef SURROUNDVIEW_h_
#define SURROUNDVIEW_h_

cv::Mat eigen2mat(Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> A);
void imshow_64F(cv::Mat img,std::string img_name);
cv::Mat bilinear_interpolation(cv::Mat img,cv::Mat pix_table,int rows, int cols);
cv::Mat project_on_ground(cv::Mat img, Sophus::SE3 T_CG,
						  Eigen::Matrix3d K_C,Eigen::Vector4d D_C,
						  cv::Mat K_G,int rows, int cols);
cv::Mat generate_surround_view(cv::Mat img_GF, cv::Mat img_GL, 
							   cv::Mat img_GB, cv::Mat img_GR, 
							   int rows, int cols);

void adjust_pose(cv::Mat img_GA, cv::Mat img_GB,
				 Eigen::Matrix<double,6,1>& rhophi_A,
				 Eigen::Matrix<double,6,1>& rhophi_B,
				 cv::Mat K_G,
				 int overlap_x, int overlap_y, int overlap_w, int overlap_h,
				 double lr);

cv::Mat ground2cam(int x,int y,cv::Mat K_G, Sophus::SE3 T_CG, Eigen::Matrix3d K_C);

void calc_ROI_AB(
				 cv::Mat& img_A, cv::Mat& img_B,
				 cv::Mat& ROI_on_A, cv::Mat& ROI_on_A_from_B,cv::Mat& P_As,
				 Eigen::Matrix3d K_A, Eigen::Matrix3d K_B, 
				 Eigen::Vector4d D_A, Eigen::Vector4d D_B,
				 Sophus::SE3 T_AG, Sophus::SE3 T_BG, 
				 cv::Mat K_G,
				 int ROI_x,int ROI_y,int ROI_w,int ROI_h
				);

void adjust_pose_V2(
					cv::Mat img_A, cv::Mat img_B,std::string A,std::string B,
					Sophus::SE3& rhophi_A_SE3,
					Eigen::Matrix3d K_A, Eigen::Matrix3d K_B, 
					Eigen::Vector4d D_A, Eigen::Vector4d D_B,
					Sophus::SE3 T_AG, Sophus::SE3 T_BG, 
					cv::Mat K_G,
					int ROI_x,int ROI_y,int ROI_w,int ROI_h,
					double threshold, double rate
				   );

#endif 