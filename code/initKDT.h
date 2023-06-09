#ifndef INITKDT_h_
#define INITKDT_h_

void initializePose(Sophus::SE3& T_FG,Sophus::SE3& T_LG,
					Sophus::SE3& T_BG,Sophus::SE3& T_RG);
void initializeK(Eigen::Matrix3d& K_F, Eigen::Matrix3d& K_L,
				 Eigen::Matrix3d& K_B, Eigen::Matrix3d& K_R);
void initializeD(Eigen::Vector4d& D_F, Eigen::Vector4d& D_L,
				 Eigen::Vector4d& D_B, Eigen::Vector4d& D_R);

#endif 