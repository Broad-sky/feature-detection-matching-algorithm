/*
 * Time  : 2022/01/01
 * Github: https://github.com/Broad-sky/feature-detection-matching-algorithm
 * Author: pingsheng shen
 */

#include"akaze_match.h"


akaze_match::akaze_match(float inlier_threshold, float nn_match_ratio)
{
	inlier_threshold = inlier_threshold;
	nn_match_ratio = nn_match_ratio;
}

akaze_match::~akaze_match()
{
}

size_t akaze_match::akaze_forward(cv::Mat & image0, cv::Mat & image1, std::vector<data_match> &vdm)
{
	std::vector<cv::KeyPoint> kpts1, kpts2;
	cv::Mat desc1, desc2;
	cv::Ptr<cv::AKAZE> akaze = cv::AKAZE::create();
	akaze->detectAndCompute(image0, cv::noArray(), kpts1, desc1);
	akaze->detectAndCompute(image1, cv::noArray(), kpts2, desc2);
	cv::BFMatcher matcher(cv::NORM_HAMMING);
	std::vector< std::vector<cv::DMatch> > nn_matches;
	matcher.knnMatch(desc1, desc2, nn_matches, 2);
	std::vector<cv::KeyPoint> matched1, matched2;
	for (size_t i = 0; i < nn_matches.size(); i++) {
		cv::DMatch first = nn_matches[i][0];
		float dist1 = nn_matches[i][0].distance;
		float dist2 = nn_matches[i][1].distance;
		if (dist1 < nn_match_ratio * dist2) {
			matched1.push_back(kpts1[first.queryIdx]);
			matched2.push_back(kpts2[first.trainIdx]);
		}
	}
	std::vector<cv::DMatch> good_matches;
	std::vector<cv::KeyPoint> inliers1, inliers2;
	for (size_t i = 0; i < matched1.size(); i++) {
		data_match dm;
		if (true) {
			int new_i = static_cast<int>(inliers1.size());
			inliers1.push_back(matched1[i]);
			inliers2.push_back(matched2[i]);
			good_matches.push_back(cv::DMatch(new_i, new_i, 0));
			dm.valid_keypoint_x0 = matched1[i].pt.x;
			dm.valid_keypoint_y0 = matched1[i].pt.y;
			dm.valid_keypoint_x1 = matched2[i].pt.x;
			dm.valid_keypoint_y1 = matched2[i].pt.y;
			dm.msscores0 = 1.0;
			vdm.push_back(dm);
		}
	}

	std::cout << "A-KAZE Matching Results" << std::endl;
	std::cout << "*******************************" << std::endl;
	std::cout << "# Keypoints 1:                        \t" << kpts1.size() << std::endl;
	std::cout << "# Keypoints 2:                        \t" << kpts2.size() << std::endl;
	std::cout << "# Matches:                            \t" << good_matches.size() << std::endl;
	std::cout << "# vdm size:                      \t" << vdm.size() << std::endl;
	std::cout << std::endl;

	return vdm.size();
}