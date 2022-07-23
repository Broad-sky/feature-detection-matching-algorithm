/*
 * Time  : 2022/01/01
 * Github: https://github.com/Broad-sky/feature-detection-matching-algorithm
 * Author: pingsheng shen
 */

#ifndef _AKAZE_MATCH_H_
#define _AKAZE_MATCH_H_
#include"opencv2/opencv.hpp"
#include"data_body.h"


class akaze_match
{
public:
	akaze_match(float inlier_threshold, float nn_match_ratio);
	~akaze_match();
	size_t akaze_forward(cv::Mat & image0, cv::Mat & image1, std::vector<data_match> &vdm);
private:
	float inlier_threshold = 2.5f; // Distance threshold to identify inliers with homography check
	float nn_match_ratio = 0.8f;   // Nearest neighbor matching ratio
};


#endif // !_AKAZE_MATCH_H_


