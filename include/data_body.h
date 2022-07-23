/*
 * Time  : 2022/01/01
 * Github: https://github.com/Broad-sky/feature-detection-matching-algorithm
 * Author: pingsheng shen
 */

#ifndef _DATA_BODY_H_
#define _DATA_BODY_H_
#include<vector>
#include<string>

struct lm_params
{
	int32_t batch_size;
	int32_t dla_core;

	bool int8;
	bool fp16;
	bool seria;

	std::vector<std::string> data_dirs;
	std::vector<std::string> input_names;
	std::vector<std::string> output_names;
};

struct point_lm_params : public lm_params
{
	std::string point_weight_file;
	int input_w;
	int input_h;
	float scores_thresh;
	int border;
};

struct match_lm_params : public lm_params
{
	std::string match_weight_file;
	float match_threshold;
};

struct infer_deleter
{
	template<typename T>
	void operator()(T*obj) const
	{
		if (obj)
		{
			obj->destroy();
		}
	}
};

struct data
{
	float x;
	float y;
	float s;
};

struct data_point
{
	float desc_w;
	float desc_h;
	int keypoint_size;
	float * descriptors;
	float * scores;
	float * keypoints;

	std::vector<data> vdt;
	int status_code;  // normal 1, error 0
};

struct maxv_indices
{
	float max_value;
	int indices;
};

struct data_match
{
	float valid_keypoint_x0;
	float valid_keypoint_y0;
	float valid_keypoint_x1;
	float valid_keypoint_y1;
	float msscores0;
};

struct data_image
{
	cv::Mat srcimg0;
	cv::Mat srcimg1;
	cv::Mat srcimg0_gray;
	cv::Mat srcimg1_gray;
	cv::Mat resized_srcimg0;
	cv::Mat resized_srcimg1;
	cv::Mat warpP0;
	cv::Mat warpP1;
	cv::Mat out_gray;
	cv::Mat out_rgb;
	cv::Mat out_match;
	cv::Mat H;

};

#endif // !_DATA_BODY_H_

