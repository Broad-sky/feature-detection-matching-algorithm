/*
 * Time  : 2022/01/01
 * Github: https://github.com/Broad-sky/feature-detection-matching-algorithm
 * Author: pingsheng shen
 */

#ifndef _LOCAL_MATCH_H_
#define _LOCAL_MATCH_H_
#include<memory>
#include<fstream>
#include<NvInfer.h>
#include"cuda_runtime_api.h"
#include<NvInferPlugin.h>
#include<math.h>
#include"opencv2/opencv.hpp"
#include"logging.h"
#include"data_body.h"
#include"tools.h"

class point_lm_trt
{
public:
	point_lm_trt();
	~point_lm_trt();
	bool initial_point_model();
	size_t forward(cv::Mat& srcimg, data_point& dp);
	int get_input_h();
	int get_input_w();

private:
	template<typename T>
	using _unique_ptr = std::unique_ptr<T, infer_deleter>;
	std::shared_ptr< nvinfer1::ICudaEngine> _engine_ptr;
	std::shared_ptr< nvinfer1::IExecutionContext> _context_ptr;
	point_lm_params _point_lm_params;


private:
	bool build_model();
	float * imnormalize(cv::Mat & img);
};

class match_lm_trt
{
public:
	match_lm_trt();
	~match_lm_trt();
	bool initial_match_model();
	size_t forward(data_point & dp0, data_point & dp1, std::vector<data_match> &vdm);

private:
	template<typename T>
	using _unique_ptr = std::unique_ptr<T, infer_deleter>;
	std::shared_ptr< nvinfer1::ICudaEngine> _engine_ptr;
	std::shared_ptr< nvinfer1::IExecutionContext> _context_ptr;
	match_lm_params _match_lm_params;

private:
	bool build_model();
	float * log_optimal_transport(float * scores, float bin_score, 
		int iters, int scores_output_h, int scores_output_w);
	int get_match_keypoint(data_point & dp0, data_point & dp1,
		float * lopt_scores, int scores_output_h, int scores_output_w, std::vector<data_match> &vdm);
	void prepare_data(data_point & dp, float * arr1, float * arr2);

};

#endif // !_LOCAL_MATCH_H_
