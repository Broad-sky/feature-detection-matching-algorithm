/*
 * Time  : 2022/01/01
 * Github: https://github.com/Broad-sky/feature-detection-matching-algorithm
 * Author: pingsheng shen
 */

#ifndef _TOOLS_H
#define _TOOLS_H
#include"opencv2/opencv.hpp"
#include"data_body.h"

namespace lm_tools {
	static inline float sigmoid(float x) {
		return static_cast<float>(1.f / (1.f + exp(-x)));
	}

	static inline bool within_bounds_2d(int64_t h, int64_t w, int64_t H, int64_t W) 
	{
		return h >= 0 && h < H && w >= 0 && w < W;
	}

	static void print_data_1dim(float*arr, int length)
	{
		for (int i = 0; i < length; i++)
		{
			std::cout << arr[i] << ",";
		}
		std::cout << "\n";
	}

	static void print_data_2dim(float*arr, int h, int w)
	{
		for (int i = 0; i < h; i++)
		{
			for (int j = 0; j < w; j++)
			{
				std::cout << arr[i*w + j] << ",";
				if ((j+1) % w == 0)
				{
					std::cout << "\n";
				}
			}
		}
	}


	static void make_matching_plot_fast(std::vector<data_image> &vdi, std::vector<data_match> & vdm)
	{

		int H0 = vdi[0].resized_srcimg0.rows;
		int W0 = vdi[0].resized_srcimg0.cols;
		int H1 = vdi[0].resized_srcimg1.rows;
		int W1 = vdi[0].resized_srcimg1.cols;
		int margin = 30;
		int H = std::max(H0, H1);
		int W = W0 + W1 + margin;

		cv::Mat out = cv::Mat(H, W, CV_8UC1);
		out = cv::Scalar(255);

		cv::Rect rect0 = cv::Rect(0, 0, W0, H0);
		vdi[0].resized_srcimg0.copyTo(out(rect0));
		cv::Rect rect1 = cv::Rect(W0 + margin, 0, W1, H1);
		vdi[0].resized_srcimg1.copyTo(out(rect1));

		cv::Mat out_c3;
		cv::cvtColor(out, out_c3, cv::COLOR_GRAY2BGR);

		if (vdm.size()>=5)
		{
			for (auto & vdm_t : vdm)
			{
				if (false)
				{
					int valid_keypoint_x0 = vdm_t.valid_keypoint_x0 * double(vdi[0].srcimg0.cols) / double(vdi[0].resized_srcimg0.cols);
					int valid_keypoint_y0 = vdm_t.valid_keypoint_y0 * double(vdi[0].srcimg0.rows) / double(vdi[0].resized_srcimg0.rows);
					int valid_keypoint_x1 = vdm_t.valid_keypoint_x1 * double(vdi[0].srcimg1.cols) / double(vdi[0].resized_srcimg1.cols);
					int valid_keypoint_y1 = vdm_t.valid_keypoint_y1 * double(vdi[0].srcimg1.rows) / double(vdi[0].resized_srcimg1.rows);
					cv::circle(vdi[0].srcimg0, cv::Point(valid_keypoint_x0, valid_keypoint_y0), 3, cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
					cv::circle(vdi[0].srcimg1, cv::Point(valid_keypoint_x1, valid_keypoint_y1), 5, cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
					cv::circle(vdi[0].srcimg0_gray, cv::Point(valid_keypoint_x0, valid_keypoint_y0), 3, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
					cv::circle(vdi[0].srcimg1_gray, cv::Point(valid_keypoint_x1, valid_keypoint_y1), 5, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
				}

				int x0 = vdm_t.valid_keypoint_x0;
				int y0 = vdm_t.valid_keypoint_y0;
				int x1 = vdm_t.valid_keypoint_x1;
				int y1 = vdm_t.valid_keypoint_y1;
				if (vdm_t.msscores0>0.6)
				{
					cv::line(out_c3, cv::Point(x0, y0), cv::Point(x1 + margin + W0, y1), cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
					cv::circle(out_c3, cv::Point(x0, y0), 2, cv::Scalar(0, 0, 255), -1, cv::LINE_AA);
					cv::circle(out_c3, cv::Point(x1 + margin + W0, y1), 2, cv::Scalar(0, 0, 255), -1, cv::LINE_AA);
				}
			}
		}
		vdi[0].out_match = out_c3;
	}
	
	static void image_color_convert(std::vector<data_image> &vdi)
	{
		if (vdi[0].srcimg1.channels() == 3)
		{
			cv::cvtColor(vdi[0].srcimg0, vdi[0].srcimg0_gray, cv::COLOR_BGR2GRAY);
			cv::cvtColor(vdi[0].srcimg1, vdi[0].srcimg1_gray, cv::COLOR_BGR2GRAY);
		}
		if (vdi[0].srcimg1.channels() == 1)
		{
			vdi[0].srcimg0_gray = vdi[0].srcimg0;
			cv::cvtColor(vdi[0].srcimg0, vdi[0].srcimg0, cv::COLOR_GRAY2BGR);
			vdi[0].srcimg1_gray = vdi[0].srcimg1;
			cv::cvtColor(vdi[0].srcimg1, vdi[0].srcimg1, cv::COLOR_GRAY2BGR);
		}
	}


}


#endif // !_TOOLS_H
