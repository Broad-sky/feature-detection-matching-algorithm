/*
 * Time  : 2022/01/01
 * Github: https://github.com/Broad-sky/feature-detection-matching-algorithm
 * Author: pingsheng shen
 */

#include"match.h"


void image_registration::initial_model()
{
	_plt = new point_lm_trt();
	_plt->initial_point_model();

	_mlt = new match_lm_trt();
	_mlt->initial_match_model();
}


image_registration::image_registration()
{
}

image_registration::~image_registration()
{
	delete _plt;
	delete _mlt;
	std::cout << "~image_registration()" << "\n";
}

int image_registration::dplearning_method_forward(std::vector<data_image> &vdi, std::vector<data_match> & vdm)
{
	int ret_mlt1 = 0;
	if (!vdi[0].srcimg0_gray.empty() && !vdi[0].srcimg1_gray.empty())
	{
		data_point dp0, dp1;
		cv::resize(vdi[0].srcimg0_gray, vdi[0].resized_srcimg0, cv::Size(_plt->get_input_w(), _plt->get_input_h()));  // INTER_LINEAR
		int ret_plt0 = _plt->forward(vdi[0].resized_srcimg0, dp0);

		cv::resize(vdi[0].srcimg1_gray, vdi[0].resized_srcimg1, cv::Size(_plt->get_input_w(), _plt->get_input_h()));
		int ret_plt1 = _plt->forward(vdi[0].resized_srcimg1, dp1);
		//assert(dp0.status_code == 1);
		//assert(dp1.status_code == 1);

		if (dp1.keypoint_size>=10)
		{
			ret_mlt1 = _mlt->forward(dp0, dp1, vdm);
			return ret_mlt1;
		}
		return -1;
	}
	else
	{
		std::cerr << "image0 or image1 is invalid!" << std::endl;
		return -1;
	}
}


int image_registration::traditional_method_forward(std::vector<data_image> &vdi, std::vector<data_match>&vdm, std::string md_name)
{
	if (!vdi[0].srcimg0_gray.empty() && !vdi[0].srcimg1_gray.empty())
	{
		int ret = 0;
		cv::resize(vdi[0].srcimg0_gray, vdi[0].resized_srcimg0, cv::Size(320, 240));  // INTER_LINEAR
		cv::resize(vdi[0].srcimg1_gray, vdi[0].resized_srcimg1, cv::Size(320, 240));
		if (md_name=="akaze")
		{
			auto _am_ptr = std::unique_ptr<akaze_match>(new akaze_match(4.5f, 0.8f));
			ret = _am_ptr->akaze_forward(vdi[0].resized_srcimg0, vdi[0].resized_srcimg1, vdm);
			return ret;
		}
		else
		{
			std::cout << "traditional method akaze vdm size: " << vdm.size() << "\n";
			return ret;
		}

	}
	else
	{
		std::cerr << "image0 or image1 is invalid!" << std::endl;
		return -1;
	}
}

IrInterface * IrInterface::createIrInterface()
{
	return new image_registration();
}
