/*
 * Time  : 2022/03/01
 * Github: https://github.com/Broad-sky/feature-detection-matching-algorithm
 * Author: pingsheng shen
 */

#include"match.h"
#include<chrono>


int main(int argc, char **argv)
{
	if (argc > 3)
	{
		std::string argv1 = std::string(argv[1]);
		std::string argv2 = std::string(argv[2]);
		std::string argv3 = std::string(argv[3]);

		if (std::string(argv[1]) != "--deeplearning"&& (std::string(argv[1]) != "--traditional")) return 0;
		std::vector<data_image> vdi;
		data_image di;
		cv::VideoCapture cap(0);
	        if (!cap.isOpened())
	        {
		   std::cout << "Failed to open camera." << std::endl;
		   return -1;
	        }
		if (argv2 == "--camera")
		{
			const char *ip = argv[3];
			int index = std::stoi(argv3);
			if (cap.isOpened())
			{
				cap>>di.srcimg0;
			}
		}
		if (argv2 == "--image-pair")
		{
			const char *image_path0 = argv[3]; // input image path
			const char *image_path1 = argv[4];  // save result image path
			di.srcimg0 = cv::imread(image_path0);
			di.srcimg1 = cv::imread(image_path1);
			if (di.srcimg0.empty() || di.srcimg0.empty())
			{
				std::cerr << "image is null, please check image path \n";
				return -1;
			}
		}
		vdi.emplace_back(di);
		cv::Mat out, out_gray;
		auto ir_ptr = std::unique_ptr<IrInterface>(IrInterface::createIrInterface());
		if (std::string(argv[1]) == "--deeplearning") ir_ptr->initial_model();
		std::vector<data_match> vdm;
		float cost_time;
		while (true)
		{
			if (argv2 == "--camera")
			{
				if (cap.isOpened())
				{
					cap >> vdi[0].srcimg1;
				}
				else
				{
					std::cout << "open camera error!" << "\n";
					break;
				}
			}
			lm_tools::image_color_convert(vdi);
			auto start = std::chrono::high_resolution_clock::now();
			if (argv1 == "--deeplearning")
			{
				int ret = ir_ptr->dplearning_method_forward(vdi, vdm);
				if (ret >= 4)
				{
					std::cout << "deep learning method detect successed!" << "\n";
				}

			}
			if (argv1 == "--traditional")
			{
				int ret = ir_ptr->traditional_method_forward(vdi, vdm, "akaze");
				if (ret >= 4)
				{
					std::cout << "traditional method detect successed!" << "\n";
				}
			}
			auto end = std::chrono::high_resolution_clock::now();
			cost_time = std::chrono::duration<float, std::milli>(end-start).count();
			std::cout<<"cost: "<<cost_time<<" ms"<<std::endl;
			{
				lm_tools::make_matching_plot_fast(vdi, vdm);
				vdm.clear();
				if (argv2 != "--image-pair")
				{
					std::string text =std::to_string((int)(1000.f / cost_time)) + " fps";
					cv::putText(vdi[0].out_match, text, cv::Point(0, 25), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
				}
				cv::imshow("match", vdi[0].out_match);
				char key = cv::waitKey(1);
				if (key=='n') vdi[0].srcimg1.copyTo(vdi[0].srcimg0);
				if (key == 'q') break;
				if (argv2 == "--image-pair") break;
			}

		}
		cap.release();
		return 0;
	}
	else
	{
		std::cerr << "-->arguments not right!" << std::endl;
		return -1;
	}
}