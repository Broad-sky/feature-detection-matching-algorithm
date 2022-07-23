/*
 * Time  : 2022/01/01
 * Github: https://github.com/Broad-sky/feature-detection-matching-algorithm
 * Author: pingsheng shen
 */

#include"local_match.h"


#ifdef __cplusplus
#define PUT_IN_REGISTER
#else
#define PUT_IN_REGISTER register
#endif

#define CV_VERSION_ID CVAUX_STR(CV_MAJOR_VERSION) CVAUX_STR(CV_MINOR_VERSION) CVAUX_STR(CV_SUBMINOR_VERSION)
#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

static Logger gLogger;



bool point_lm_trt::build_model()
{
	initLibNvInferPlugins(&gLogger.getTRTLogger(), "");
	nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
	if (!runtime)
	{
		return false;
	}

	char* model_deser_buffer{ nullptr };
	const std::string engine_file_path(_point_lm_params.point_weight_file);
	std::ifstream ifs;
	int ser_length;
	ifs.open(engine_file_path.c_str(), std::ios::in | std::ios::binary);  
	if (ifs.is_open())
	{
		ifs.seekg(0, std::ios::end);  
		ser_length = ifs.tellg();  
		ifs.seekg(0, std::ios::beg);    
		model_deser_buffer = new char[ser_length];  
		ifs.read(model_deser_buffer, ser_length);    
		ifs.close();
	}
	else
	{
		return false;
	}

	_engine_ptr = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(model_deser_buffer, ser_length, nullptr), infer_deleter());
	if (!_engine_ptr)
	{
		return false;
	}
	else
	{
		std::cout << "load engine successed!" << std::endl;
	}
	delete[] model_deser_buffer;

	_context_ptr = std::shared_ptr<nvinfer1::IExecutionContext>(_engine_ptr->createExecutionContext(), infer_deleter());
	if (!_context_ptr)
	{
		return false;
	}
	return true;
}

bool point_lm_trt::initial_point_model()
{
	bool ret = build_model();
	if (!ret)
	{
		std::cout << "build point model is failed!" << std::endl;
	}
	return ret;
}

size_t point_lm_trt::forward(cv::Mat& srcimg, data_point& dp)
{
	float * blob(nullptr);
	blob = imnormalize(srcimg);
	if (blob==nullptr)
	{
		std::cout << "imnormalize error! " << std::endl;
		dp.status_code = 0;
		return -1;
	}

	int batch_size = _engine_ptr->getMaxBatchSize();
	int dummy_inputIndex = _engine_ptr->getBindingIndex(_point_lm_params.input_names[0].c_str());
	assert(_engine_ptr->getBindingDataType(dummy_inputIndex) == nvinfer1::DataType::kFLOAT);
	auto dummy_input_dims = _engine_ptr->getBindingDimensions(dummy_inputIndex);
	int dummy_input_size = 1;
	for (int i = 0; i < dummy_input_dims.nbDims; i++)
	{
		dummy_input_size *= dummy_input_dims.d[i];
	}
	const int scores_outputIndex = _engine_ptr->getBindingIndex(_point_lm_params.output_names[0].c_str());
	assert(_engine_ptr->getBindingDataType(scores_outputIndex) == nvinfer1::DataType::kFLOAT);
	int scores_size = 1;
	auto scores_dims = _engine_ptr->getBindingDimensions(scores_outputIndex);
	for (int i = 0; i < scores_dims.nbDims; i++)
	{
		scores_size *= scores_dims.d[i];
	}

	const int descriptors_outputIndex = _engine_ptr->getBindingIndex(_point_lm_params.output_names[1].c_str());
	assert(_engine_ptr->getBindingDataType(descriptors_outputIndex) == nvinfer1::DataType::kFLOAT);
	int descriptors_size = 1;
	auto descriptors_dims = _engine_ptr->getBindingDimensions(descriptors_outputIndex);
	for (int i = 0; i < descriptors_dims.nbDims; i++)
	{
		descriptors_size *= descriptors_dims.d[i];
	}
	float * scores_output = new float[scores_size];
	float * descriptors_output = new float[descriptors_size];

	void* buffers[3];
	assert(1 * 1 * _point_lm_params.input_h * _point_lm_params.input_w == dummy_input_size);
	CHECK(cudaMalloc(&buffers[dummy_inputIndex], dummy_input_size * sizeof(float)));
	CHECK(cudaMalloc(&buffers[scores_outputIndex], scores_size * sizeof(float)));
	CHECK(cudaMalloc(&buffers[descriptors_outputIndex], descriptors_size * sizeof(float)));

	cudaStream_t stream;
	CHECK(cudaStreamCreate(&stream));
	CHECK(cudaMemcpyAsync(buffers[dummy_inputIndex], blob, dummy_input_size * sizeof(float), cudaMemcpyHostToDevice, stream));
	bool status = _context_ptr->enqueue(batch_size, buffers, stream, nullptr);

	delete[] blob;
	blob = nullptr;

	if (!status)
	{
		std::cout << "execute ifer error! " << std::endl;
		dp.status_code = 0;
		return -1;
	}
	CHECK(cudaMemcpyAsync(scores_output, buffers[scores_outputIndex], scores_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
	CHECK(cudaMemcpyAsync(descriptors_output, buffers[descriptors_outputIndex], descriptors_size * sizeof(float), cudaMemcpyDeviceToHost, stream));

	cudaStreamDestroy(stream);
	CHECK(cudaFree(buffers[dummy_inputIndex]));
	CHECK(cudaFree(buffers[scores_outputIndex]));
	CHECK(cudaFree(buffers[descriptors_outputIndex]));
	int scores_h = _point_lm_params.input_h;
	int scores_w = _point_lm_params.input_w;
	int border = _point_lm_params.border;
	for (int y = 0; y < scores_h; y++)
	{
		for (int x = 0; x < scores_w; x++)
		{
			float score = scores_output[y*scores_w + x];
			if (score>_point_lm_params.scores_thresh 
				&& x>=border && x<(scores_w-border)&&y>=border&&y<(scores_h-border))
			{
				data dt;
				dt.x = x;  
				dt.y = y;  
				dt.s = score;
				dp.vdt.push_back(dt);
			}
		}
	}
	int vdt_size = dp.vdt.size();
	dp.keypoint_size = vdt_size;

	int desc_fea_c = descriptors_dims.d[1];
	int desc_fea_h = descriptors_dims.d[2];
	int desc_fea_w = descriptors_dims.d[3];

	float * desc_channel_sum_sqrts = new float[desc_fea_h*desc_fea_w];
	for (int dfh = 0; dfh < desc_fea_h; dfh++)
	{
		for (int dfw = 0; dfw < desc_fea_w; dfw++)
		{
			float desc_channel_sum_temp = 0.f;
			for (int dfc = 0; dfc < desc_fea_c; dfc++)
			{
				desc_channel_sum_temp += descriptors_output[dfc*desc_fea_w*desc_fea_h + dfh*desc_fea_w + dfw]*
					descriptors_output[dfc*desc_fea_w*desc_fea_h + dfh*desc_fea_w + dfw];
			}
			float desc_channel_sum_sqrt = std::sqrt(desc_channel_sum_temp);
			desc_channel_sum_sqrts[dfh*desc_fea_w + dfw] = desc_channel_sum_sqrt;
		}

	}
	
	for (int dfh = 0; dfh < desc_fea_h; dfh++)
	{
		for (int dfw = 0; dfw < desc_fea_w; dfw++)
		{
			for (int dfc = 0; dfc < desc_fea_c; dfc++)
			{
				descriptors_output[dfc*desc_fea_w*desc_fea_h + dfh*desc_fea_w + dfw] = 
					descriptors_output[dfc*desc_fea_w*desc_fea_h + dfh*desc_fea_w + dfw] / desc_channel_sum_sqrts[dfh*desc_fea_w + dfw];
			}
		}
	}
	int s = 8;
	float * descriptors_output_f = new float[desc_fea_c*vdt_size];
	float * descriptors_output_sqrt = new float[vdt_size];
	int count = 0;
	for (auto & _vdt : dp.vdt)
	{
		float ix = ((_vdt.x - s / 2 + 0.5) / (desc_fea_w*s - s / 2 - 0.5))*(desc_fea_w - 1);
		float iy = (_vdt.y - s / 2 + 0.5)/ (desc_fea_h*s - s / 2 - 0.5)*(desc_fea_h - 1);

		int ix_nw = std::floor(ix);
		int iy_nw = std::floor(iy);

		int ix_ne = ix_nw + 1;
		int iy_ne = iy_nw;

		int ix_sw = ix_nw;
		int iy_sw = iy_nw + 1;

		int ix_se = ix_nw + 1;
		int iy_se = iy_nw + 1;

		float nw = (ix_se - ix)    * (iy_se - iy);
		float ne = (ix - ix_sw) * (iy_sw - iy);
		float sw = (ix_ne - ix)    * (iy - iy_ne);
		float se = (ix - ix_nw) * (iy - iy_nw);

		float descriptors_channel_sum_l2 = 0.f;
		for (int dfc = 0; dfc < desc_fea_c; dfc++)
		{
			float res = 0.f;

			if (lm_tools::within_bounds_2d(iy_nw, ix_nw, desc_fea_h, desc_fea_w))
			{
				res += descriptors_output[dfc*desc_fea_h*desc_fea_w + iy_nw*desc_fea_w + ix_nw] * nw;
			}
			if (lm_tools::within_bounds_2d(iy_ne, ix_ne, desc_fea_h, desc_fea_w))
			{
				res += descriptors_output[dfc*desc_fea_h*desc_fea_w + iy_ne*desc_fea_w + ix_ne] * ne;
			}
			if (lm_tools::within_bounds_2d(iy_sw, ix_sw, desc_fea_h, desc_fea_w))
			{
				res += descriptors_output[dfc*desc_fea_h*desc_fea_w + iy_sw*desc_fea_w + ix_sw] * sw;
			}
			if (lm_tools::within_bounds_2d(iy_se, ix_se, desc_fea_h, desc_fea_w))
			{
				res += descriptors_output[dfc*desc_fea_h*desc_fea_w + iy_se*desc_fea_w + ix_se] * se;
			}
			descriptors_output_f[dfc*vdt_size + count] = res;
			descriptors_channel_sum_l2 += res*res;
		}
		descriptors_output_sqrt[count] = descriptors_channel_sum_l2;
		for (int64_t dfc = 0; dfc < desc_fea_c; dfc++)
		{
			descriptors_output_f[dfc*vdt_size + count] /= std::sqrt(descriptors_output_sqrt[count]);
		}
		count++;
	}

	delete[]scores_output;
	delete[]descriptors_output;
	delete[]descriptors_output_sqrt;
	scores_output = nullptr;
	descriptors_output = nullptr;
	descriptors_output_sqrt = nullptr;
	
	dp.descriptors = descriptors_output_f;
	dp.desc_h = desc_fea_c;
	dp.desc_w = vdt_size;
	dp.status_code = 1;
	return 1;
}

point_lm_trt::point_lm_trt()
{
	cudaSetDevice(0);
	_point_lm_params.input_names.push_back("dummy_input");
	_point_lm_params.output_names.push_back("scores");
	_point_lm_params.output_names.push_back("descriptors");
	_point_lm_params.dla_core = -1;
	_point_lm_params.int8 = false;
	_point_lm_params.fp16 = false;
	_point_lm_params.batch_size = 1;
	_point_lm_params.seria = false;
	_point_lm_params.point_weight_file = "../engines/point_model.bin";
	_point_lm_params.input_h = 240;
	_point_lm_params.input_w = 320;
	_point_lm_params.scores_thresh = 0.01;
	_point_lm_params.border = 4;
}

point_lm_trt::~point_lm_trt()
{
}

int point_lm_trt::get_input_h()
{
	return _point_lm_params.input_h;
}

int point_lm_trt::get_input_w()
{
	return _point_lm_params.input_w;
}

float * point_lm_trt::imnormalize(cv::Mat & img)
{
	int img_h = img.rows;
	int img_w = img.cols;
	float * blob = new float[img_h*img_w];
	for (int h = 0; h < img_h; h++)
	{
		for (int w = 0; w < img_w; w++)
		{
			blob[img_w*h + w] = ((float)img.at<uchar>(h, w)) / 255.f;
		}
	}
	return blob;
}

bool match_lm_trt::build_model()
{
	initLibNvInferPlugins(&gLogger.getTRTLogger(), "");
	nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
	if (!runtime)
	{
		return false;
	}

	char* model_deser_buffer{ nullptr };
	const std::string engine_file_path(_match_lm_params.match_weight_file);
	std::ifstream ifs;
	int ser_length;
	ifs.open(engine_file_path.c_str(), std::ios::in | std::ios::binary); 
	if (ifs.is_open())
	{
		ifs.seekg(0, std::ios::end);  
		ser_length = ifs.tellg(); 
		ifs.seekg(0, std::ios::beg);  
		model_deser_buffer = new char[ser_length]; 
		ifs.read(model_deser_buffer, ser_length);
		ifs.close();
	}
	else
	{
		return false;
	}

	_engine_ptr = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(model_deser_buffer, ser_length, nullptr), infer_deleter());
	if (!_engine_ptr)
	{
		std::cout << "load engine failed!" << std::endl;
		return false;
	}
	delete[] model_deser_buffer;

	_context_ptr = std::shared_ptr<nvinfer1::IExecutionContext>(_engine_ptr->createExecutionContext(), infer_deleter());
	if (!_context_ptr)
	{
		return false;
	}
	return true;
}

bool match_lm_trt::initial_match_model()
{
	bool ret = build_model();
	if (!ret)
	{
		std::cout << "build model is failed!" << std::endl;
	}
	return ret;
}

size_t match_lm_trt::forward(data_point & dp0, data_point & dp1, std::vector<data_match> &vdm)
{
	float * keypoint0 = new float[dp0.keypoint_size * 2];
	float * scores0 = new float[dp0.keypoint_size];
	float * descriptors0 = dp0.descriptors;
	float * keypoint1 = new float[dp1.keypoint_size * 2];
	float * scores1 = new float[dp1.keypoint_size];
	float * descriptors1 = dp1.descriptors;
	prepare_data(dp0, keypoint0, scores0);
	prepare_data(dp1, keypoint1, scores1);
	int batch_size = _engine_ptr->getMaxBatchSize();
	int keypoints0_inputIndex = _engine_ptr->getBindingIndex(_match_lm_params.input_names[0].c_str());
	assert(_engine_ptr->getBindingDataType(keypoints0_inputIndex) == nvinfer1::DataType::kFLOAT);
	int keypoints1_inputIndex = _engine_ptr->getBindingIndex(_match_lm_params.input_names[1].c_str());
	assert(_engine_ptr->getBindingDataType(keypoints1_inputIndex) == nvinfer1::DataType::kFLOAT);
	int descriptors0_inputIndex = _engine_ptr->getBindingIndex(_match_lm_params.input_names[2].c_str());
	assert(_engine_ptr->getBindingDataType(descriptors0_inputIndex) == nvinfer1::DataType::kFLOAT);
	int descriptors1_inputIndex = _engine_ptr->getBindingIndex(_match_lm_params.input_names[3].c_str());
	assert(_engine_ptr->getBindingDataType(descriptors1_inputIndex) == nvinfer1::DataType::kFLOAT);
	int scores0_inputIndex = _engine_ptr->getBindingIndex(_match_lm_params.input_names[4].c_str());
	assert(_engine_ptr->getBindingDataType(scores0_inputIndex) == nvinfer1::DataType::kFLOAT);
	int scores1_inputIndex = _engine_ptr->getBindingIndex(_match_lm_params.input_names[5].c_str());
	assert(_engine_ptr->getBindingDataType(scores1_inputIndex) == nvinfer1::DataType::kFLOAT);
	int scores_outputIndex = _engine_ptr->getBindingIndex(_match_lm_params.output_names[0].c_str());
	assert(_engine_ptr->getBindingDataType(scores_outputIndex) == nvinfer1::DataType::kFLOAT);
	void* buffers[7];
	CHECK(cudaMalloc(&buffers[keypoints0_inputIndex], dp0.keypoint_size * 2 * sizeof(float)));
	CHECK(cudaMalloc(&buffers[keypoints1_inputIndex], dp1.keypoint_size * 2 * sizeof(float)));
	CHECK(cudaMalloc(&buffers[descriptors0_inputIndex], dp0.keypoint_size * 256 * sizeof(float)));
	CHECK(cudaMalloc(&buffers[descriptors1_inputIndex], dp1.keypoint_size * 256 * sizeof(float)));
	CHECK(cudaMalloc(&buffers[scores0_inputIndex], dp0.keypoint_size * sizeof(float)));
	CHECK(cudaMalloc(&buffers[scores1_inputIndex], dp1.keypoint_size * sizeof(float)));
	CHECK(cudaMalloc(&buffers[scores_outputIndex], dp0.keypoint_size * dp1.keypoint_size * sizeof(float)));

	cudaStream_t stream;
	CHECK(cudaStreamCreate(&stream));
	CHECK(cudaMemcpyAsync(buffers[keypoints0_inputIndex], keypoint0, dp0.keypoint_size * 2 * sizeof(float), cudaMemcpyHostToDevice, stream));
	CHECK(cudaMemcpyAsync(buffers[keypoints1_inputIndex], keypoint1, dp1.keypoint_size * 2 * sizeof(float), cudaMemcpyHostToDevice, stream));
	CHECK(cudaMemcpyAsync(buffers[descriptors0_inputIndex], descriptors0, dp0.keypoint_size * 256 * sizeof(float), cudaMemcpyHostToDevice, stream));
	CHECK(cudaMemcpyAsync(buffers[descriptors1_inputIndex], descriptors1, dp1.keypoint_size * 256 * sizeof(float), cudaMemcpyHostToDevice, stream));
	CHECK(cudaMemcpyAsync(buffers[scores0_inputIndex], scores0, dp0.keypoint_size * sizeof(float), cudaMemcpyHostToDevice, stream));
	CHECK(cudaMemcpyAsync(buffers[scores1_inputIndex], scores1, dp1.keypoint_size * sizeof(float), cudaMemcpyHostToDevice, stream));
	_context_ptr->setBindingDimensions(keypoints0_inputIndex, nvinfer1::Dims2(dp0.keypoint_size, 2));
	_context_ptr->setBindingDimensions(keypoints1_inputIndex, nvinfer1::Dims2(dp1.keypoint_size, 2));

	_context_ptr->setBindingDimensions(descriptors0_inputIndex, nvinfer1::Dims2(256, dp0.keypoint_size));
	_context_ptr->setBindingDimensions(descriptors1_inputIndex, nvinfer1::Dims2(256, dp1.keypoint_size));
	
	nvinfer1::Dims ddi0, ddi1;
	ddi0.nbDims = 1;
	ddi0.d[0] = dp0.keypoint_size;
	ddi1.nbDims = 1;
	ddi1.d[0] = dp1.keypoint_size;

	_context_ptr->setBindingDimensions(scores0_inputIndex, ddi0);
	_context_ptr->setBindingDimensions(scores1_inputIndex, ddi1);
	bool status = _context_ptr->enqueue(batch_size, buffers, stream, nullptr);
	if (!status)
	{
		std::cout << "execute ifer error! " << std::endl;
		return 101;
	}
	float * scores_output = new float[dp0.keypoint_size*dp1.keypoint_size];
	CHECK(cudaMemcpyAsync(scores_output, buffers[scores_outputIndex], dp0.keypoint_size*dp1.keypoint_size * sizeof(float), cudaMemcpyDeviceToHost, stream));

	cudaStreamDestroy(stream);
	CHECK(cudaFree(buffers[keypoints0_inputIndex]));
	CHECK(cudaFree(buffers[keypoints1_inputIndex]));
	CHECK(cudaFree(buffers[descriptors0_inputIndex]));
	CHECK(cudaFree(buffers[descriptors1_inputIndex]));

	CHECK(cudaFree(buffers[scores0_inputIndex]));
	CHECK(cudaFree(buffers[scores1_inputIndex]));
	CHECK(cudaFree(buffers[scores_outputIndex]));

	int scores_output_h = dp0.keypoint_size;
	int scores_output_w = dp1.keypoint_size;
	float bin_score = 4.4124f;
	int iters = 20;
	float*lopt_scores{nullptr};
	lopt_scores = log_optimal_transport(scores_output, bin_score, iters, scores_output_h, scores_output_w);
	delete[]scores_output;
	scores_output = nullptr;
	int ret_gmk = get_match_keypoint(dp0, dp1, lopt_scores, scores_output_h, scores_output_w, vdm);
	delete[] keypoint0;
	delete[] scores0;
	delete[] descriptors0;
	delete[] keypoint1;
	delete[] scores1;
	delete[] descriptors1;
	delete[] lopt_scores;
	keypoint0 = nullptr;
	scores0 = nullptr;
	descriptors0 = nullptr;
	keypoint1 = nullptr;
	scores1 = nullptr;
	descriptors1 = nullptr;
	lopt_scores = nullptr;
	return ret_gmk;
}

int match_lm_trt::get_match_keypoint(data_point & dp0, data_point & dp1, 
	float * lopt_scores, int scores_output_h, int scores_output_w, std::vector<data_match> &vdm)
{
	std::vector<maxv_indices> vmi0, vmi1;
	for (int i = 0; i < scores_output_h + 1; i++)
	{
		float temp_max_value0 = lopt_scores[0];
		maxv_indices mi0;
		for (int j = 0; j < scores_output_w + 1; j++)
		{
			if ((j + 1) % (scores_output_w + 1) != 0 && i != scores_output_h)
			{
				float current_value = lopt_scores[i*(scores_output_w + 1) + j];
				if (temp_max_value0<current_value)
				{
					temp_max_value0 = current_value;
					mi0.indices = j;
				}
			}
		}
		if (i != scores_output_h)
		{
			mi0.max_value = temp_max_value0;
			vmi0.emplace_back(mi0);
		}
	}

	for (int i = 0; i < scores_output_w + 1; i++)
	{
		float temp_max_value1 = lopt_scores[0];
		maxv_indices mi1;
		for (int j = 0; j < scores_output_h + 1; j++)
		{
			if ((j + 1) % (scores_output_h + 1) != 0 && j != scores_output_h)
			{
				float current_value = lopt_scores[j*(scores_output_w + 1) + i];
				if (temp_max_value1<current_value)
				{
					temp_max_value1 = current_value;
					mi1.indices = j;
				}
			}
		}
		if (i != scores_output_w)
		{
			mi1.max_value = temp_max_value1;
			vmi1.emplace_back(mi1);
		}
	}
	int vmi1_size = vmi1.size();
	if (vmi0.size()>=10 && vmi1_size >= 10)
	{
		for (int i = 0; i < vmi0.size(); i++)
		{
			int vmi0_index = vmi0[i].indices;
			if (vmi1[vmi0_index].indices == i)
			{
				float temp_mscores0 = std::exp(vmi1[vmi0[i].indices].max_value);
				if (temp_mscores0>_match_lm_params.match_threshold)
				{
					data_match dm0;
					dm0.msscores0 = temp_mscores0;
					dm0.valid_keypoint_x0 = dp0.vdt[i].x;
					dm0.valid_keypoint_y0 = dp0.vdt[i].y;

					if (vmi0[i].indices>dp1.keypoint_size)
					{
						std::cout << "get valid keypoint need carefull" << std::endl;
					}
					dm0.valid_keypoint_x1 = dp1.vdt[vmi0[i].indices].x;
					dm0.valid_keypoint_y1 = dp1.vdt[vmi0[i].indices].y;
					vdm.emplace_back(dm0);
				}
			}
		}
		return vdm.size();
	}
	else
	{
		return 0;
	}
	

}


float * match_lm_trt::log_optimal_transport(float * scores, float bin_score, int iters, 
	int scores_output_h, int scores_output_w)
{
	float norm = -std::log(scores_output_h + scores_output_w);
	int socres_new_size = scores_output_h*scores_output_w + scores_output_h + scores_output_w + 1;
	float * scores_new = new float[socres_new_size];

	for (int i = 0; i < scores_output_h+1; i++)
	{
		for (int j = 0; j < scores_output_w+1; j++)
		{
			if ((j+1)% (scores_output_w+1)==0 || i==scores_output_h)
			{
				scores_new[i*(scores_output_w + 1)+j] = bin_score;
			}
			else
			{
				scores_new[i*(scores_output_w + 1) + j] = scores[i*scores_output_w + j];
			}
		}
	}

	float * log_mu = new float[scores_output_h + 1];
	float * log_nu = new float[scores_output_w + 1];
	for (int i = 0; i < scores_output_h + 1; i++)
	{
		if (i==scores_output_h){
			log_mu[i] = std::log(scores_output_w) + norm;
		}
		else{
			log_mu[i] = norm;
		}
	}

	for (int i = 0; i < scores_output_w + 1; i++)
	{
		if (i == scores_output_w){
			log_nu[i] = std::log(scores_output_h) + norm;
		}
		else{
			log_nu[i] = norm;
		}
	}

	float * v = new float[scores_output_w + 1];
	float * u = new float[scores_output_h + 1];
	memset(v, 0.f, (scores_output_w + 1) * sizeof(float));
	memset(u, 0.f, (scores_output_h + 1) * sizeof(float));

	for (int iter = 0; iter < iters; iter++)
	{
		for (int i = 0; i < scores_output_h+1; i++)
		{
			float zv_sum_exp = 0.f;
			for (int j = 0; j < scores_output_w+1; j++)
			{
				zv_sum_exp += std::exp(scores_new[i*(scores_output_w+1) + j] + v[j]);
			}
			u[i] = log_mu[i] - std::log(zv_sum_exp);
		}
		for (int i = 0; i < scores_output_w + 1; i++)
		{
			float zu_sum_exp = 0.f;
			for (int j = 0; j < scores_output_h + 1; j++)
			{
				zu_sum_exp += std::exp(scores_new[j*(scores_output_w+1) + i] + u[j]);
			}
			v[i] = log_nu[i] - std::log(zu_sum_exp);
		}
	}

	for (int i = 0; i < scores_output_h + 1; i++)
	{
		for (int j = 0; j < scores_output_w + 1; j++)
		{
			scores_new[i*(scores_output_w+1) + j] = scores_new[i*(scores_output_w+1) + j] + u[i] + v[j]-norm;
		}
	}
	delete[] log_mu;
	delete[] log_nu;
	delete[] u;
	delete[] v;
	log_mu = nullptr;
	log_nu = nullptr;
	u = nullptr;
	v = nullptr;

	return scores_new;
}

void match_lm_trt::prepare_data(data_point & dp, float * arr1, float * arr2)
{
	int count_k = 0;
	int count_s = 0;
	for (auto & dpv : dp.vdt)
	{
		arr1[count_k] = dpv.x;
		arr1[count_k + 1] = dpv.y;
		count_k += 2;

		arr2[count_s] = dpv.s;
		count_s++;
	}
}


match_lm_trt::match_lm_trt()
{
	cudaSetDevice(0);
	_match_lm_params.input_names.push_back("keypoints0");
	_match_lm_params.input_names.push_back("keypoints1");
	_match_lm_params.input_names.push_back("descriptors0");
	_match_lm_params.input_names.push_back("descriptors1");
	_match_lm_params.input_names.push_back("scores0");
	_match_lm_params.input_names.push_back("scores1");

	_match_lm_params.output_names.push_back("scores");
	_match_lm_params.dla_core = -1;
	_match_lm_params.int8 = false;
	_match_lm_params.fp16 = false;
	_match_lm_params.batch_size = 1;
	_match_lm_params.seria = false;
	_match_lm_params.match_weight_file = "../engines/match_model_indoor.bin";
	_match_lm_params.match_threshold = 0.2f;

}

match_lm_trt::~match_lm_trt()
{
}
