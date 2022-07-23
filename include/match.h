/*
 * Time  : 2022/01/01
 * Github: https://github.com/Broad-sky/feature-detection-matching-algorithm
 * Author: pingsheng shen
 */

#ifndef _MATCH_H_
#define _MATCH_H_
#include"local_match.h"
#include"akaze_match.h"
//#include"surf_match.h"
#include"tools.h"

class IrInterface
{
public:
	static IrInterface * createIrInterface();
	virtual void initial_model() = 0;
	virtual int dplearning_method_forward(std::vector<data_image> &vdi, std::vector<data_match>&vdm) = 0;
	virtual int traditional_method_forward(std::vector<data_image> &vdi, std::vector<data_match>&vdm, std::string md_name) = 0;
};


class image_registration : public IrInterface
{
public:
	image_registration();
	~image_registration();

	void initial_model();

	int dplearning_method_forward(std::vector<data_image> &vdi, std::vector<data_match> & vdm);
	int traditional_method_forward(std::vector<data_image> &vdi, std::vector<data_match>&vdm, std::string md_name);


private:
	point_lm_trt * _plt;
	match_lm_trt * _mlt;

};

#endif // !_MATCH_H_
