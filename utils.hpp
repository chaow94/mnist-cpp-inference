#include <string>
#include <vector>
#include <opencv/cv.h>
//created by wangchao()

/*Convert Mat to Vector
 * */
void mat2vector(cv::Mat img, std::vector<float> &res);

/*Fully connected layer
 * */
std::vector<float> fc(std::vector<float> feature_map, int input_nodes, int output_nodes, float *weight, float *bias, bool use_relu);
std::vector<double> softmax(std::vector<float> feature_map);
