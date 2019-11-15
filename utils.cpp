#include <string>
#include <vector>
#include <math.h>
#include <opencv/cv.h>
#include <iostream>

//created by wangchao()
using namespace cv;

/*Convert Mat to Vector
 * */
void mat2vector(cv::Mat img, std::vector<float> &res)
{
    for (int i = 0; i < img.rows; i++)
    {
        for (int j = 0; j < img.cols; j++)
        {
            res.push_back(img.at<uchar>(i, j) - 127.5);
        }
    }
}

/*Fully connected layer
 * */
std::vector<float>
fc(std::vector<float> feature_map, int input_nodes, int output_nodes, float *weight, float *bias, bool use_relu)
{
    /*weight: input_nodes *output_nodes
 */
    std::vector<float> res(output_nodes, 0);

    for (int i = 0; i < input_nodes; i++)
    {
        for (int j = 0; j < output_nodes; j++)
        {
            res[j] += feature_map[i] * weight[i * output_nodes + j];
        }
    }

    for (int i = 0; i < output_nodes; ++i)
    {
        res[i] += bias[i];
        if (use_relu)
        {
            if (res[i] < 0)
            {
                res[i] = 0;
            }
        }
    }
    return res;
}

/*softmax layer
 * */
std::vector<double> softmax(std::vector<float> logits)
{
    std::vector<double> res;
    float e = 0.0;
    // To prevent overflow, we subtract each element from the maximum value.
    double max_value = *std::max_element(logits.begin(), logits.end());
    for (int i = 0; i < logits.size(); i++)
    {
        e += exp(logits[i] - max_value);
    }

    for (int i = 0; i < logits.size(); i++)
    {
        res.push_back(exp(logits[i] - max_value) * 1.0 / e);
    }

    return res;
}
