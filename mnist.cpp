#include "cnpy/cnpy.h"
#include "utils.hpp"
#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <complex>
#include <cstdlib>
#include <map>
#include <string>

void split_string(const std::string &s, std::vector<std::string> &v, const std::string &c)
{
    std::string::size_type pos1, pos2;
    pos2 = s.find(c);
    pos1 = 0;
    while (std::string::npos != pos2)
    {
        v.push_back(s.substr(pos1, pos2 - pos1));

        pos1 = pos2 + c.size();
        pos2 = s.find(c, pos1);
    }
    if (pos1 != s.length())
        v.push_back(s.substr(pos1));
}

template <class Type>
Type stringToNum(const std::string &str)
{
    std::istringstream iss(str);
    Type num;
    iss >> num;
    return num;
}

int main()
{
    cv::Mat srcImage = cv::imread("../123.png", 0);

    std::vector<float> img;
    mat2vector(srcImage, img);

    cnpy::NpyArray arr = cnpy::npy_load("../weights/hidden1_biases.npy");

    float *b1 = arr.data<float>();

    cnpy::NpyArray arr1 = cnpy::npy_load("../weights/hidden1_weights.npy");
    float *w1 = arr1.data<float>();

    std::vector<float> tmp = fc(img, 784, 128, w1, b1, true);

    arr = cnpy::npy_load("../weights/hidden2_biases");
    float *b2 = arr.data<float>();
    arr1 = cnpy::npy_load("../weights/hidden2_weights");
    float *w2 = arr1.data<float>();

    std::vector<float> tmp1 = fc(tmp, 128, 32, w2, b2, true);

    arr = cnpy::npy_load("../weights/softmax_linear_biases");
    float *b3 = arr.data<float>();
    arr1 = cnpy::npy_load("../weights/softmax_linear_weights");
    float *w3 = arr1.data<float>();

    // for (int i = 0; i < 320; i++) {
    //     std::cout << "==："<< w3[i] << std::endl;
    // }

    std::vector<float> tmp2 = fc(tmp1, 32, 10, w3, b3, false);

    std::vector<double> pred = softmax(tmp2);
    for (int i = 0; i < pred.size(); i++)
    {
        std::cout << "+++：" << pred[i] << std::endl;
    }
    std::cout << "result is number: " << std::max_element(pred.begin(), pred.end()) -pred.begin() << std::endl;
    return 0;
}
