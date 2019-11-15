# mnist-cpp-inference
使用c++实现mnist的推断过程

说明：


权重使用tf官方的代码训练的，版本是1.14，代码在 ./tf_mnist 里

权重保存在npy格式的文件里，放在了 ./weights/ 里。

感谢[@rogersce大佬的cnpy](https://github.com/rogersce/cnpy)，可以使用c++读取npy格式，方便不少。

读取格式：

```cpp
cnpy::NpyArray arr = cnpy::npy_load("../weights/hidden1_biases.npy");

float *b1 = arr.data<float>(); //长度为 input_nodes * output_nodes
 
```

编译运行：

```
mkdir build && cd build
cmake ..
make 
./mnist
```