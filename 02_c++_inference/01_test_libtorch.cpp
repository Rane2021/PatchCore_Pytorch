#include "torch/script.h"
#include "torch/torch.h"
#include <iostream>
#include <memory>
#include<time.h>


int main() {
    torch::jit::script::Module module;
    try{
        // module = torch::jit::load("/media/tianru/Rane/CODE/04_huagong_proj/06_Pytorch_inference/models/resnet18_trace.pt");
        module = torch::jit::load("/media/tianru/Rane/CODE/04_huagong_proj/06_Pytorch_inference/results/patchcore/mvtec/bottle/patchcore_trace.pt");
    }
    catch(const c10::Error& e){
        std::cerr << "load model error!\n";
        return -1;
    }

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::ones({1, 3, 224, 224}, torch::kFloat));
    // torch::Tensor res = module.forward(inputs).toTensor();
    auto res = module.forward(inputs).toTensor();

    // std::cout << res.max() << "\n";
    std::cout << res << "\n";
    std::cout << "模型加载预测成功!\n";

    // run time test
    clock_t start,end;
    start=clock();
    std::cout << "开始测试运行时间...\n";
    for (int i = 0; i < 100; ++i)
    {
        res = module.forward(inputs).toTensor();
    }
    end=clock();
    std::cout<<"100次预测时间: "<<(double)(end-start)/CLOCKS_PER_SEC/10 << "s" <<std::endl;  // 10%参数时候，预测100次约 25s

}


