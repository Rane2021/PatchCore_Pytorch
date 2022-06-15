#include "torch/script.h"
#include "torch/torch.h"
#include <iostream>
#include <memory>
#include<time.h>
#include<opencv2/opencv.hpp>


/*
cam ip 123 图像截取区域
cam_123 = [1742, 186, 2372, 783]  :左上角点（1742, 186）； 右下角点：（2372, 783）
*/




int main() {
    torch::jit::script::Module module;
    try{
        // module = torch::jit::load("/media/tianru/Rane/CODE/04_huagong_proj/06_Pytorch_inference/models/resnet18_trace.pt");
        module = torch::jit::load("/media/tianru/Rane/CODE/04_huagong_proj/06_Pytorch_inference/results/patchcore/mvtec/bottle/patchcore_trace_0615.pt");
    }
    catch(const c10::Error& e){
        std::cerr << "load model error!\n";
        return -1;
    }

    // preprocee image
    // cv::Mat read_image = cv::imread("/media/tianru/Rane/CODE/04_huagong_proj/PatchCore_Dataset/bottle/test/broken/20220527093434_ps.png", cv::IMREAD_COLOR);  // py:2.5105  c++: 2.50047
    // cv::Mat read_image = cv::imread("/media/tianru/Rane/CODE/04_huagong_proj/PatchCore_Dataset/bottle/test/broken/20220525190426.png", cv::IMREAD_COLOR);  // py:0.964  c++: 0.936821
    // need crop
    cv::Mat read_image = cv::imread("/media/tianru/Rane/CODE/04_huagong_proj/PatchCore_Dataset/ori_data/cam_123_ps_0614/20220525190426.jpg", cv::IMREAD_COLOR);  // py:0.964  c++: 0.936821

    if (read_image.empty() || !read_image.data){
        std::cout << "read image fail" << std::endl;
        return -1;
    }
    // image crop
    cv::Rect crop_rect(1742, 186, 2372-1742, 783-186);
    read_image = read_image(crop_rect);

    cv::cvtColor(read_image, read_image, cv::COLOR_BGR2RGB);
    cv::resize(read_image, read_image, cv::Size(224, 224), 0, 0, cv::INTER_LINEAR);
    read_image.convertTo(read_image, CV_32FC3, 1.0 / 255.0);

    torch::Tensor tensor_image = torch::from_blob(read_image.data, {1, read_image.rows, read_image.cols, 3});
    tensor_image = tensor_image.permute({0, 3 ,1 ,2});
    tensor_image[0][0] = tensor_image[0][0].sub_(0.485).div_(0.229);
    tensor_image[0][1] = tensor_image[0][1].sub_(0.456).div_(0.224);
    tensor_image[0][2] = tensor_image[0][2].sub_(0.406).div_(0.225);

    std::vector<torch::jit::IValue> inputs;
    // inputs.push_back(torch::ones({1, 3, 224, 224}, torch::kFloat));
    inputs.push_back(tensor_image);

    // model pred    
    auto res = module.forward(inputs).toTensor();

    // std::cout << res.max() << "\n";
    std::cout << res << "\n";
    std::cout << "模型加载预测成功!\n";

    // run time test
    clock_t start,end;
    start=clock();
    std::cout << "开始测试运行时间...\n";
    // for (int i = 0; i < 100; ++i)
    // {
    //     res = module.forward(inputs).toTensor();
    // }
    end=clock();
    std::cout<<"100次预测时间: "<<(double)(end-start)/CLOCKS_PER_SEC/10 << "s" <<std::endl;  // 10%参数时候，预测100次约 25s

    return 0;
}
