#include <cstdio>
#include "cuda_runtime.h"
#include "opencv2/opencv.hpp"
#include "cuAprilTags.h"

void loadImageToCuAprilTagsInput(const std::string& imagePath, cuAprilTagsImageInput_t& inputImage) {
  // Load image using OpenCV
  cv::Mat imgGs = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
  if (imgGs.empty()) {
    throw std::runtime_error("Failed to load image!");
  }

  cv::Mat img;
  cv::cvtColor(imgGs, img, cv::COLOR_GRAY2RGB);

  // Check if conversion worked
  if (img.type() != CV_8UC3) {
    throw std::runtime_error("Failed to convert to RGB!");
  }



  cudaMalloc(&inputImage.dev_ptr, img.rows * img.cols * sizeof(uchar3) * 8);
  cudaError_t memcpyErr = cudaMemcpy(inputImage.dev_ptr, img.data, img.rows * img.cols*sizeof(uchar3)*8, cudaMemcpyHostToDevice); // skeptical
  std::cout << cudaGetErrorString(memcpyErr) << "\n";

  // cv::Mat outimg(img.rows, img.cols, CV_8UC3);
  // outimg.create(img.rows, img.cols, CV_8UC3);
  // cv::UMat
  unsigned char* data = new unsigned char[img.rows * img.cols * sizeof(uchar3) * 8];
  cudaMemcpy(data, inputImage.dev_ptr, img.rows * img.cols * sizeof(uchar3) * 8, cudaMemcpyDeviceToHost);
  cv::Mat outimg(img.rows, img.cols, CV_8UC3, data);
  cv::imwrite("out.jpg", outimg);


  inputImage.width = img.cols;
  inputImage.height = img.rows;
  inputImage.pitch = img.cols*sizeof(uchar3);


}

int main(int argc, char ** argv)
{
  (void) argc;
  (void) argv;

  printf("hello world apriltags package\n");

  cuAprilTagsImageInput_t inputImage;
  loadImageToCuAprilTagsInput("image2.jpg", inputImage);


  cuAprilTagsCameraIntrinsics_t intrinsics{
    .fx = 1991.333,
    .fy = 1982.145,
    .cx = 766.253,
    .cy = 652.362,
  };
  cuAprilTagsHandle detector = nullptr;
  cudaStream_t stream = {};

  const int error = nvCreateAprilTagsDetector(&detector, 1600, 1304, 4, cuAprilTagsFamily::NVAT_TAG36H11, &intrinsics, 0.1651);
  if (error != 0) {
    throw std::runtime_error("Failed to create cuAprilTags detector (error code " + std::to_string(error) + ")");
  }
  auto streamErr = cudaStreamCreate(&stream);
  if (streamErr != 0) {
    std::cout << cudaGetErrorString(streamErr);
  }

  uint32_t num_detections;
  std::vector<cuAprilTagsID_t> tags(10);
  std::cout << "iso begin\n";
  const int error2 = cuAprilTagsDetect(detector, &inputImage, tags.data(), &num_detections, 10, stream);
  for (auto tag: tags) {
    std::cout<<tag.id<<"\n";
  }
  std::cout << "iso end\n";
  if (error2 != 0) {
    throw std::runtime_error("Failed to run AprilTags detector (error code " + std::to_string(error2) + ")");
  }

  if (num_detections > 0) {
    std::cout << "first id found: " << tags[0].id << "\n";
    std::cout << tags[0].translation[0] << "," << tags[0].translation[1] << "," << tags[0].translation[2] << "\n";
  } else {
    std::cout << "no detections.\n";
  }

  cudaFree(inputImage.dev_ptr);
  cudaStreamDestroy(stream);
  cuAprilTagsDestroy(detector);

  return 0;
}
