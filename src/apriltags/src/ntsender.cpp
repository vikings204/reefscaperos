#include "cuda_runtime.h"
#include "opencv2/opencv.hpp"
#include "cuAprilTags.h"


int main(int argc, char ** argv)
{
  (void) argc;
  (void) argv;

  cv::VideoCapture cap{0, cv::CAP_V4L2};
  cap.set(cv::CAP_PROP_FRAME_WIDTH, 1600);
  cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1304);
  cap.set(cv::CAP_PROP_AUTO_EXPOSURE, 0.5);
  cap.set(cv::CAP_PROP_EXPOSURE, 1);
  cap.set(cv::CAP_PROP_GAIN, 0);
  cv::Mat frame;
  cuAprilTagsImageInput_t imageInput;

  cuAprilTagsCameraIntrinsics_t intrinsics{
    1991.333,
    1982.145,
    766.253,
    652.362,
  };
  cuAprilTagsHandle detector = nullptr;
  cudaStream_t stream = {};

  const int error = nvCreateAprilTagsDetector(&detector, 1600, 1304, 4, cuAprilTagsFamily::NVAT_TAG36H11, &intrinsics, 0.1651);
  std::cout << "create error code: " << error << "\n";
  auto streamErr = cudaStreamCreate(&stream);
  if (streamErr != 0) {
    std::cout << cudaGetErrorString(streamErr);
  }

  uint32_t num_detections;
  std::vector<cuAprilTagsID_t> tags(10);

  const cudaError_t mallocErr = cudaMalloc(&imageInput.dev_ptr, 1304 * 1600 * sizeof(uchar3));
  std::cout << "malloc error: " << cudaGetErrorString(mallocErr) << "\n";

  while (true) {
    cap >> frame;
    cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
    std::cout << "capt\n";

    const cudaError_t memcpyErr = cudaMemcpy(imageInput.dev_ptr, frame.data, 1304 * 1600 * sizeof(uchar3), cudaMemcpyHostToDevice);
    std::cout << "memcpy error: " << cudaGetErrorString(memcpyErr) << "\n";
    imageInput.width = frame.cols;
    imageInput.height = frame.rows;
    imageInput.pitch = frame.cols*sizeof(uchar3);

    const int error2 = cuAprilTagsDetect(detector, &imageInput, tags.data(), &num_detections, tags.capacity(), stream);
    std::cout << "detect error code: " << error2 << "\n";

    bool quit = false;
    if (num_detections > 0) {
        for (auto t : tags) {
          if (t.id == 0) {
            continue;
          }
          std::cout << "id=" << t.id << " tx=" << t.translation[0] << " ty=" << t.translation[1] << " tz=" << t.translation[2] << "\n";

          if (t.id == 5) {
            quit = true;
          }
        }
      } else {
        std::cout << "no detections.\n";
      }

    if (quit) {
      break;
    }
   }

  cudaFree(imageInput.dev_ptr);
  cudaStreamDestroy(stream);
  cuAprilTagsDestroy(detector);

  return 0;
}
