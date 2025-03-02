#include "cuda_runtime.h"
#include "opencv2/opencv.hpp"
#include "cuAprilTags.h"
#include <networktables/NetworkTableInstance.h>
#include <networktables/NetworkTable.h>
#include <networktables/DoubleArrayTopic.h>

int main(int argc, char ** argv)
{
    (void) argc;
    (void) argv;

    cv::VideoCapture cap{0, cv::CAP_V4L2};
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1600);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1304);
    cap.set(cv::CAP_PROP_AUTO_EXPOSURE, 0.5);
    cap.set(cv::CAP_PROP_EXPOSURE, 1);
    // cap.set(cv::CAP_PROP_GAIN, 0);
    cap.set(cv::CAP_PROP_GAIN, 4);
    cv::Mat frame;
    cuAprilTagsImageInput_t imageInput;

    auto inst = nt::NetworkTableInstance::GetDefault();
    auto table = inst.GetTable("datatable");
    inst.StartClient4("jetson");
    inst.SetServerTeam(204);  // where TEAM=190, 294, etc, or use inst.setServer("hostname") or similar
    inst.StartDSClient();
    auto nt_ids_pub = table->GetDoubleArrayTopic("ids").Publish();
    auto nt_tx_pub = table->GetDoubleArrayTopic("tx").Publish();
    auto nt_ty_pub = table->GetDoubleArrayTopic("ty").Publish();
    auto nt_tz_pub = table->GetDoubleArrayTopic("tz").Publish();
    std::vector<double> nt_ids(22);
    std::vector<double> nt_tx(22);
    std::vector<double> nt_ty(22);
    std::vector<double> nt_tz(22);
    nt_ids_pub.SetDefault(nt_ids);
    nt_tx_pub.SetDefault(nt_tx);
    nt_ty_pub.SetDefault(nt_ty);
    nt_tz_pub.SetDefault(nt_tz);

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
    std::vector<cuAprilTagsID_t> tags(22);

    const cudaError_t mallocErr = cudaMalloc(&imageInput.dev_ptr, 1304 * 1600 * sizeof(uchar3));
    std::cout << "malloc error: " << cudaGetErrorString(mallocErr) << "\n";

    // cap >> frame;
    // cv::imwrite("stupid.jpg", frame);
    // std::exit(0);

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

        nt_ids.clear();
        nt_tx.clear();
        nt_ty.clear();
        nt_tz.clear();

        if (num_detections > 0) {
            for (auto t : tags) {
                if (t.id == 0) {
                    continue;
                }
                std::cout << "id=" << t.id << " tx=" << t.translation[0] << " ty=" << t.translation[1] << " tz=" << t.translation[2] << " r0=" << t.orientation[0] << " r1=" << t.orientation[1] << " r2=" << t.orientation[2] << " r3=" << t.orientation[3] << " r4=" << t.orientation[4] << " r5=" << t.orientation[5] << " r6=" << t.orientation[6] << " r7=" << t.orientation[7] << " r8=" << t.orientation[8] << "\n";
                nt_ids[t.id] = t.id;
                nt_tx[t.id] = t.translation[0];
                nt_ty[t.id] = t.translation[1];
                nt_tz[t.id] = t.translation[2];
            }
        } else {
            std::cout << "no detections.\n";
        }

        std::cout << "pub\n";
        nt_ids_pub.Set(std::span(nt_ids));
        nt_tx_pub.Set(std::span(nt_tx));
        nt_ty_pub.Set(std::span(nt_ty));
        nt_tz_pub.Set(std::span(nt_tz));
        std::cout << "done pub\n";
    }

  cudaFree(imageInput.dev_ptr);
  cudaStreamDestroy(stream);
  cuAprilTagsDestroy(detector);

  return 0;
}
