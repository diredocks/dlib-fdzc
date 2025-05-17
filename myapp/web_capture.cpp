#include <dlib/image_io.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/opencv.h>
#include <iostream>
#include <nadjieb/mjpeg_streamer.hpp>
#include <opencv2/opencv.hpp>
#include <string>

#define INTV 1       // 间隔多少帧做一次人脸识别
#define SAVE_INTV 60 //
using namespace std; // 使用标准命名空间
using MJPEGStreamer = nadjieb::MJPEGStreamer;

int main() {
  // 创建VideoCapture对象，参数0表示打开默认摄像头，也可以指定视频文件路径或摄像头索引号
  cv::VideoCapture cap(0);
  // 检查摄像头是否成功打开
  if (!cap.isOpened()) {
    cerr << "无法打开摄像头" << endl;
    return -1;
  }

  // Initilize streamer
  std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, 90};
  MJPEGStreamer streamer;
  streamer.start(8080);

  // 初始化人脸检测器
  dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
  dlib::shape_predictor pose_model;
  dlib::deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;

  // Initilize countdown
  int detect_countdown = 0;
  int save_countdown = 0;
  int count = 0;

  // 主循环，持续捕获和显示视频帧
  while (streamer.isRunning()) {
    cv::Mat frame; // 创建一个Mat对象来存储当前帧
    cap >> frame;  // 从摄像头捕获一帧图像，运算符>>重载了视频帧捕获功能

    // 检查捕获的帧是否为空,如果为空则退出循环
    if (frame.empty()) {
      std::cerr << "frame not grabbed\n";
      exit(EXIT_FAILURE);
    }

    // 将OpenCV Mat转换为dlib格式
    dlib::cv_image<dlib::bgr_pixel> dlib_img(frame);

    if (--detect_countdown <= 0) {
      detect_countdown = INTV; // 重置计数器

      // 识别一下人脸
      std::vector<dlib::rectangle> faces = detector(dlib_img);

      // 绘制人脸识别框
      for (const auto &face : faces) {
        cv::rectangle(frame, cv::Point(face.left(), face.top()),
                      cv::Point(face.right(), face.bottom()),
                      cv::Scalar(0, 255, 0), 2);

        auto shape = pose_model(dlib_img, face);
        dlib::matrix<dlib::bgr_pixel> face_chip;

        // 调用 extract_image_chip 提取对齐的人脸图像
        dlib::extract_image_chip(
            dlib_img, dlib::get_face_chip_details(shape, 150, 0.25), face_chip);

        if (--save_countdown <= 0) {
          save_countdown = SAVE_INTV;
          // 保存对齐后的人脸图像
          std::string filename =
              "aligned_faces/" + std::to_string(count++) + ".jpg";
          // 将 dlib 的 face_chip 转换为 OpenCV 的 Mat
          cv::Mat face_mat = dlib::toMat(face_chip);
          // 使用 OpenCV 保存图像
          cv::imwrite(filename, face_mat, params);
        }
      }
    }

    std::vector<uchar> buff_bgr;
    cv::imencode(".jpg", frame, buff_bgr, params);
    streamer.publish("/bgr", std::string(buff_bgr.begin(), buff_bgr.end()));

    // 保留一个小的延迟
    dlib::sleep(30);
  }

  streamer.stop();
  return 0; // 程序正常退出
}
