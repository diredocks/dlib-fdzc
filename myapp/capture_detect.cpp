#include <dlib/gui_widgets.h> // 添加dlib窗口支持
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/opencv.h>
#include <filesystem>
#include <opencv2/opencv.hpp>
#define SAVE_INTV 600 // 每隔多少帧保存一张人脸图像
#define INTV 1        // 间隔多少帧做一次人脸识别
using namespace std;  // 使用标准命名空间

int main() {
  int count = 0;

  // 创建VideoCapture对象，参数0表示打开默认摄像头，也可以指定视频文件路径或摄像头索引号
  cv::VideoCapture cap(0);
  // 检查摄像头是否成功打开
  if (!cap.isOpened()) {
    cerr << "无法打开摄像头" << endl;
    return -1;
  }

  // 将识别到的人脸保存到一个目录下
  std::filesystem::create_directory("recognized_faces");
  int save_frame_count = 0;
  int face_img_index = 0;

  // 创建dlib图像窗口
  dlib::image_window win;

  // 初始化人脸检测器
  dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
  dlib::shape_predictor pose_model;
  dlib::deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;
  int detect_countdown = 0;

  // 主循环，持续捕获和显示视频帧
  while (true) {

    cv::Mat frame; // 创建一个Mat对象来存储当前帧

    // 从摄像头捕获一帧图像，运算符>>重载了视频帧捕获功能
    cap >> frame;
    save_frame_count++;

    // 检查捕获的帧是否为空,如果为空则退出循环
    if (frame.empty())
      break;

    // 将OpenCV Mat转换为dlib格式
    dlib::cv_image<dlib::bgr_pixel> dlib_img(frame);

    win.set_image(dlib_img);

    // 使用dlib窗口的点击事件检测代替waitKey
    if (win.is_closed()) { // 检查窗口是否被关闭
      break;
    }

    if (--detect_countdown <= 0) {
      detect_countdown = INTV; // 重置计数器

      // 识别一下人脸
      std::vector<dlib::rectangle> faces = detector(dlib_img);

      // 把特征点加入数组
      std::vector<dlib::full_object_detection> shapes;
      for (unsigned long i = 0; i < faces.size(); ++i)
        shapes.push_back(pose_model(dlib_img, faces[i]));

      // 清空显示窗口的内容，方便重新绘制矩形框
      win.clear_overlay();
      // 在dlib窗口中显示图像和人脸框
      win.add_overlay(faces, dlib::rgb_pixel(255, 0, 0));

      // 显示结果： 打印人脸数量，每个矩形框的四个点
      std::cout << "检测到 " << faces.size() << " 张人脸：\n";
      for (const auto &face : faces) {
        std::cout << "  左-上: (" << face.left() << ", " << face.top() << "), "
                  << "右-下: (" << face.right() << ", " << face.bottom()
                  << ")\n";

        // 保存人脸图像（每隔 SAVE_INTV 帧）
        if (save_frame_count % SAVE_INTV == 0) {
          save_frame_count = 0;
          // 计算裁剪区域，并裁剪人脸区域
          cv::Rect face_rect(
              std::max((int)face.left(), 0), std::max((int)face.top(), 0),
              std::min((int)face.width(), frame.cols - (int)face.left()),
              std::min((int)face.height(), frame.rows - (int)face.top()));

          if (face_rect.width > 0 && face_rect.height > 0) {
            cv::Mat face_img = frame(face_rect).clone(); // 克隆人脸区域
            std::string filename = "recognized_faces/face_" +
                                   std::to_string(face_img_index++) + ".jpg";
            cv::imwrite(filename, face_img);
            std::cout << "Saved face to " << filename << "\n";
          }
        }
      }
    }

    // 保留一个小的延迟
    dlib::sleep(30);
  }
  return 0; // 程序正常退出
}
