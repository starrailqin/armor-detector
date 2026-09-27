// ============================================================
// detector_node.cpp
// 功能：YOLO 检测装甲板 + 传统视觉找灯条角点 + PnP 定位
// 输出：所有检测到的装甲板的 AimInfo
// ============================================================

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <aim_interfaces/msg/aim_info.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

#include <vector>
#include <string>
#include <algorithm>
#include <cmath>

// ============================================================
// 相机内参（任务给的）
// ============================================================
const double CAMERA_MATRIX_DATA[9] = {
    1462.3697, 0.0,       398.59394,
    0.0,       1469.68385, 110.68997,
    0.0,       0.0,        1.0
};

const double DIST_COEFFS_DATA[5] = {
    0.003518, -0.311778, -0.016581, 0.023682, 0.0000
};

// ============================================================
// 装甲板尺寸（任务给的，单位：毫米）
// ============================================================
const float ARMOR_WIDTH = 160.0;
const float ARMOR_HEIGHT = 80.0;

// ============================================================
// YOLO 参数
// ============================================================
const int YOLO_INPUT_SIZE = 640;
const int YOLO_NUM_CLASSES = 36;
const float CONF_THRESHOLD = 0.5;
const float NMS_THRESHOLD = 0.45;

// ============================================================
// 类别映射：36 类 → 9 类
// ============================================================

int mapClassId(int dataset_id) {
    int type_id = dataset_id % 9;
    // 哨兵输出 7
    if (type_id == 0) return 7;
    return type_id;
}
// ============================================================
// 检测结果
// ============================================================
struct Detection {
    cv::Rect bbox;
    float confidence;
    int class_id;
    int dataset_id;
};

// ============================================================
// 装甲板检测节点
// ============================================================
class ArmorDetector : public rclcpp::Node {
public:
    ArmorDetector() : Node("armor_detector"),
                      env_(ORT_LOGGING_LEVEL_WARNING, "armor_detector") {

        image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/sensor_img", 10,
            std::bind(&ArmorDetector::imageCallback, this, std::placeholders::_1));

        // 发布 AimInfo（可以有多个）
        aim_pub_ = this->create_publisher<aim_interfaces::msg::AimInfo>(
            "/aim_target", 10);

        vis_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
            "/armor_debug_image", 10);

        camera_matrix_ = cv::Mat(3, 3, CV_64F, (void*)CAMERA_MATRIX_DATA).clone();
        dist_coeffs_ = cv::Mat(1, 5, CV_64F, (void*)DIST_COEFFS_DATA).clone();

        std::string model_path = "/home/keyis123456/armor_ws/src/armor_detector/best.onnx";

        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options);

        Ort::AllocatorWithDefaultOptions allocator;
        input_name_ = session_->GetInputNameAllocated(0, allocator).get();
        output_name_ = session_->GetOutputNameAllocated(0, allocator).get();

        RCLCPP_INFO(this->get_logger(), "YOLO model loaded: %s", model_path.c_str());
    }

private:
    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg) {

        cv_bridge::CvImagePtr cv_ptr;
        try {
            cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
            return;
        }
        cv::Mat frame = cv_ptr->image;

        // ====================================================
        // 1. YOLO 推理
        // ====================================================
        std::vector<Detection> detections = yoloInference(frame);

        RCLCPP_INFO(this->get_logger(), "YOLO detected %zu armors", detections.size());

        // ====================================================
        // 2. 遍历所有检测结果，每个都做 PnP
        // ====================================================
        int detection_index = 0;

        for (const auto& det : detections) {
            // --------------------------------------------------
            // 2.1 找灯条角点
            // --------------------------------------------------
            std::vector<cv::Point2f> image_points = findLightBarCorners(frame, det.bbox);

            // 如果没找到灯条，用 bbox 兜底
            if (image_points.size() != 4) {
                RCLCPP_WARN(this->get_logger(), "Armor %d: light bar not found, using bbox",
                            detection_index);

                image_points.clear();
                image_points.push_back(cv::Point2f(det.bbox.x, det.bbox.y));
                image_points.push_back(cv::Point2f(det.bbox.x + det.bbox.width, det.bbox.y));
                image_points.push_back(cv::Point2f(det.bbox.x + det.bbox.width,
                                                    det.bbox.y + det.bbox.height));
                image_points.push_back(cv::Point2f(det.bbox.x, det.bbox.y + det.bbox.height));
            }

            // --------------------------------------------------
            // 2.2 PnP 解算
            // --------------------------------------------------
            std::vector<cv::Point3f> object_points;
            object_points.push_back(cv::Point3f(-ARMOR_WIDTH/2, -ARMOR_HEIGHT/2, 0));
            object_points.push_back(cv::Point3f( ARMOR_WIDTH/2, -ARMOR_HEIGHT/2, 0));
            object_points.push_back(cv::Point3f( ARMOR_WIDTH/2,  ARMOR_HEIGHT/2, 0));
            object_points.push_back(cv::Point3f(-ARMOR_WIDTH/2,  ARMOR_HEIGHT/2, 0));

            cv::Mat rvec, tvec;
            bool success = cv::solvePnP(object_points, image_points,
                                         camera_matrix_, dist_coeffs_,
                                         rvec, tvec, false, cv::SOLVEPNP_ITERATIVE);

            if (!success) {
                RCLCPP_WARN(this->get_logger(), "Armor %d: PnP failed", detection_index);
                continue;
            }

            int16_t coord_x = (int16_t)tvec.at<double>(0);
            int16_t coord_y = (int16_t)tvec.at<double>(1);
            int16_t coord_z = (int16_t)tvec.at<double>(2);
		
	// ====================================================
// 坐标变换：相机坐标系 → 机器人坐标系
// ====================================================

// 1. 把 PnP 的 tvec 转成 cv::Mat
double cam_x = tvec.at<double>(0);  // 单位：毫米
double cam_y = tvec.at<double>(1);
double cam_z = tvec.at<double>(2);

// 2. 转成米（任务给的外参是米）
cam_x /= 1000.0;
cam_y /= 1000.0;
cam_z /= 1000.0;

// 3. 旋转角（度 → 弧度）
double roll  = 0.0  * CV_PI / 180.0;
double pitch = 60.0 * CV_PI / 180.0;
double yaw   = 20.0 * CV_PI / 180.0;

// 4. 计算旋转矩阵 R = Rz(yaw) × Ry(pitch) × Rx(roll)
cv::Mat Rx = (cv::Mat_<double>(3, 3) <<
    1, 0, 0,
    0, cos(roll), -sin(roll),
    0, sin(roll), cos(roll));

cv::Mat Ry = (cv::Mat_<double>(3, 3) <<
    cos(pitch), 0, sin(pitch),
    0, 1, 0,
    -sin(pitch), 0, cos(pitch));

cv::Mat Rz = (cv::Mat_<double>(3, 3) <<
    cos(yaw), -sin(yaw), 0,
    sin(yaw), cos(yaw), 0,
    0, 0, 1);

cv::Mat R = Rz * Ry * Rx;

// 5. 平移向量（米）
cv::Mat t = (cv::Mat_<double>(3, 1) << 0.08, 0.0, 0.05);

// 6. 相机坐标（米）
cv::Mat cam_pos = (cv::Mat_<double>(3, 1) << cam_x, cam_y, cam_z);

// 7. 变换到机器人坐标系
cv::Mat robot_pos = R * cam_pos + t;

// 8. 转成毫米（任务要求输出 Int16，单位可能是毫米）
int16_t robot_x = (int16_t)(robot_pos.at<double>(0) * 1000);
int16_t robot_y = (int16_t)(robot_pos.at<double>(1) * 1000);
int16_t robot_z = (int16_t)(robot_pos.at<double>(2) * 1000);

// 9. 用机器人坐标替换相机坐标
coord_x = robot_x;
coord_y = robot_y;
coord_z = robot_z;
            // --------------------------------------------------
            // 2.3 发布这个装甲板的 AimInfo
            // --------------------------------------------------
            auto aim_msg = aim_interfaces::msg::AimInfo();
            aim_msg.coordinate = {coord_x, coord_y, coord_z};
            aim_msg.type = det.class_id;
            aim_pub_->publish(aim_msg);

            RCLCPP_INFO(this->get_logger(), "Armor %d: type=%d, pos=(%d, %d, %d) mm",
                        detection_index, det.class_id, coord_x, coord_y, coord_z);

            // --------------------------------------------------
            // 2.4 画框和角点
            // --------------------------------------------------
            cv::rectangle(frame, det.bbox, cv::Scalar(0, 255, 0), 2);

            std::string label = "Type: " + std::to_string(det.class_id) +
                                " (" + std::to_string((int)(det.confidence * 100)) + "%)";
            cv::putText(frame, label, cv::Point(det.bbox.x, det.bbox.y - 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);

            // 画角点
            for (size_t k = 0; k < image_points.size(); k++) {
                cv::circle(frame, image_points[k], 4, cv::Scalar(0, 0, 255), -1);
            }

            // 画距离
            std::string dist_label = "Z: " + std::to_string(coord_z) + "mm";
            cv::putText(frame, dist_label, cv::Point(det.bbox.x, det.bbox.y + det.bbox.height + 20),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 0), 1);

            detection_index++;
        }

        // ====================================================
        // 3. 发布可视化图像
        // ====================================================
        sensor_msgs::msg::Image::SharedPtr vis_msg =
            cv_bridge::CvImage(msg->header, "bgr8", frame).toImageMsg();
        vis_pub_->publish(*vis_msg);
    }

    // ========================================================
    // 在 bbox 区域里找灯条，提取四个角点
    // ========================================================
    std::vector<cv::Point2f> findLightBarCorners(const cv::Mat& frame, const cv::Rect& bbox) {
        std::vector<cv::Point2f> corners;

        // 1. 裁剪 bbox 区域（稍微扩大一点）
        int margin = 10;
        cv::Rect roi_rect(
            std::max(0, bbox.x - margin),
            std::max(0, bbox.y - margin),
            std::min(frame.cols - bbox.x + margin, bbox.width + 2 * margin),
            std::min(frame.rows - bbox.y + margin, bbox.height + 2 * margin)
        );
        cv::Mat roi = frame(roi_rect);

        // 2. 转灰度
        cv::Mat gray;
        cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY);

        // 3. 二值化
        cv::Mat binary;
        cv::threshold(gray, binary, 200, 255, cv::THRESH_BINARY);

        // 4. 找轮廓
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        // 5. 筛选灯条
        std::vector<cv::RotatedRect> light_bars;
        for (auto& contour : contours) {
            double area = cv::contourArea(contour);
            if (area < 5) continue;

            cv::RotatedRect rect = cv::minAreaRect(contour);
            float w = rect.size.width;
            float h = rect.size.height;
            float ratio = std::max(w, h) / std::min(w, h);

            if (ratio > 1.5 && ratio < 10) {
                light_bars.push_back(rect);
            }
        }

        // 6. 至少需要两个灯条
        if (light_bars.size() < 2) return corners;

        // 7. 按 x 坐标排序
        std::sort(light_bars.begin(), light_bars.end(),
                  [](const cv::RotatedRect& a, const cv::RotatedRect& b) {
                      return a.center.x < b.center.x;
                  });

        cv::RotatedRect left_bar = light_bars[0];
        cv::RotatedRect right_bar = light_bars[light_bars.size() - 1];

        // 8. 提取四个角点
        cv::Point2f left_pts[4], right_pts[4];
        left_bar.points(left_pts);
        right_bar.points(right_pts);

        cv::Point2f left_top = left_pts[0], left_bottom = left_pts[0];
        for (int i = 1; i < 4; i++) {
            if (left_pts[i].y < left_top.y) left_top = left_pts[i];
            if (left_pts[i].y > left_bottom.y) left_bottom = left_pts[i];
        }

        cv::Point2f right_top = right_pts[0], right_bottom = right_pts[0];
        for (int i = 1; i < 4; i++) {
            if (right_pts[i].y < right_top.y) right_top = right_pts[i];
            if (right_pts[i].y > right_bottom.y) right_bottom = right_pts[i];
        }

        corners.push_back(left_top + cv::Point2f(roi_rect.x, roi_rect.y));
        corners.push_back(right_top + cv::Point2f(roi_rect.x, roi_rect.y));
        corners.push_back(right_bottom + cv::Point2f(roi_rect.x, roi_rect.y));
        corners.push_back(left_bottom + cv::Point2f(roi_rect.x, roi_rect.y));

        return corners;
    }

    // ========================================================
    // YOLO 推理
    // ========================================================
    std::vector<Detection> yoloInference(const cv::Mat& frame) {
        std::vector<Detection> detections;

        int img_w = frame.cols;
        int img_h = frame.rows;
        float scale = std::min((float)YOLO_INPUT_SIZE / img_w,
                               (float)YOLO_INPUT_SIZE / img_h);
        int new_w = (int)(img_w * scale);
        int new_h = (int)(img_h * scale);
        int pad_x = (YOLO_INPUT_SIZE - new_w) / 2;
        int pad_y = (YOLO_INPUT_SIZE - new_h) / 2;

        cv::Mat resized;
        cv::resize(frame, resized, cv::Size(new_w, new_h));

        cv::Mat padded(YOLO_INPUT_SIZE, YOLO_INPUT_SIZE, CV_8UC3,
                       cv::Scalar(114, 114, 114));
        resized.copyTo(padded(cv::Rect(pad_x, pad_y, new_w, new_h)));

        cv::Mat blob;
        cv::dnn::blobFromImage(padded, blob, 1.0/255.0,
                                cv::Size(YOLO_INPUT_SIZE, YOLO_INPUT_SIZE),
                                cv::Scalar(), true, false);

        std::vector<int64_t> input_shape = {1, 3, YOLO_INPUT_SIZE, YOLO_INPUT_SIZE};
        size_t input_size = 1 * 3 * YOLO_INPUT_SIZE * YOLO_INPUT_SIZE;

        Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault);

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            mem_info, (float*)blob.data, input_size,
            input_shape.data(), input_shape.size());

        const char* input_names[] = {input_name_.c_str()};
        const char* output_names[] = {output_name_.c_str()};

        auto output_tensors = session_->Run(
            Ort::RunOptions{nullptr},
            input_names, &input_tensor, 1,
            output_names, 1);

        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
        int num_boxes = output_shape[2];

        std::vector<cv::Rect> boxes;
        std::vector<float> confidences;
        std::vector<int> class_ids;

        for (int i = 0; i < num_boxes; i++) {
            float cx = output_data[0 * num_boxes + i];
            float cy = output_data[1 * num_boxes + i];
            float w  = output_data[2 * num_boxes + i];
            float h  = output_data[3 * num_boxes + i];

            float max_score = 0;
            int max_id = -1;
            for (int c = 0; c < YOLO_NUM_CLASSES; c++) {
                float score = output_data[(4 + c) * num_boxes + i];
                if (score > max_score) {
                    max_score = score;
                    max_id = c;
                }
            }

            if (max_score < CONF_THRESHOLD) continue;

            float x = (cx - w / 2 - pad_x) / scale;
            float y = (cy - h / 2 - pad_y) / scale;
            float rw = w / scale;
            float rh = h / scale;

            boxes.push_back(cv::Rect((int)x, (int)y, (int)rw, (int)rh));
            confidences.push_back(max_score);
            class_ids.push_back(mapClassId(max_id));
        }

        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD, indices);

        for (int idx : indices) {
            Detection det;
            det.bbox = boxes[idx];
            det.confidence = confidences[idx];
            det.class_id = class_ids[idx];
            detections.push_back(det);
        }

        return detections;
    }

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Publisher<aim_interfaces::msg::AimInfo>::SharedPtr aim_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr vis_pub_;

    cv::Mat camera_matrix_;
    cv::Mat dist_coeffs_;

    Ort::Env env_;
    std::unique_ptr<Ort::Session> session_;
    std::string input_name_;
    std::string output_name_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ArmorDetector>());
    rclcpp::shutdown();
    return 0;
}
