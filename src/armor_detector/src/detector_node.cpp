#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <aim_interfaces/msg/aim_info.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

#include <vector>
#include <string>
#include <algorithm>

const double CAMERA_MATRIX_DATA[9] = {
    1462.3697, 0.0,       398.59394,
    0.0,       1469.68385, 110.68997,
    0.0,       0.0,        1.0
};

const double DIST_COEFFS_DATA[5] = {
    0.003518, -0.311778, -0.016581, 0.023682, 0.0000
};

const float ARMOR_WIDTH = 160.0;
const float ARMOR_HEIGHT = 80.0;


const int YOLO_INPUT_SIZE = 640;
const int YOLO_NUM_CLASSES = 36;   
const float CONF_THRESHOLD = 0.5; 
const float NMS_THRESHOLD = 0.45; 
int mapClassId(int dataset_id) {
    
    int type_id = dataset_id % 9;
    return type_id;
}


struct Detection {
    cv::Rect bbox;
    float confidence;
    int class_id;     
    int dataset_id;   
};

class ArmorDetector : public rclcpp::Node {
public:
    ArmorDetector() : Node("armor_detector"),
                      env_(ORT_LOGGING_LEVEL_WARNING, "armor_detector") {
        image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/sensor_img", 10,
            std::bind(&ArmorDetector::imageCallback, this, std::placeholders::_1));
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
        RCLCPP_INFO(this->get_logger(), "Input: %s", input_name_.c_str());
        RCLCPP_INFO(this->get_logger(), "Output: %s", output_name_.c_str());
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

      
        std::vector<Detection> detections = yoloInference(frame);

     
     
        int16_t coord_x = 0, coord_y = 0, coord_z = 0;
        int16_t armor_type = 0;

        if (!detections.empty()) {
            
            auto best = std::max_element(detections.begin(), detections.end(),
                [](const Detection& a, const Detection& b) {
                    return a.confidence < b.confidence;
                });

            armor_type = best->class_id;

           
            std::vector<cv::Point2f> image_points;
            image_points.push_back(cv::Point2f(best->bbox.x, best->bbox.y));                          // 左上
            image_points.push_back(cv::Point2f(best->bbox.x + best->bbox.width, best->bbox.y));       // 右上
            image_points.push_back(cv::Point2f(best->bbox.x + best->bbox.width,
                                                best->bbox.y + best->bbox.height));                    // 右下
            image_points.push_back(cv::Point2f(best->bbox.x, best->bbox.y + best->bbox.height));      // 左下

            std::vector<cv::Point3f> object_points;
            object_points.push_back(cv::Point3f(-ARMOR_WIDTH/2, -ARMOR_HEIGHT/2, 0));
            object_points.push_back(cv::Point3f( ARMOR_WIDTH/2, -ARMOR_HEIGHT/2, 0));
            object_points.push_back(cv::Point3f( ARMOR_WIDTH/2,  ARMOR_HEIGHT/2, 0));
            object_points.push_back(cv::Point3f(-ARMOR_WIDTH/2,  ARMOR_HEIGHT/2, 0));

            cv::Mat rvec, tvec;
            bool success = cv::solvePnP(object_points, image_points,
                                         camera_matrix_, dist_coeffs_,
                                         rvec, tvec, false, cv::SOLVEPNP_ITERATIVE);

            if (success) {
                coord_x = (int16_t)tvec.at<double>(0);
                coord_y = (int16_t)tvec.at<double>(1);
                coord_z = (int16_t)tvec.at<double>(2);

                RCLCPP_INFO(this->get_logger(), "PnP: x=%d, y=%d, z=%d, type=%d",
                            coord_x, coord_y, coord_z, armor_type);
            }

           
            cv::rectangle(frame, best->bbox, cv::Scalar(0, 255, 0), 2);
            std::string label = "Type: " + std::to_string(armor_type) +
                                " (" + std::to_string((int)(best->confidence * 100)) + "%)";
            cv::putText(frame, label, cv::Point(best->bbox.x, best->bbox.y - 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
        }

   
        auto aim_msg = aim_interfaces::msg::AimInfo();
        aim_msg.coordinate = {coord_x, coord_y, coord_z};
        aim_msg.type = armor_type;
        aim_pub_->publish(aim_msg);

      
        sensor_msgs::msg::Image::SharedPtr vis_msg =
            cv_bridge::CvImage(msg->header, "bgr8", frame).toImageMsg();
        vis_pub_->publish(*vis_msg);
    }

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
        // output_shape: [1, 40, 8400]
        int num_boxes = output_shape[2];   // 8400
        int num_channels = output_shape[1]; // 40
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

    // ONNX Runtime
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
