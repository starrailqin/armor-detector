#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <aim_interfaces/msg/aim_info.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>


class ArmorDetector : public rclcpp::Node {
public:
    ArmorDetector() : Node("armor_detector") {
        image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/sensor_img",
            10,
            std::bind(&ArmorDetector::imageCallback, this,
                      std::placeholders::_1));
	aim_pub_=this->create_publisher<aim_interfaces::msg::AimInfo>(
		"/aim_target",
		10);
	vis_pub_=this->create_publisher<sensor_msgs::msg::Image>(
		"/armor_debug_image",
		10
	);

	

        RCLCPP_INFO(this->get_logger(), "Armor detector started");
    }

private:
    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
	cv_bridge::CvImagePtr cv_ptr;
	try{
		cv_ptr=cv_bridge::toCvCopy(msg,"bgr8");
	}catch(cv_bridge::Exception& e){
		RCLCPP_ERROR(this->get_logger(),"cv_bridge exception: %s", e.what());
		return ;
		
		}
	cv::Mat frame=cv_ptr->image;
        RCLCPP_INFO(this->get_logger(), "Received image: %d x %d",
                    frame.cols, frame.rows);
    
	auto aim_msg = aim_interfaces::msg::AimInfo();
	aim_msg.coordinate={100,200,300};
	aim_msg.type=7;
	aim_pub_->publish(aim_msg);
	
	cv::rectangle(frame,cv::Point(100,100),cv::Point(500,400),cv::Scalar(0,255,0),3);
	std::string label="type:"+std::to_string(aim_msg.type);
	cv::putText(frame,label,cv::Point(100,90),cv::FONT_HERSHEY_SIMPLEX,1.0,cv::Scalar(0,255,0),2);
	sensor_msgs::msg::Image::SharedPtr vis_msg=cv_bridge::CvImage(msg->header,"bgr8",frame).toImageMsg();
	vis_pub_->publish(*vis_msg);
	

	RCLCPP_INFO(this->get_logger(),"Published AimInfo:coord=[%d,%d,%d],type=%d",
		aim_msg.coordinate[0],aim_msg.coordinate[1],
		aim_msg.coordinate[2],aim_msg.type
		);
	}
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Publisher<aim_interfaces::msg::AimInfo>::SharedPtr aim_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr vis_pub_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ArmorDetector>());
    rclcpp::shutdown();
    return 0;
}
