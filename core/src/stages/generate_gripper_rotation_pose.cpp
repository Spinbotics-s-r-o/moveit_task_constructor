#include <moveit/task_constructor/stages/generate_gripper_rotation_pose.h>
#include <moveit/task_constructor/storage.h>
#include <moveit/task_constructor/marker_tools.h>
#include <rviz_marker_tools/marker_creation.h>

#include <moveit/planning_scene/planning_scene.h>
#include <moveit/robot_state/conversions.h>

#include <Eigen/Geometry>
#if __has_include(<tf2_eigen/tf2_eigen.hpp>)
#include <tf2_eigen/tf2_eigen.hpp>
#else
#include <tf2_eigen/tf2_eigen.h>
#endif

namespace moveit {
namespace task_constructor {
namespace stages {

static const rclcpp::Logger LOGGER = rclcpp::get_logger("GenerateGripperRotationPose");

GenerateGripperRotationPose::GenerateGripperRotationPose(const std::string& name) : GeneratePose(name) {
	auto& p = properties();
	p.declare<std::string>("eef", "name of end-effector");
	p.declare<std::string>("object");
	p.declare<double>("angle_delta", 0.1, "angular steps (rad)");

	p.declare<boost::any>("pregrasp", "pregrasp posture");
	p.declare<boost::any>("grasp", "grasp posture");
}

static void applyPreGrasp(moveit::core::RobotState& state, const moveit::core::JointModelGroup* jmg,
                          const Property& diff_property) {
	try {
		// try named joint pose
		const std::string& diff_state_name{ boost::any_cast<std::string>(diff_property.value()) };
		if (!state.setToDefaultValues(jmg, diff_state_name)) {
			throw moveit::Exception{ "unknown state '" + diff_state_name + "'" };
		}
		return;
	} catch (const boost::bad_any_cast&) {
	}

	try {
		// try RobotState
		const moveit_msgs::msg::RobotState& robot_state_msg =
		    boost::any_cast<moveit_msgs::msg::RobotState>(diff_property.value());
		if (!robot_state_msg.is_diff)
			throw moveit::Exception{ "RobotState message must be a diff" };
		const auto& accepted = jmg->getJointModelNames();
		for (const auto& joint_name_list :
		     { robot_state_msg.joint_state.name, robot_state_msg.multi_dof_joint_state.joint_names })
			for (const auto& name : joint_name_list)
				if (std::find(accepted.cbegin(), accepted.cend(), name) == accepted.cend())
					throw moveit::Exception("joint '" + name + "' is not part of group '" + jmg->getName() + "'");
		robotStateMsgToRobotState(robot_state_msg, state);
		return;
	} catch (const boost::bad_any_cast&) {
	}

	throw moveit::Exception{ "no named pose or RobotState message" };
}

void GenerateGripperRotationPose::init(const core::RobotModelConstPtr& robot_model) {
	InitStageException errors;
	try {
		GeneratePose::init(robot_model);
	} catch (InitStageException& e) {
		errors.append(e);
	}

	const auto& props = properties();

	// check angle_delta
	if (props.get<double>("angle_delta") == 0.)
		errors.push_back(*this, "angle_delta must be non-zero");

	if (errors)
		throw errors;
}

void GenerateGripperRotationPose::onNewSolution(const SolutionBase& s) {
	upstream_solutions_.push(&s);
}

void GenerateGripperRotationPose::compute() {
	if (upstream_solutions_.empty())
		return;
	planning_scene::PlanningScenePtr scene = upstream_solutions_.pop()->end()->scene()->diff();

	// set end effector pose
	const auto& props = properties();
	geometry_msgs::msg::PoseStamped target_pose_msg;
	//target_pose_msg.header.frame_id = scene->getPlanningFrame();
	//target_pose_msg.header.frame_id = "tcp";
	target_pose_msg.header.frame_id = "base_link";
	//target_pose_msg.pose = properties().get<geometry_msgs::msg::PoseStamped>("pose").pose;

	geometry_msgs::msg::PoseArray gen_poses_vis;
	gen_poses_vis.header = target_pose_msg.header;

	double current_angle = 0.0;
	while (current_angle < 2. * M_PI && current_angle > -2. * M_PI) {
		target_pose_msg.pose = properties().get<geometry_msgs::msg::PoseStamped>("pose").pose;
		
		tf2::Quaternion q;
		q.setW(target_pose_msg.pose.orientation.w);
		q.setX(target_pose_msg.pose.orientation.x);
		q.setY(target_pose_msg.pose.orientation.y);
		q.setZ(target_pose_msg.pose.orientation.z);
		
		tf2::Quaternion rot;
		rot.setRPY(0,0,current_angle);

		q = rot*q;

		target_pose_msg.pose.orientation.x = q.getX();
		target_pose_msg.pose.orientation.y = q.getY();
		target_pose_msg.pose.orientation.z = q.getZ();
		target_pose_msg.pose.orientation.w = q.getW();

		InterfaceState state(scene);
		state.properties().set("target_pose", target_pose_msg);

		SubTrajectory trajectory;
		trajectory.setCost(0.0);
		trajectory.setComment(std::to_string(current_angle));

		current_angle += props.get<double>("angle_delta");

		// add frame at target pose
		rviz_marker_tools::appendFrame(trajectory.markers(), target_pose_msg, 0.1, "grasp frame");

		spawn(std::move(state), std::move(trajectory));
	}

}
}  // namespace stages
}  // namespace task_constructor
}  // namespace moveit
