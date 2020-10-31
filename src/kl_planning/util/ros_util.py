import cv2
from cv_bridge import CvBridge

import rospy
from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker, MarkerArray
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import TransformStamped


def rgb_to_msg(rgb_array):
    rgb_img = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    rgb_msg = CvBridge().cv2_to_imgmsg(rgb_img)
    return rgb_msg


def msg_to_rgb(img_msg):
    rgb_img = cv2.cvtColor(msg_to_img(img_msg), cv2.COLOR_BGR2RGB)
    return rgb_img


def img_to_msg(img_array):
    img_msg = CvBridge().cv2_to_imgmsg(img_array)
    return img_msg


def msg_to_img(img_msg):
    img = CvBridge().imgmsg_to_cv2(img_msg)
    return img


def get_marker_msg(obj_data, marker_id=0):
    marker = Marker()
    marker.id = marker_id
    if obj_data['type'] == 'cube':
        marker.type = Marker.CUBE
    elif obj_data['type'] == 'cylinder':
        marker.type = Marker.CYLINDER
    elif obj_data['type'] == 'sphere':
        marker.type = Marker.SPHERE
    elif obj_data['type'] == 'text':
        marker.type = Marker.TEXT_VIEW_FACING
        marker.text = obj_data['text']
    else:
        raise ValueError(f"Unknown marker type: {obj_data['type']}")
    marker.action = Marker.MODIFY
    marker.pose.position.x = obj_data['position'][0]
    marker.pose.position.y = obj_data['position'][1]
    marker.pose.position.z = obj_data['position'][2]
    marker.pose.orientation.x = obj_data['orientation'][0]
    marker.pose.orientation.y = obj_data['orientation'][1]
    marker.pose.orientation.z = obj_data['orientation'][2]
    marker.pose.orientation.w = obj_data['orientation'][3]
    marker.header.frame_id = obj_data['parent_frame']
    if 'length' in obj_data and 'width' in obj_data and 'height' in obj_data:
        marker.scale.x = obj_data['length']
        marker.scale.y = obj_data['width']
        marker.scale.z = obj_data['height']
    if 'color' in obj_data:
        marker.color.r = obj_data['color'][0]
        marker.color.g = obj_data['color'][1]
        marker.color.b = obj_data['color'][2]
        marker.color.a = obj_data['color'][3]
    return marker


def get_marker_array_msg(objects):
    """
    Takes object configs as dictionary specifying pose, color, dimensions, type, etc. 
    and creates a MarkerArray msg that can be published to rviz for visualization.
    """
    array = MarkerArray()
    for i, (obj_id, obj_data) in enumerate(objects.items()):
        marker = get_marker_msg(obj_data, i)
        array.markers.append(marker)
    return array
        

def get_joint_state_msg(joint_pos):
    joint_state_msg = JointState()
    # TODO joint names not yet in data, should do that
    joint_state_msg.name = [f'panda_joint{j}' for j in range(1, 8)]
    joint_state_msg.name += [f'panda_finger_joint{j}' for j in [1, 2]]
    joint_state_msg.position = joint_pos
    return joint_state_msg


def get_tf_msg(tf, idx):
    tf_msg = TFMessage()
    for child_frame, data in tf.items():
        tf_stamped = TransformStamped()
        tf_stamped.header.frame_id = data['parent_frame']
        tf_stamped.child_frame_id = child_frame
        tf_stamped.transform.translation.x = data['position'][idx][0]
        tf_stamped.transform.translation.y = data['position'][idx][1]
        tf_stamped.transform.translation.z = data['position'][idx][2]
        tf_stamped.transform.rotation.x = data['orientation'][idx][0]
        tf_stamped.transform.rotation.y = data['orientation'][idx][1]
        tf_stamped.transform.rotation.z = data['orientation'][idx][2]
        tf_stamped.transform.rotation.w = data['orientation'][idx][3]
        tf_msg.transforms.append(tf_stamped)
    return tf_msg


def publish_msg(msg, publisher):
    msg.header.stamp = rospy.Time.now()
    publisher.publish(msg)


def publish_tf_msg(msg, publisher):
    for tf_stamped in msg.transforms:
        tf_stamped.header.stamp = rospy.Time.now()
        publisher.publish(msg)


def publish_marker_msg(msg, publisher):
    for marker in msg.markers:
        marker.header.stamp = rospy.Time.now()
        publisher.publish(msg)
