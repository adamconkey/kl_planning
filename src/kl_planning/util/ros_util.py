from visualization_msgs.msg import Marker, MarkerArray


def get_marker_msg(obj_data, marker_id=0):
    marker = Marker()
    marker.id = marker_id
    if obj_data['type'] == 'cube':
        marker.type = Marker.CUBE
    elif obj_data['type'] == 'cylinder':
        marker.type = Marker.CYLINDER
    marker.action = Marker.MODIFY
    marker.pose.position.x = obj_data['position'][0]
    marker.pose.position.y = obj_data['position'][1]
    marker.pose.position.z = obj_data['position'][2]
    marker.pose.orientation.x = obj_data['orientation'][0]
    marker.pose.orientation.y = obj_data['orientation'][1]
    marker.pose.orientation.z = obj_data['orientation'][2]
    marker.pose.orientation.w = obj_data['orientation'][3]
    marker.header.frame_id = obj_data['parent_frame']
    marker.scale.x = obj_data['length']
    marker.scale.y = obj_data['width']
    marker.scale.z = obj_data['height']
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
        
