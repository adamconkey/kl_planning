from visualization_msgs.msg import Marker, MarkerArray


def get_marker_array_msg(objects):
    """
    Takes object configs as dictionary specifying pose, color, dimensions, type, etc. 
    and creates a MarkerArray msg that can be published to rviz for visualization.
    """
    array = MarkerArray()
    for i, (obj_id, obj_data) in enumerate(objects.items()):
        marker = Marker()
        marker.id = i
        marker.type = obj_data['type']
        marker.action = Marker.MODIFY
        marker.pose.position.x = obj_data['position'][idx][0]
        marker.pose.position.y = obj_data['position'][idx][1]
        marker.pose.position.z = obj_data['position'][idx][2]
        marker.pose.orientation.x = obj_data['orientation'][idx][0]
        marker.pose.orientation.y = obj_data['orientation'][idx][1]
        marker.pose.orientation.z = obj_data['orientation'][idx][2]
        marker.pose.orientation.w = obj_data['orientation'][idx][3]
        marker.header.frame_id = obj_dta['parent_frame']
        marker.scale.x = obj_data['length']
        marker.scale.y = obj_data['width']
        marker.scale.z = obj_data['height']
        marker.color.r = obj_data['color'][0]
        marker.color.g = obj_data['color'][1]
        marker.color.b = obj_data['color'][2]
        marker.color.a = obj_data['color'][3]
        array.markers.append(marker)
    return array
        
