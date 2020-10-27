import pybullet
import pybullet_data


def show_frame(model_id, link_index=-1, axis_length=0.2, axis_width=5, client=pybullet):
    """
    Displays frame with colored RGB colored XYZ axes.
    """
    client.addUserDebugLine(lineFromXYZ=[0, 0, 0],
                            lineToXYZ=[axis_length, 0, 0],
                            lineColorRGB=[1, 0, 0],
                            lineWidth=axis_width,
                            parentObjectUniqueId=model_id,
                            parentLinkIndex=link_index)
    client.addUserDebugLine(lineFromXYZ=[0, 0, 0],
                            lineToXYZ=[0, axis_length, 0],
                            lineColorRGB=[0, 1, 0],
                            lineWidth=axis_width,
                            parentObjectUniqueId=model_id,
                            parentLinkIndex=link_index)
    client.addUserDebugLine(lineFromXYZ=[0, 0, 0],
                            lineToXYZ=[0, 0, axis_length],
                            lineColorRGB=[0, 0, 1],
                            lineWidth=axis_width,
                            parentObjectUniqueId=model_id,
                            parentLinkIndex=link_index)


def add_mesh(position, orientation, filename, shift=[0, 0, 0],
             scale=[1, 1, 1], with_frame=True, client=pybullet):
    """
    Adds mesh object to environment.
    """
    client.setAdditionalSearchPath(pybullet_data.getDataPath())
    shape_id = client.createVisualShape(shapeType=pybullet.GEOM_MESH,
                                        fileName=filename,
                                        rgbaColor=[1, 1, 1, 1],
                                        specularColor=[0.4, .4, 0],
                                        visualFramePosition=shift,
                                        meshScale=scale)
    mesh_id = client.createMultiBody(baseVisualShapeIndex=shape_id,
                                     basePosition=position,
                                     baseOrientation=orientation)
    if with_frame:
        show_frame(mesh_id, client=client)

    return mesh_id



def add_sphere(radius, origin=[0, 0, 0], client=pybullet):
    """
    Adds sphere collision object to Pybullet environment.
    
    Note both shape ID and object ID are returned, they are both useful for certain function
    calls to Pybullet but I think you typically only want the object ID.

    Args:
        radius (float): Radius of sphere
        origin (List): X,Y,Z position for sphere origin
    Returns:
        shape_id (int): ID associated with geometry collision shape
        obj_id (int): ID associated with object
    """
    shape_id = client.createCollisionShape(pybullet.GEOM_SPHERE, radius=radius)
    obj_id = client.createMultiBody(baseMass=0,
                                    baseCollisionShapeIndex=shape_id,
                                    basePosition=origin)
    return shape_id, obj_id
    


def load_urdf(urdf_path, origin=[0, 0, 0], client=pybullet):
    """
    Loads URDF into pybullet.

    Args:
        urdf_path (str): Absolute path to URDF file.
        origin (List): X,Y,Z origin point where robot base will be spawned.
    Returns:
        robot_id (int): ID associated with robot model
    """
    robot_id = client.loadURDF(urdf_path, origin, useFixedBase=True)
    return robot_id


def add_slider(label, min_val, max_val, start_val=0, client=pybullet):
    """
    Add debug slider to pybullet panel.
    
    Args:
        label (str): Label for slider.
        min_val (float): Minimum value slider can be set to.
        max_val (float): Maximum value slider can be set to.
        start_val (float): Initial value slider is set to.
    Returns:
        slider_id (int): ID associated with slider so you can retrieve values from it.
    """
    slider_id = client.addUserDebugParameter(label, min_val, max_val, start_val)
    return slider_id
