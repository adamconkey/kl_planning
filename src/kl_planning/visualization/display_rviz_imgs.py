#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image

from kl_planning.srv import DisplayImage, DisplayImageResponse


class RvizImageDisplay:
    def __init__(self):
        self.rate = rospy.Rate(1)
        self.img_msg = None
        self.img_pub = rospy.Publisher("/image", Image, queue_size=1)
        self.img_srv = rospy.Service("/display_image", DisplayImage, self._display_img)

    def run(self):
        rospy.loginfo("Ready to display images")
        while not rospy.is_shutdown():
            if self.img_msg:
                self.img_pub.publish(self.img_msg)
            self.rate.sleep()

    def _display_img(self, req):
        self.img_msg = req.image
        return DisplayImageResponse(success=True)


if __name__ == '__main__':
    rospy.init_node("display_rviz_imgs")
    display = RvizImageDisplay()
    try:
        display.run()
    except rospy.ROSInterruptException:
        pass
