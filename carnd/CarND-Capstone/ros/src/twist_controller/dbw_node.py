#!/usr/bin/env python

import rospy
from std_msgs.msg import Bool
from dbw_mkz_msgs.msg import ThrottleCmd, SteeringCmd, BrakeCmd, SteeringReport
from geometry_msgs.msg import TwistStamped
import math

from vehicle import Vehicle
from twist_controller import Controller

'''
You can build this node only after you have built (or partially built) the `waypoint_updater` node.

You will subscribe to `/twist_cmd` message which provides the proposed linear and angular velocities.
You can subscribe to any other message that you find important or refer to the document for list
of messages subscribed to by the reference implementation of this node.

One thing to keep in mind while building this node and the `twist_controller` class is the status
of `dbw_enabled`. While in the simulator, its enabled all the time, in the real car, that will
not be the case. This may cause your PID controller to accumulate error because the car could
temporarily be driven by a human instead of your controller.

We have provided two launch files with this node. Vehicle specific values (like vehicle_mass,
wheel_base) etc should not be altered in these files.

We have also provided some reference implementations for PID controller and other utility classes.
You are free to use them or build your own.

Once you have the proposed throttle, brake, and steer values, publish it on the various publishers
that we have created in the `__init__` function.

'''

class DBWNode(object):
    def __init__(self):
        rospy.init_node('dbw_node')
        
        self.loglevel = rospy.get_param('/loglevel', 3)
        self.control_update_frequency = rospy.get_param('~control_update_frequency', 50)
        self.vehicle = Vehicle(
            vehicle_mass = rospy.get_param('/vehicle_mass', 1736.35),
            fuel_capacity = rospy.get_param('/fuel_capacity', 13.5),
            brake_deadband = rospy.get_param('/brake_deadband', .1),
            decel_limit = rospy.get_param('/decel_limit', -8.5),
            accel_limit = rospy.get_param('/accel_limit', 1.),
            wheel_radius = rospy.get_param('/wheel_radius', 0.2413),
            wheel_base = rospy.get_param('/wheel_base', 2.8498),
            steering_ratio = rospy.get_param('/steer_ratio', 14.8),
            max_lateral_accel = rospy.get_param('/max_lat_accel', 3.),
            max_steering_angle = rospy.get_param('/max_steer_angle', 8.))

        if self.loglevel > 4:
            rospy.loginfo("Vehicle %s", self.vehicle)
        self.steer_pub = rospy.Publisher('/vehicle/steering_cmd', SteeringCmd, queue_size=1)
        self.throttle_pub = rospy.Publisher('/vehicle/throttle_cmd', ThrottleCmd, queue_size=1)
        self.brake_pub = rospy.Publisher('/vehicle/brake_cmd', BrakeCmd, queue_size=1)

        self.current_velocity = None
        self.current_angular_velocity = None
        self.enabled = False
        self.linear_velocity = None
        self.angular_velocity = None
        self.throttle = 0
        self.steering = 0
        self.brake = 0
        
        self.controller = Controller(self.vehicle)

        rospy.Subscriber('/vehicle/dbw_enabled', Bool, self.dbw_enabled_cb)
        rospy.Subscriber('/twist_cmd', TwistStamped, self.twist_cb)
        rospy.Subscriber('/current_velocity', TwistStamped, self.velocity_cb)
        
        self.loop()

    def loop(self):
        rate = rospy.Rate(self.control_update_frequency)
        while not rospy.is_shutdown():
            if self.enabled and not None in (self.current_velocity, self.linear_velocity, self.angular_velocity):
                self.throttle, self.brake, self.steering = self.controller.control( self.linear_velocity,
                                                                                    self.angular_velocity,
                                                                                    self.current_velocity)
                self.publish(self.throttle, self.brake, self.steering)
            rate.sleep()

    def publish(self, throttle, brake, steer):
        tcmd = ThrottleCmd()
        tcmd.enable = True
        tcmd.pedal_cmd_type = ThrottleCmd.CMD_PERCENT
        tcmd.pedal_cmd = throttle
        self.throttle_pub.publish(tcmd)

        scmd = SteeringCmd()
        scmd.enable = True
        scmd.steering_wheel_angle_cmd = steer
        self.steer_pub.publish(scmd)

        bcmd = BrakeCmd()
        bcmd.enable = True
        bcmd.pedal_cmd_type = BrakeCmd.CMD_TORQUE
        bcmd.pedal_cmd = brake
        self.brake_pub.publish(bcmd)

    def dbw_enabled_cb(self, msg):
        if self.enabled != msg:
            self.enabled = msg
            self.controller.reset()
            rospy.loginfo("DBW enabled: %s", self.enabled)
    
    def twist_cb(self, msg):
        self.linear_velocity = abs(msg.twist.linear.x)
        self.angular_velocity = msg.twist.angular.z
#        rospy.loginfo("Twist linear v: %s, angular v: %s", self.linear_velocity, self.angular_velocity)
    
    def velocity_cb(self, msg):
        self.current_velocity = msg.twist.linear.x
        self.current_angular_velocity = msg.twist.angular.z
#        rospy.loginfo("Current v: %s, angular v: %s", self.current_velocity, self.current_angular_velocity)
        
if __name__ == '__main__':
    DBWNode()