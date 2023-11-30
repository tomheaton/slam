# SLAM

Note from Gonk:
* There's this new file ('ros2slam.py') which does the same thing (for now) as the slam.py. However 'slam.py' was written for ROS1 at first but I tried to install ubuntu 18 on my PC and couldn't so I got ubuntu 20.04 instead, and then I had some problems installing the EUFS sim with ROS1 and turns out they now recommend Ubuntu 20.04 and ROS2(galactic) hence I started using ROS2 and so I had to
rewrite the file.
* It also plays the bag file by itself - make sure to change the .bag filepath at the end of the python file


### NOTES(for stuff you might forget):

TODO: add python libraries needed

### Common errors:
* make sure to source ros (/opt/ros/<distro>/setup.*sh)
* eufs msgs not available - makes sure u ran the eufs/install/setup.*sh file
* make sure to change the .bag filepath at the end of the python file

#### Bagfiles
* Record
rosbag record -o <output_filename>

* Play
rosbag play <filename>
(Needs the ros nodes for the car)


#### How to run eufs sim

ROS1:
* roslaunch eufs_launcher eufs_launcher.play
ROS2:
* ros2 launch eufs_launcher eufs_launcher.launch.py


#### Questions:

- How should we represent the data?
    - Poses: Usually the poses are represented as a col vector and every 3 elements (since each pose has x,y,yaw) a new pose is represented - so it's a n*3 rows and 1 col vector
    - Landmarks: Should these have their own vector? One for each cone?
    - Covariance Matrix: I think it's usually represented as a symmetric matrix that has pose and landmark information, so a row would be, pose1_x,pose1_y,pose1_yaw, ..., poseN_x, poseN_y, poseN_yaw, landmark1_x, landmark1_y, ..., landmarkN_x, landmarkN_y
