# called by systemd, file at /etc/systemd/system/vikings.service
source /home/team204/.rosdevenv
echo starting..
ros2 run apriltags ntsender
