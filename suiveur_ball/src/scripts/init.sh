#!/usr/bin/env bash

echo 'alias cm="cd ~/catkin_ws && catkin_make && source devel/setup.bash && cd"' >> ~/.bash_aliases
alias cls='clear'
. ~/.bash_aliases
echo "init aliases done !"
