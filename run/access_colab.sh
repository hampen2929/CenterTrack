#!/bin/bash

for i in `seq 0 12`
do
  echo "[$i]" ` date '+%y/%m/%d %H:%M:%S'` "connected."
  xdg-open https://colab.research.google.com/drive/1NZiRsNc7sIbYxI2N0cYhvEL8B-D6n2qN#scrollTo=Wbxay6u1FE5P
#   xdg-open https://www.google.com/
  sleep 3600
done

# sudo apt-get install xdg-utils
# chmod 755 access_colab.sh
# ./run/access_colab.sh
