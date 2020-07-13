#!/bin/bash
source=$1
target=xy

	source env-fpga-openvino-2019r3.sh-20200614

	# remove
	rm $target

	# compile
	#g++ $source -o $target -lOpenCL

	AOCL_COMPILE_CONFIG=$(aocl compile-config)
	AOCL_LINK_CONFIG=$(aocl link-config)
	echo -e "AOCL_COMPILE_CONFIG:\t$AOCL_COMPILE_CONFIG"
	echo -e "AOCL_LINK_CONFIG:\t$AOCL_LINK_CONFIG"
	g++ $AOCL_COMPILE_CONFIG $source $AOCL_LINK_CONFIG -o $target

	# run
	./$target

# usage
# ./compile-and-run-fpga.sh main.cpp
