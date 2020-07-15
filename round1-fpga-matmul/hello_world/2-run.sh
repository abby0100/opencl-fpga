#!/bin/bash

	source 0-env-fpga-openvino-2019r3.sh-20200614

	# run
	make clean
	make
	CL_CONTEXT_EMULATOR_DEVICE_INTELFPGA=1 ./bin/host

# usage
# ./1-compile.sh
# ./2-run.sh | tee my.log
