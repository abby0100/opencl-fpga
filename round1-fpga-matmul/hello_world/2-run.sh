#!/bin/bash

	source env-fpga-openvino-2019r3.sh-20200614

	# run
	make clean
	make
	CL_CONTEXT_EMULATOR_DEVICE_INTELFPGA=1 ./bin/host

# usage
# ./2-run.sh
