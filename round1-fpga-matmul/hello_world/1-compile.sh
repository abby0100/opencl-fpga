#!/bin/bash

	source env-fpga-openvino-2019r3.sh-20200614

	# compile
	aoc -march=emulator device/hello_world.cl -o bin/hello_world.aocx -board=a10gx


