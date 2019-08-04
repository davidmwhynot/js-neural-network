#!/bin/bash

node index.js 10 300 64 1 | tee /g/nnlog3/output.1.log & \
node index.js 10 400 64 1 | tee /g/nnlog3/output.2.log & \
node index.js 10 500 64 1 | tee /g/nnlog3/output.3.log

node index.js 10 300 128 1 | tee /g/nnlog3/output.4.log & \
node index.js 10 400 128 1 | tee /g/nnlog3/output.5.log & \
node index.js 10 500 128 1 | tee /g/nnlog3/output.6.log

node index.js 10 300 64 10 | tee /g/nnlog3/output.7.log & \
node index.js 10 400 64 10 | tee /g/nnlog3/output.8.log & \
node index.js 10 500 64 10 | tee /g/nnlog3/output.9.log

node index.js 10 300 128 10 | tee /g/nnlog3/output.10.log & \
node index.js 10 400 128 10 | tee /g/nnlog3/output.11.log & \
node index.js 10 500 128 10 | tee /g/nnlog3/output.12.log
