#!/bin/bash

node index.js 5 200 64 1 | tee /g/nnlog2/output.10.log
echo "10"
node index.js 5 300 64 1 | tee /g/nnlog2/output.20.log
echo "20"
node index.js 5 400 64 1 | tee /g/nnlog2/output.30.log
echo "30"
node index.js 5 500 64 1 | tee /g/nnlog2/output.40.log
echo "40"
node index.js 5 200 32 1 | tee /g/nnlog2/output.50.log
echo "50"
node index.js 5 300 32 1 | tee /g/nnlog2/output.60.log
echo "60"
node index.js 5 400 32 1 | tee /g/nnlog2/output.70.log
echo "70"
node index.js 5 500 32 1 | tee /g/nnlog2/output.80.log
echo "80"
node index.js 5 200 32 1 | tee /g/nnlog2/output.90.log
echo "90"
node index.js 5 300 128 1 | tee /g/nnlog2/output.100.log
echo "100"
node index.js 5 400 128 1 | tee /g/nnlog2/output.110.log
echo "110"
node index.js 5 500 128 1 | tee /g/nnlog2/output.120.log
echo "120"
node index.js 5 200 64 10 | tee /g/nnlog2/output.130.log
echo "130"
node index.js 5 300 64 10 | tee /g/nnlog2/output.140.log
echo "140"
node index.js 5 400 64 10 | tee /g/nnlog2/output.150.log
echo "150"
node index.js 5 500 64 10 | tee /g/nnlog2/output.160.log
echo "160"
node index.js 5 200 32 10 | tee /g/nnlog2/output.170.log
echo "170"
node index.js 5 300 32 10 | tee /g/nnlog2/output.180.log
echo "180"
node index.js 5 400 32 10 | tee /g/nnlog2/output.190.log
echo "190"
node index.js 5 500 32 10 | tee /g/nnlog2/output.200.log
echo "200"
node index.js 5 200 32 10 | tee /g/nnlog2/output.210.log
echo "210"
node index.js 5 300 128 10 | tee /g/nnlog2/output.220.log
echo "220"
node index.js 5 400 128 10 | tee /g/nnlog2/output.230.log
echo "230"
node index.js 5 500 128 10 | tee /g/nnlog2/output.240.log
echo "240"
