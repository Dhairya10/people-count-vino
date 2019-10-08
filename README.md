# people-count-vino
This project uses Python and OpenVINO toolkit for counting the number of people in a given frame. <br>
OpenVINO toolkit is used for faster inference.<br>
RMNet is used here as a backbone model. <br>
It is a combination of ResNet and MobileNet. 

### OpemVINO Toolkit Setup 

#### Installation Guide <br>
https://software.intel.com/en-us/openvino-toolkit/choose-download

#### Getting Started Guide <br>
https://software.intel.com/en-us/openvino-toolkit/documentation/get-started


### Directory Structure 

```
.
├── inference.py
├── output_snapshots
├── people_count.py
└── resources
    └── config.json

2 directories, 3 files

```
`people_count.py` - It is the python program which counts the number of people in a given frame. <br>
`config.json` - It is a json file which determines the source of the video. You can specify the url to a video file or your webcam id.

### Executing the Program 

#### Step 1 - 
Run the following command to activate the OpenVINO environment. <br><br>
`source /opt/intel/openvino/bin/setupvars.sh -pyver 3.6` . <br> 
<br>
#### NOTE -  Specify the pyver parameter according to your system. <br>

#### Step 2 - 
Run the python file by passing in the required command line arguments. <br><br>
`python3 people_count.py -m /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader/Retail/object_detection/pedestrian/rmnet_ssd/0013/dldt/FP32/person-detection-retail-0013.xml -l /opt/intel/openvino/inference_engine/lib/intel64/libcpu_extension.dylib -d CPU -pt 0.7`
<br>
<br>
#### NOTE - Make sure that you have properly installed and configured the OpenVINO toolkit before running the python program.


### References

This project borrows heavily from store-aisle-monitor-python project from the Intel IOT Devkit. <br>

Store Aisle Monitor - https://github.com/intel-iot-devkit/store-aisle-monitor-python

Intel IOT Devkit - https://github.com/intel-iot-devkit


### Usage

Intel allows you to use or modify their code if you meet the below mentioned conditions. <br>
If you want to use any code from this project or from any project of the Intel iot-devkit then you should crefully read the below menioned conditions. <br>


```
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```
