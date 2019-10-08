# people-count-vino
This is a python program which counts the number of people in a given frame. <br>
It uses the OpenVINO toolkit for faster inference.<br>
RMNet is used here as a backbone model. <br>
It is a combination of ResNet and MobileNet. 

### Directory Structure 

```
.
├── people_count.py
└── resources
    └── config.json

1 directory, 2 files`
```
`people_count.py` - It is the python program which counts the number of people in a given frame. <br>
`config.json` - It is a json file which determines the source of the video. You can specify the url to a video file or your webcam id.

