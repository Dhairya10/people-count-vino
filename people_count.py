import os
import sys
import time
from argparse import ArgumentParser
import pathlib
import cv2
import numpy as np
import json
from inference import Network

isasyncmode = True
CONFIG_FILE = 'resources/config.json'
MULTIPLICATION_FACTOR = 5

# To get current working directory
CWD = os.getcwd()

# Creates subdirectory to save output snapshots
pathlib.Path(CWD + '/output_snapshots/').mkdir(parents=True, exist_ok=True)

def build_argparser():
    parser = ArgumentParser()
    parser.add_argument("-m", "--model",
                        help="Path to an .xml file with a trained model.",
                        required=True, type=str)
    parser.add_argument("-l", "--cpu_extension",
                        help="MKLDNN (CPU)-targeted custom layers. Absolute "
                             "path to a shared library with the kernels impl.",
                        type=str, default=None)
    parser.add_argument("-d", "--device",
                        help="Specify the target device to infer on; "
                             "CPU, GPU, FPGA, HDDL or MYRIAD is acceptable. Application"
                             " will look for a suitable plugin for device "
                             "specified (CPU by default)", default="CPU", type=str)
    parser.add_argument("-pt", "--prob_threshold",
                        help="Probability threshold for detections filtering",
                        default=0.5, type=float)
    parser.add_argument("-f", "--flag", help="sync or async", default="async", type=str)

    return parser


def apply_time_stamp_and_save(image, people_count):
    """
    Saves snapshots with timestamps.
    """
    current_date_time = time.strftime("%y-%m-%d_%H:%M:%S", time.gmtime())
    file_name = current_date_time + "_PCount_" + str(people_count) + ".png"
    file_path = CWD + "/output_snapshots/"
    local_file_name = "output_" + file_name
    file_name = file_path + local_file_name
    cv2.imwrite(file_name, image)

def main():
    global CONFIG_FILE
    global is_async_mode
    args = build_argparser().parse_args()

    assert os.path.isfile(CONFIG_FILE), "{} file doesn't exist".format(CONFIG_FILE)
    config = json.loads(open(CONFIG_FILE).read())
    for idx, item in enumerate(config['inputs']):
        if item['video'].isdigit():
            input_stream = int(item['video'])
            cap = cv2.VideoCapture(input_stream)
            if not cap.isOpened():
                print("\nCamera not plugged in... Exiting...\n")
                sys.exit(0)
        else:
            input_stream = item['video']
            cap = cv2.VideoCapture(input_stream)
            if not cap.isOpened():
                print("\nUnable to open video file... Exiting...\n")
                sys.exit(0)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if args.flag == "async":
        is_async_mode = True
        print('Application running in async mode')
    else:
        is_async_mode = False
        print('Application running in sync mode')

    # Initialise the class
    infer_network = Network()
    # Load the network to IE plugin to get shape of input layer
    n, c, h, w = infer_network.load_model(args.model, args.device, 1, 1, 2, args.cpu_extension)[1]

    print("To stop the execution press Esc button")
    initial_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    initial_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_count = 1
    accumulated_image = np.zeros((initial_h, initial_w), np.uint8)
    mog = cv2.createBackgroundSubtractorMOG2()
    ret, frame = cap.read()
    cur_request_id = 0
    next_request_id = 1

    while cap.isOpened():
        ret, next_frame = cap.read()
        start_time = time.time()

        if not ret:
            break
        frame_count = frame_count + 1
        in_frame = cv2.resize(next_frame, (w, h))
        # Change data layout from HWC to CHW
        in_frame = in_frame.transpose((2, 0, 1))
        in_frame = in_frame.reshape((n, c, h, w))

        # Start asynchronous inference for specified request.
        inf_start = time.time()
        if isasyncmode:
            infer_network.exec_net(next_request_id, in_frame)
        else:
            infer_network.exec_net(cur_request_id, in_frame)
        # Wait for the result
        if infer_network.wait(cur_request_id) == 0:
            det_time = time.time() - inf_start
            people_count = 0

            # Converting to Grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Remove the background
            fgbgmask = mog.apply(gray)
            # Thresholding the image
            thresh = 2
            max_value = 2
            threshold_image = cv2.threshold(fgbgmask, thresh, max_value,
                                              cv2.THRESH_BINARY)[1]
            # Adding to the accumulated image
            accumulated_image = cv2.add(threshold_image, accumulated_image)
            colormap_image = cv2.applyColorMap(accumulated_image, cv2.COLORMAP_HOT)

            # Results of the output layer of the network
            res = infer_network.get_output(cur_request_id)
            for obj in res[0][0]:
                # Draw only objects when probability more than specified threshold
                if obj[2] > args.prob_threshold:
                    xmin = int(obj[3] * initial_w)
                    ymin = int(obj[4] * initial_h)
                    xmax = int(obj[5] * initial_w)
                    ymax = int(obj[6] * initial_h)
                    class_id = int(obj[1])
                    # Draw bounding box
                    color = (min(class_id * 12.5, 255), min(class_id * 7, 255),
                    min(class_id * 5, 255))
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                    people_count = people_count + 1

            people_count_message = "People Count : " + str(people_count)
            cv2.putText(frame, people_count_message, (15, 65), cv2.FONT_HERSHEY_COMPLEX, 1,
                     (0, 0, 0), 2)

            cv2.imshow("Detection Results", frame)

            time_interval = MULTIPLICATION_FACTOR * fps
            if frame_count % time_interval == 0:
                apply_time_stamp_and_save(frame, people_count)

        frame = next_frame
        if isasyncmode:
            cur_request_id, next_request_id = next_request_id, cur_request_id
        print("FPS : {}".format(1/(time.time() - start_time)))

        # Frames are read at an interval of 1 millisecond
        key = cv2.waitKey(1)
        if key == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    infer_network.clean()


if __name__ == '__main__':
    sys.exit(main() or 0)
