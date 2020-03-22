### About
This app can detect eye blinking. It also provides the recommended number of blinks over time.

### Support
- python 3.x
- macOS

### Usage
**Be sure to use the default terminal provided by the Mac. If iTerm is used, the camera app can't be accessed.**

install third party library
```bash
$ pip install -r requirements.txt
```

you can see the arguments
```bash
$ python main.py -h

optional arguments:
  -h, --help            show this help message and exit
  -t THRESHOLD, --threshold THRESHOLD
                        threshold to determine close eyes
  -v [VERBOSE], --verbose [VERBOSE]
                        show frame on your face
```

run
```bash
$ python main.py
// if you want to see the frame
$ python main.py -v
```
