# robot-memory-visualisation

## Installation
### Conda:
```
conda create --name <env> --file requirements_conda.txt
conda activate <env>
```

### Pip:
```
python3 -m venv <env>
source <env>/bin/activate
pip install -r requirements_pip.txt
```


# Set Up

prepare a web cam, and connect to the pc.

If the program complains about the camera's id when initializing the camera module, you may need to change the camera's id to your system's actual id.
```
camera = Camera(`CHANGE THIS`, width, height)
```


# How to Run
If you want to simulate the robot moving from region1 to region2, and perform instance segmentation, run:
```
python simulate_robot_moving.py
```

the program will:

- stay in region 1 (run for 10 seconds)
- move to region 2 (pause for 10 seconds) (you should adjust camera's position during this time)
- stay in region 2 (resume running for 10 seconds)
- move to region 1 (pause for 10 seconds) (you should put the camera back to where it was in region 1)
- ...

You can customize the regions and timing in the simulate_robot_moving.py

.


If you only want to track objects moving in/out of a room, run:
```
python tracking_yolo_SORT.py
```

The recordings of objects moving in/out will be stored in the current directory.