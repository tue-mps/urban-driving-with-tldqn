# urban-driving-with-tldqn
## Experimental Study on Thresholded Lexicographic Deep Q-Network Under Complex Urban Driving Scenarios
### Abstract
Autonomous driving in urban scenarios is extremely challenging for autonomous vehicles (AV)s. Besides complex interactions with different types of road users, the AV must also consider multiple aspects simultaneously while driving, such as safety, traffic rules, and the comfort of passengers. In this work, we conduct an experimental study on the performance of the Thresholded Lexicographic Deep Q-Network (TLDQN) algorithm in autonomous urban driving. We employ the CARLA simulator and design multiple objectives to represent challenging urban driving scenarios. The scenario includes various road users, i.e., pedestrians, vehicles, and cyclists. Furthermore, we consider the full dynamics of the ego vehicle and train the algorithms to directly control the low-level signals, e.g., throttle. We conduct a set of experiments demonstrating that TLDQN outperforms other algorithms in our designed scenarios. Furthermore, we show that TLDQN, compared to the traditional multi-objective algorithms, is less sensitive to assigned priorities for the objectives before training. Finally, we show that by adjusting some hyper-parameters of TLDQN after training, we can change the driving style of the ego vehicle.
## System requirement
```
python=3.8
torch=1.10.0+cu113
scenic=2.0.0
carla=0.9.13
```
## Setup
carla installation: https://carla.readthedocs.io/en/latest/start_quickstart/ \
pytorch installation: https://pytorch.org/get-started/locally/ \
scenic installation: https://scenic-lang.readthedocs.io/en/latest/install_notes.html
## Getstart
After installing the `scenic` package, place all files in this folder in the root of the scenic folder and replace the original files. You can run `scenic/examples/carla_scenic/Carla_Challenge/train_and_test.py` to train and test the agent.
### Trained models
All trained models will be saved in `scenic/examples/carla_scenic/Carla_Challenge`, according to different objectives, there has three folder called `collision`, `path`, `speed`. Folder `scenic/trained_models` includes previously trained models, which can be directly used as demonstration. Put these trained models in `scenic/examples/carla_scenic/Carla_Challenge` to use them.
### Other files
`RL-agent/DDQN.py`: pytorch program of DDQN model that used in TLDQN-based agent.\
`simulators/carla/simulator.py`: The script that used to control the simulation environment.\
`simulators/carla/utils/generate_traffic.py`: The script that used to generate traffic in current scenario.\
`simulators/carla/utils/HUD_render.py`: This script is used to generate a HUD that can monitor the status of the simulator.\
`simulators/carla/utils/Reward_functions.py`: This script includes the defination of reward functions for each objective.




