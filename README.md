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
After installing the `scenic` package, place all files in this folder in the root of the scenic folder and replace the original files.


