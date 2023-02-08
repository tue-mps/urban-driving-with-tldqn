import math

import numpy as np
import matplotlib.pyplot as plt


def speed_reward(ego_car_speed, tolerance=5):
    """
    :param ego_car_speed: the current speed of the ego vehicle
    :param tolerance: the acceptable range for the speed limit
    """
    vehicle_vel = math.sqrt(ego_car_speed[0] ** 2 + ego_car_speed[1] ** 2)
    current_speed = 3.6 * vehicle_vel
    if current_speed >= 20 and current_speed <= 35:
        reward = 1
    elif current_speed < 20:
        reward = current_speed / 10 - 1
    else:
        reward = -current_speed / 5 + 8
    if reward <= -1:
        reward = -1
    return reward

def pathfollowing_reward(current_state = None, current_route = None, ego_car_location = None):

    # the heading direction error
    route_distance = current_state[0]
    distance1 = current_state[1]
    distance2 = current_state[2]
    angel_diff1 = current_state[3]
    angle_diff2 = current_state[4]
    ego_diff = (distance1 / (distance2 + 0.1)) * angel_diff1 + ((distance2 - distance1) / (distance2 + 0.1)) * angle_diff2
    heading_reward = math.cos(ego_diff)

    # route cross error

    # this reward will show the distance between the refer path and ego-vehicle
    distance_bound = 1  # m here is the tolerance of the distance error
    # calculate the distance between vehicle position and the current trace
    point1 = current_route[0]
    point2 = current_route[1]
    A = point2[1] - point1[1]
    B = point1[0] - point2[0]
    C = (point1[1] - point2[1]) * point1[0] + (point2[0] - point1[0]) * point1[1]
    distance = np.abs(A * ego_car_location[0] + B * ego_car_location[1] + C) / (np.sqrt(A ** 2 + B ** 2) + 0.1)
    distance_reward = (distance_bound - distance) / distance_bound

    reward = (heading_reward + distance_reward) / 2

    if reward <= -1:
        reward = -1
    return reward

def collision_avoidence_reward(relative_location, ego_car_speed,collision_flag = 0):
    vehicle_vel = math.sqrt(ego_car_speed[0] ** 2 + ego_car_speed[1] ** 2)
    distance = math.sqrt(relative_location[0] ** 2 + relative_location[1] ** 2)
    # first judge whether there have any collison
    if collision_flag == 1:
        reward = -1
    # if there don't have any collsion, then the collision reward will based on safety distance with dangerous vehicle
    else:
        # here means that the system actually didn't detect any dangerous actors
        if relative_location == [0, 0]:
            reward = 0
        #when there have dangerous actors
        else:
            if distance >= 3:
                collision_time = distance / (vehicle_vel + 0.01)
                if collision_time >= 1.5:
                    reward = 1
                else:
                    reward = 2 * collision_time - 2
            else:
                if vehicle_vel != 0:
                    reward = -1
                else:
                    reward = 1
    return reward

# x = np.arange(100)
# y = [speed_reward([i / 3.6, 0]) for i in x]
# plt.plot(x, y)
# plt.xlabel('speed[km/h]')
# plt.ylabel('reward')
# plt.show()






# def pathfollowing_reward(ego_car_speed, ego_car_location, last_ego_car_location, trace, vector_mode=False,
#                          destination=None):
#
#     ego_car_speed = math.sqrt(ego_car_speed[0] ** 2 + ego_car_speed[1] ** 2)
#     # this reward will show whether the verhicle is going to the right direction
#     goal = destination
#     Lt = math.sqrt((last_ego_car_location[0] - goal[0]) ** 2 + (last_ego_car_location[1] - goal[1]) ** 2)
#     Lt1 = math.sqrt((ego_car_location[0] - goal[0]) ** 2 + (ego_car_location[1] - goal[1]) ** 2)
#     r_goal = (Lt - Lt1) / (ego_car_speed + 0.1)
#
#     # this reward will show the distance between the refer path and ego-vehicle
#     distance_bound = 1  # m here is the tolerance of the distance error
#     # calculate the distance between vehicle position and the current trace
#     point1 = trace[0]
#     point2 = trace[1]
#     A = point2[1] - point1[1]
#     B = point1[0] - point2[0]
#     C = (point1[1] - point2[1]) * point1[0] + (point2[0] - point1[0]) * point1[1]
#     distance = np.abs(A * ego_car_location[0] + B * ego_car_location[1] + C) / (np.sqrt(A ** 2 + B ** 2) + 0.1)
#     distance_reward = (distance_bound - distance) / distance_bound
#     if vector_mode:
#         return [r_goal, distance_reward]
#     else:
#         return r_goal + distance_reward
