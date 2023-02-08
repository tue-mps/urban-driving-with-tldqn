import logging
import carla
from numpy import random

def get_actor_blueprints(world, filter, generation):
    bps = world.get_blueprint_library().filter(filter)

    if generation.lower() == "all":
        return bps

    # If the filter returns only one bp, we assume that this one needed
    # and therefore, we ignore the generation
    if len(bps) == 1:
        return bps

    try:
        int_generation = int(generation)
        # Check if generation is in available generations
        if int_generation in [1, 2]:
            bps = [x for x in bps if int(x.get_attribute('generation')) == int_generation]
            return bps
        else:
            print("   Warning! Actor Generation is not valid. No actor will be spawned.")
            return []
    except:
        print("   Warning! Actor Generation is not valid. No actor will be spawned.")
        return []

def generate_traffic(vehicle_num = 30, ped_num = 30, carla_client = None):
    tm_port = 8000
    vehicle_list = []
    ped_list = []
    all_id = []

    world = carla_client.get_world()

    traffic_manager = carla_client.get_trafficmanager(tm_port)
    traffic_manager.set_global_distance_to_leading_vehicle(2.5)
    traffic_manager.set_hybrid_physics_mode(True)
    traffic_manager.set_hybrid_physics_radius(70.0)

    blueprints = get_actor_blueprints(world, "vehicle.*", "ALL")
    blueprints_ped = get_actor_blueprints(world, 'walker.pedestrian.*', "ALL")
    blueprints = sorted(blueprints, key=lambda bp: bp.id)

    spawn_points = world.get_map().get_spawn_points()
    number_of_spawn_points = len(spawn_points)
    if vehicle_num < number_of_spawn_points:
        random.shuffle(spawn_points)
    elif vehicle_num > number_of_spawn_points:
        msg = 'requested %d vehicles, but could only find %d spawn points'
        logging.warning(msg, vehicle_num, number_of_spawn_points)
        vehicle_num = number_of_spawn_points

    SpawnActor = carla.command.SpawnActor
    SetAutopilot = carla.command.SetAutopilot
    FutureActor = carla.command.FutureActor

    #####################################
    # spawn vehicles

    batch = []
    for n, transform in enumerate(spawn_points):
        if n >= vehicle_num:
            break
        blueprint = random.choice(blueprints)
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        if blueprint.has_attribute('driver_id'):
            driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
            blueprint.set_attribute('driver_id', driver_id)
        blueprint.set_attribute('role_name', 'autopilot')
        batch.append(SpawnActor(blueprint, transform)
                     .then(SetAutopilot(FutureActor, True, traffic_manager.get_port())))

    for response in carla_client.apply_batch_sync(batch, False):
        if response.error:
            logging.error(response.error)
        else:
            vehicle_list.append(response.actor_id)

    ##############################
    # spawn pedestrians
    percentagePedestriansRunning = 0.0
    percentagePedestriansCrossing = 0.0

    # find all the possible points for generating the pedestrains
    spawn_points = []
    for i in range(ped_num):
        spawn_point = carla.Transform()
        loc = world.get_random_location_from_navigation()
        if (loc != None):
            spawn_point.location = loc
            spawn_points.append(spawn_point)
    # spawn pedestrains
    batch = []
    ped_speed = []
    for spawn_point in spawn_points:
        ped_bp = random.choice(blueprints_ped)
        # ped is not invincible
        if ped_bp.has_attribute('is_invincible'):
            ped_bp.set_attribute('is_invincible', 'false')
        # set the max speed
        if ped_bp.has_attribute('speed'):
            if (random.random() > percentagePedestriansRunning):
                # walking
                ped_speed.append(ped_bp.get_attribute('speed').recommended_values[1])
            else:
                # running
                ped_speed.append(ped_bp.get_attribute('speed').recommended_values[2])
        else:
            print("Walker has no speed")
            ped_speed.append(0.0)
        batch.append(SpawnActor(ped_bp, spawn_point))
    results = carla_client.apply_batch_sync(batch, True)
    ped_speed2 = []
    for i in range(len(results)):
        if results[i].error:
            logging.error(results[i].error)
        else:
            ped_list.append({"id": results[i].actor_id})
            ped_speed2.append(ped_speed[i])
    ped_speed = ped_speed2
    #spawn the ped controller
    batch = []
    ped_controller_bp = world.get_blueprint_library().find('controller.ai.walker')
    for i in range(len(ped_list)):
        batch.append(SpawnActor(ped_controller_bp, carla.Transform(), ped_list[i]["id"]))
    results = carla_client.apply_batch_sync(batch, True)
    for i in range(len(results)):
        if results[i].error:
            logging.error(results[i].error)
        else:
            ped_list[i]["con"] = results[i].actor_id

    # we put together the walkers and controllers id to get the objects from their id
    for i in range(len(ped_list)):
        all_id.append(ped_list[i]["con"])
        all_id.append(ped_list[i]["id"])
    all_actors = world.get_actors(all_id)
    world.tick()

    # setup the behavior model for the peds
    world.set_pedestrians_cross_factor(percentagePedestriansCrossing)

    for i in range(0, len(all_id), 2):
        # start walker
        all_actors[i].start()
        # set walk to random point
        all_actors[i].go_to_location(world.get_random_location_from_navigation())

        #max speed
        all_actors[i].set_max_speed(float(ped_speed[int(i/2)]))

    print('spawned %d vehicles and %d walkers, press Ctrl+C to exit.' % (len(vehicle_list), len(ped_list)))

    # Example of how to use Traffic Manager parameters
    traffic_manager.global_percentage_speed_difference(30.0)

    return vehicle_list, all_id, all_actors


