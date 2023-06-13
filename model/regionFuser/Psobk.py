import sys
import tqdm
from model.regionFuser.Evaluation import PsoFitness, DiscretePsoFitness


def pso(cfg, dictPseudoBoxesForModel, dictGtBoxes):
    import pyswarms as ps

    n_model = len(cfg.val_test.model_path)
    n_cam = len(cfg.box_fusion.cam_list)
    n_class = cfg.dataset.num_classes_with_boxes
    n_dim = n_model * n_cam * n_class

    # Define optimization problems
    bounds = [[0 for x in range(n_dim)], [1 for x in range(n_dim)]]
    options = {'c1': cfg.box_fusion.optimizer_param["c1"],
               'c2': cfg.box_fusion.optimizer_param["c2"],
               'w': cfg.box_fusion.optimizer_param["w"]}

    # Create optimizer object
    optimizer = ps.single.GlobalBestPSO(n_particles=cfg.box_fusion.optimizer_param["num_particles"], dimensions=n_dim, options=options, bounds=bounds)

    fitFunc = PsoFitness(n_model, n_cam, n_class, dictGtBoxes, dictPseudoBoxesForModel, cfg)

    # Run optimizer
    best_error, best_position = optimizer.optimize(fitFunc, iters=cfg.box_fusion.optimizer_param["max_iterations"], n_processes=8)

    # Print Results
    print("Best position:", best_position)
    print("Best error:", best_error)

    return best_position


def discretePso(cfg, dictPseudoBoxesForModel, dictGtBoxes):
    import random
    import numpy as np

    n_model = len(cfg.val_test.model_path)
    n_cam = len(cfg.box_fusion.cam_list)
    n_class = cfg.dataset.num_classes_with_boxes
    n_dim = n_model * n_cam * n_class

    # Define the problem
    objective_function = DiscretePsoFitness(n_model, n_cam, n_class, dictGtBoxes, dictPseudoBoxesForModel, cfg)

    # Define the PSO parameters
    num_particles = cfg.box_fusion.optimizer_param["num_particles"]
    num_dimensions = n_dim
    max_iterations = cfg.box_fusion.optimizer_param["max_iterations"]
    w = cfg.box_fusion.optimizer_param["w"]
    c1 = cfg.box_fusion.optimizer_param["c1"]
    c2 = cfg.box_fusion.optimizer_param["c2"]

    # Set the boundary parameters
    boundary = np.zeros((num_dimensions, 2))
    for i in range(num_dimensions):
        boundary[i][0] = 0
        boundary[i][1] = 9

    # Initialize the particles
    particles = np.zeros((num_particles, num_dimensions))
    for i in range(num_particles):
        for j in range(num_dimensions):
            particles[i][j] = random.randint(boundary[j][0], boundary[j][1])

    # Initialize the velocities
    velocities = np.zeros((num_particles, num_dimensions))

    # Initialize the personal best positions and fitness values
    personal_best_positions = particles.copy()
    personal_best_fitness_values = np.zeros(num_particles)
    for i in range(num_particles):
        personal_best_fitness_values[i] = objective_function(personal_best_positions[i])

    # Initialize the global best position and fitness value
    global_best_position = np.zeros(num_dimensions)
    global_best_fitness_value = float('inf')

    # Update the particles
    progressBar = tqdm.tqdm(range(max_iterations), colour='green', file=sys.stdout)
    for i in progressBar:
        for j in range(num_particles):
            # Update the velocity
            velocities[j] = w * velocities[j] + c1 * random.random() * (
                    personal_best_positions[j] - particles[j]) + c2 * random.random() * (
                                    global_best_position - particles[j])

            # Update the position
            for k in range(num_dimensions):
                particles[j][k] = max(min(particles[j][k] + int(velocities[j][k]), boundary[k][1]), boundary[k][0])

            # Compute the fitness value
            fitness_value = objective_function(particles[j])

            # Update the personal best position and fitness value
            if fitness_value < personal_best_fitness_values[j]:
                personal_best_fitness_values[j] = fitness_value
                personal_best_positions[j] = particles[j].copy()

            # Update the global best position and fitness value
            if fitness_value < global_best_fitness_value:
                global_best_fitness_value = fitness_value
                global_best_position = particles[j].copy()
                progressBar.write(f"global_best_fitness_value : {global_best_fitness_value}")

    # Output the result
    print("The optimal solution is:")
    print(global_best_position)
    print("The optimal fitness value is:")
    print(global_best_fitness_value)

    return global_best_position


def binaryPso(cfg, dictPseudoBoxesForModel, dictGtBoxes):
    from pyswarms.discrete.binary import BinaryPSO

    n_model = len(cfg.val_test.model_path)
    n_cam = len(cfg.box_fusion.cam_list)
    n_class = cfg.dataset.num_classes_with_boxes
    n_dim = n_model * n_cam * n_class

    # Define optimization problems
    options = {'c1': cfg.box_fusion.optimizer_param["c1"],
               'c2': cfg.box_fusion.optimizer_param["c2"],
               'w': cfg.box_fusion.optimizer_param["w"],
               'k': cfg.box_fusion.optimizer_param["k"],
               'p': cfg.box_fusion.optimizer_param["p"]}

    # Create optimizer object
    optimizer = BinaryPSO(n_particles=cfg.box_fusion.optimizer_param["num_particles"], dimensions=n_dim, options=options)

    fitFunc = PsoFitness(n_model, n_cam, n_class, dictGtBoxes, dictPseudoBoxesForModel, cfg)

    # Run optimizer
    best_error, best_position = optimizer.optimize(fitFunc, iters=cfg.box_fusion.optimizer_param["max_iterations"], n_processes=8)

    # Print Results
    print("Best position:", best_position)
    print("Best error:", best_error)

    return best_position
