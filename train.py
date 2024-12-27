""" Train INANO.Gano
"""


import warnings
from argparse import ArgumentParser

from solver import Solver

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-train_config_path", default="train.yaml")
    parser.add_argument("-dataset_path")
    parser.add_argument("-device", default="cuda")
    parser.add_argument("-load_model", action="store_true")
    parser.add_argument("-load_opt", action="store_true")
    parser.add_argument("-store_model_path")
    parser.add_argument("-load_model_path")
    parser.add_argument("-summary_steps", default=2, type=int)
    parser.add_argument("-save_steps", default=5, type=int)
    parser.add_argument("-iterations", default=5, type=int)

    args = parser.parse_args()
    solver = Solver(
        trainConfig_path=args.train_config_path, args=args
    )

    if args.iterations > 0:
        solver.train(n_iterations=args.iterations)

    completed = input("[INANO]training completed 🍹")