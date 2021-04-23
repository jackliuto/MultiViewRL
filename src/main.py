# This is a commandline program that make sure we can run our code on server

# import packages
import argparse

# import own files
from solver import Solver 


#  variables
mode = "test"


data_path = "../data/"
model_path = "../model/"

def main(args):
    # torch.manual_seed(args.seed)
    # np.random.seed(args.seed)
    # random.seed(args.seed)
    solver = Solver(args)
    if args.mode.lower() == "train_ego":
        solver.train_ego()
    elif args.mode.lower() == "train_alo":
        solver.train_alo()
    elif args.mode.lower() == "train_double":
        solver.train_double()
    elif args.mode.lower() == "test":
        solver.test()
    elif args.mode.lower() == "sandbox":
        solver.sandbox()
    else:
        print("{} not implemted.".format(args.mode))
        raise ValueError
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This is the source code for Graph Navigation in 3D enviroments.')
    parser.add_argument('--mode', default=mode, type=str, help='Set the mode of the solver, train or test')

    parser.add_argument('--data_path', default=data_path, type=str, help='Path to the data folder')
    parser.add_argument('--model_path', default=model_path, type=str, help='Path to the model folder')

    args = parser.parse_args()
    main(args)

