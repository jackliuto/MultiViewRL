# import packages
import argparse

# import own files
from solver import Solver


#  variables
mode = "val"
seed = 1

data_path = "../data/data_salad/"
model_path = "../model/"

env_seed = {
    'train':list(range(1,11)),
    'val':[13],
    'test':[14]
}
object_types = ['Apple','Tomato','Lettuce']
# object_types = ['Cup','Kettle','Pot']
controller_setting = {
    'scene':'FloorPlan7',
    'gridSize':0.25,
    'rotateStepDegrees':30,
    'renderObjectImage':True,
    'visibilityDistance':1.0
}




def main(args):
    solver = Solver(args)
    if args.mode.lower() == "train":
        solver.gen_data()
    elif args.mode.lower() == "val":
        solver.gen_data()
    elif args.mode.lower() == "test":
        solver.gen_data()    
    elif args.mode.lower() == "gen_data":
        solver.gen_data()
    else:
        print("{} not implemted.".format(args.mode))
        raise ValueError
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This is the source code for Graph Navigation in 3D enviroments.')
    parser.add_argument('--mode', default=mode, type=str, help='Set the mode of the solver, train or test')

    parser.add_argument('--data_path', default=data_path, type=str, help='Path to the data folder')
    parser.add_argument('--model_path', default=model_path, type=str, help='Path to the model folder')

    parser.add_argument('--object_types', default=object_types, type=list, help='type of objects to detect')
    parser.add_argument('--controller_setting', default=controller_setting, type=dict, help='settings for ithor controller')
    parser.add_argument('--env_seed', default=env_seed, type=dict, help='seed for train/valid/test enviroments')


    args = parser.parse_args()
    main(args)
