# import package
import numpy as np

from ai2thor.controller import Controller

from PIL import Image

# import own files
from utils import get_agent_map_data

class Solver(object):
    def __init__(self,args):
        self.mode = args.mode
        self.data_path = args.data_path
        self.model_path = args.model_path
    
    def train(self):
        print("train")
    
    def test(self):
        print("test")

    def sandbox(self):
        with Controller(scene='FloorPlan1', gridSize=0.25, renderObjectImage=True) as c:
            event = c.step(action="RotateRight")
            positions = c.step(action="GetReachablePositions").metadata["actionReturn"]
            print(event.metadata['agent']['rotation'])



        print("sandbox")

        # ## code for generate top down view 
        # for i in range(1,11):
        #     controller = Controller(scene='FloorPlan20'+str(i), gridSize=0.25, renderObjectImage=True)
        #     map_view = get_agent_map_data(controller)['frame']
        #     im = Image.fromarray(map_view)
        #     im.save(self.data_path+'map20'+str(i)+'.png')

        ## code for testing random location of objects
        # for i in range(1,11):
        #     controller = Controller(scene='FloorPlan205', gridSize=0.25, renderObjectImage=True)
        #     controller.step(action='InitialRandomSpawn',randomSeed=i)
        #     map_view = get_agent_map_data(controller)['frame']
        #     im = Image.fromarray(map_view)
        #     im.save(self.data_path+'map205_'+str(i)+'.png')

        ## code for generate first person view
        # event = controller.step(action='MoveAhead')
        # print(event.instance_detections2D)
        # im = Image.fromarray(event.frame)
        # im.save(self.data_path+'test1.png')

        ## code for check last action sucess or fail
        # event.metadata['lastActionSuccess']

        # event = controller.step(action='MoveAhead')
        # im = Image.fromarray(event.frame)
        # im.save(self.data_path+'test2.png')

        # event = controller.step(action='MoveAhead')
        # im = Image.fromarray(event.frame)
        # im.save(self.data_path+'test3.png')
        

