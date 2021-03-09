# import package
import numpy as np

from ai2thor.controller import Controller

from PIL import Image

# import own files
from utils import get_agent_map_data


data_path = "../data/"
file_n = ""

## code for generate top down view 
# for i in range(1,31):
#     with Controller(scene='FloorPlan'+file_n+str(i), gridSize=0.25, renderObjectImage=True) as controller:
#         map_view = get_agent_map_data(controller)['frame']
#         im = Image.fromarray(map_view)
#         im.save(data_path+'map'+file_n+str(i)+'.png')

# # code for testing random location of objects
# for i in range(1,2):
#     with Controller(scene='FloorPlan7', gridSize=0.25, renderObjectImage=True) as controller:
#         controller.step(action='InitialRandomSpawn',randomSeed=i)
#         map_view = get_agent_map_data(controller)['frame']
#         im = Image.fromarray(map_view)
#         im.save(data_path+'map7_'+str(i)+'.png')

# # code for first person object detection
# with Controller(scene='FloorPlan7', gridSize=0.25, renderObjectImage=True) as controller:
#     # event = controller.step('Pass')
#     # print(event.class_detections2D)
#     o = []
#     for obj in controller.last_event.metadata['objects']:        
#         o.append(obj['objectType'])
#     o.sort()
#     print(o)
#         # controller.step(action='InitialRandomSpawn',randomSeed=i)
#         # map_view = get_agent_map_data(controller)['frame']
#         # im = Image.fromarray(map_view)
#         # im.save(data_path+'map7_'+str(i)+'.png')





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

