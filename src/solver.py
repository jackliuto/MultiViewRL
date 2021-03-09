# import package
import numpy as np

from ai2thor.controller import Controller

from PIL import Image

# import own files
from utils import convert2darknet

class Solver(object):
    def __init__(self,args):
        self.mode = args.mode
        self.data_path = args.data_path
        self.model_path = args.model_path
        self.controller_setting = args.controller_setting
        self.object_types = args.object_types
        self.env_seed = args.env_seed
    
    def train(self):
        print("train")
    
    def test(self):
        print("test")

    def gen_data(self):
        n = 0
        env_seed_list = self.env_seed[self.mode]
        for s in env_seed_list:
            with Controller(**self.controller_setting) as c:
                event = c.step(action='InitialRandomSpawn',randomSeed=s,forceVisible=True)
                event = c.step(action='GetReachablePositions')
                positions = event.metadata['reachablePositions']            
                for pos in positions:
                    event = c.step(action='Teleport', **pos)                
                    for i in range(360//self.controller_setting['rotateStepDegrees']):   
                        n = self.extract_info(event,n)
                        event = c.step('LookUp',degrees=30.0)
                        n = self.extract_info(event,n) 
                        event = c.step('LookDown',degrees=60.0)
                        n = self.extract_info(event,n) 
                        event = c.step('LookUp',degrees=30.0)                                  
                        event = c.step('RotateRight')   
               

    def extract_info(self,event,n):
        visible_objects = [o["objectId"] for o in event.metadata["objects"] if o['visible']]
        detected_objects = {k:v for k,v in event.instance_detections2D.items() if k in visible_objects}
        target_objects = {k:v for k,v in detected_objects.items() if k.split('|')[0] in self.object_types}
        if len(target_objects) != 0:
            image = Image.fromarray(event.frame)
            label = convert2darknet(target_objects,self.object_types,imgw=300.0,imgh=300.0,n=n)
            if len(label) > 0:
                image.save(self.data_path+'images/'+self.mode+'/'+str(n)+.'jpeg','JPEG')
                with open(self.data_path+'labels/'+self.mode+'/'+str(n)+'.txt','w') as f:
                    f.write(label)
                n += 1
        return n


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
        
