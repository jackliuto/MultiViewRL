import numpy as np
from PIL import Image
from ai2thor.controller import Controller

# with Controller(scene='FloorPlan1', gridSize=0.25,renderObjectImage=True) as c:
#     event = c.step(action='MoveAhead')
    # print(event.instance_detections2D)
    
    # event.frame
    # event.cv2image
    # event.metadata

with Controller(scene='FloorPlan1', gridSize=0.25,renderObjectImage=True) as c:
    event = c.step(action='GetReachablePositions')
    positions = event.metadata['reachablePositions']
    n = 0
    for pos in positions:
        event = c.step(action='Teleport', **pos)
        Image.fromarray(event.frame).save('./'+str(n)+'.jpg')
        print(event.instance_detections2D)
        n+=1
        if n > 8:
            raise ValueError
        # event = c.step(action='RotateLeft')
        # Image.fromarray(event.frame).save('./images/'+str(n)+'.jpg')
        # n+=1
        # event = c.step(action='RotateLeft')
        # Image.fromarray(event.frame).save('./images/'+str(n)+'.jpg')
        # n+=1
        # event = c.step(action='RotateLeft')
        # Image.fromarray(event.frame).save('./images/'+str(n)+'.jpg')
        # n+=1

        