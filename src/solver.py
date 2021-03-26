# import package
import numpy as np
import time
import torch
from pathlib import Path

from ai2thor.controller import Controller

from PIL import Image
import torchvision.transforms as transforms
from matplotlib.pyplot import imshow
import imageio

# import own files
from utils import get_agent_map_data
from agent import DQNAgent
from logger import MetricLogger


class Solver(object):
    def __init__(self,args):
        self.mode = args.mode
        self.data_path = args.data_path
        self.model_path = args.model_path

        # enviroment settings
        self.target_objects = ["Pillow"]
        self.actions = ["MoveAhead","MoveBack","RotateLeft","RotateRight"]
        self.state_dim = (3, 256, 256)
        self.controller_setting = dict(
            scene='FloorPlan204',             
            agentMode="default",
            gridSize=0.25, 
            snapToGrid=True,
            rotateStepDegrees=90, 
            renderObjectImage=True,
            width=self.state_dim[1],
            height=self.state_dim[2],
            fieldOfView=60
        )
    
    def end_episode(self, event):
        visible_objects = [o["objectId"] for o in event.metadata["objects"] if o['visible']]
        detected_objects = {k:v for k,v in event.instance_detections2D.items() if k in visible_objects}
        seen_objects = {k:v for k,v in detected_objects.items() if k.split('|')[0] in self.target_objects}
        object_seen = False
        if len(seen_objects) != 0:
            object_seen = True
        return object_seen
    
    def reward(self, action, object_detected, action_failure):
        total_reward = 0
        if object_detected:
            total_reward += 100
        if action_failure:
            total_reward -= 10
        # penalize movement rotations and move ackward
        if action != 0:
            total_reward -= 1
        return total_reward
    
    def extract_state(self, controller, event):
        ego = np.transpose(event.frame, (2, 0, 1))
        ego = torch.tensor(ego.copy(), dtype=torch.float)
        alo = np.transpose(get_agent_map_data(controller)['frame'], (2, 0, 1))
        alo = torch.tensor(alo.copy(), dtype=torch.float)
        # alo = np.moveaxis(get_agent_map_data(controller)['frame'], -1, 0).astype(np.float32)
        return dict(ego=ego, alo=alo)

    def env_step(self, controller, action):
        event = controller.step(action=self.actions[action])
        object_detected = self.end_episode(event)
        action_failure = event.metadata['lastActionSuccess']
        reward = self.reward(action, object_detected, action_failure)
        next_state = self.extract_state(controller, event)
        return next_state, reward, object_detected


    def train(self):

        # setup
        dataPath = Path(self.data_path) 
        modelPath = Path(self.model_path)
        logger = MetricLogger(modelPath)
        dqn_agent = DQNAgent(self.state_dim, len(self.actions),modelPath)
        episodes = 500


        with Controller(**self.controller_setting) as c:

            # # event = c.reset(**self.controller_setting)
            # event = c.step(action='InitialRandomSpawn', randomSeed=5, forceVisible=True)
            # # event = c.step(action='InitialRandomSpawn', randomSeed=5, forceVisible=True)  
            # # event = c.step(action='InitialRandomSpawn', randomSeed=5, forceVisible=True)  
            # # event = c.step(action='InitialRandomSpawn', randomSeed=5, forceVisible=True)  
            # # event = c.step(action='InitialRandomSpawn', randomSeed=5, forceVisible=True)                 
            # map_view = get_agent_map_data(c)['frame']
            # im = Image.fromarray(map_view)
            # im.save('../data/debug/'+'pick'+'.png')
            # raise ValueError

            event = c.step(action='InitialRandomSpawn', randomSeed=5, forceVisible=True)
            agent_init = event.metadata["agent"]
            for e in range(episodes):
                # initialial state
                event = c.step(action='Teleport', **agent_init)
                state = self.extract_state(c, event)
                ego_state = state['ego']
                alo_state = state['alo']
                n = 0
                while True:                   
                    # get state tuple
                    action = dqn_agent.act(ego_state)
                    next_state, reward, done = self.env_step(c, action)
                    ego_next_state = state['ego']
                    alo_next_state = state['alo']

                    # memorize
                    dqn_agent.cache(ego_state,ego_next_state, action, reward, done)

                    # learn
                    q, loss = dqn_agent.learn()

                    # log
                    logger.log_step(reward, loss, q)

                    state = next_state
                    ego_state = state['ego']
                    alo_state = state['alo']

                    ### Code for save images
                    # im_ego = np.transpose(ego_state, (1, 2, 0)).numpy()
                    # im_ego = transforms.ToPILImage()(np.uint8(im_ego))                  
                    # im_ego.save('../data/debug/'+'ego'+str(n)+'.png')

                    # im_alo = np.transpose(alo_state, (1, 2, 0)).numpy()
                    # im_alo = transforms.ToPILImage()(np.uint8(im_alo))  
                    # im_alo.save('../data/debug/'+'alo'+str(n)+'.png')

                    n += 1
                    if done or n > 1000:
                        break                   

                logger.log_episode()
                logger.record(episode=e, epsilon=dqn_agent.exploration_rate, step=dqn_agent.curr_step)

            
            
            # episodes = 1
            # for e in range(episodes):

            
    def test(self):
        # setup
        dataPath = Path(self.data_path) 
        modelPath = Path(self.model_path)
        logger = MetricLogger(modelPath)
        dqn_agent = DQNAgent(self.state_dim, len(self.actions),modelPath)
        dqn_agent.load('../model/dqn_20.chkpt')
        episodes = 1000

        gif_array = []
        gif_location = '../test.gif'


        with Controller(**self.controller_setting) as c:

            event = c.step(action='InitialRandomSpawn', randomSeed=5, forceVisible=True)
            agent_init = event.metadata["agent"]

            for e in range(episodes):
                # initialial state
                event = c.step(action='Teleport', **agent_init)
                state = self.extract_state(c, event)
                ego_state = state['ego']
                alo_state = state['alo']
                n = 0
                while True:                   
                    # get state tuple
                    action = dqn_agent.act_no_grad(ego_state)
                    next_state, reward, done = self.env_step(c, action)
                    ego_next_state = state['ego']
                    alo_next_state = state['alo']

                    ###### convert to gif
                    im_ego = np.transpose(ego_state, (1, 2, 0)).numpy()
                    im_ego = transforms.ToPILImage()(np.uint8(im_ego))
                    gif_array.append(im_ego)

                    # # memorize
                    # dqn_agent.cache(ego_state,ego_next_state, action, reward, done)

                    # # learn
                    # q, loss = dqn_agent.learn()

                    # # log
                    # logger.log_step(reward, loss, q)

                    state = next_state
                    ego_state = state['ego']
                    alo_state = state['alo']

                    ### Code for save images
                    # im_ego = np.transpose(ego_state, (1, 2, 0)).numpy()
                    # im_ego = transforms.ToPILImage()(np.uint8(im_ego))                  
                    # im_ego.save('../data/debug/'+'ego'+str(n)+'.png')

                    # im_alo = np.transpose(alo_state, (1, 2, 0)).numpy()
                    # im_alo = transforms.ToPILImage()(np.uint8(im_alo))  
                    # im_alo.save('../data/debug/'+'alo'+str(n)+'.png')

                    n += 1
                    if done or n > 100:
                        imageio.mimsave(gif_location, gif_array)
                        break                   

                # logger.log_episode()
                # logger.record(episode=e, epsilon=dqn_agent.exploration_rate, step=dqn_agent.curr_step)




