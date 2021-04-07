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
from moviepy.editor import *

# import own files
from utils import get_agent_map_data
from agent import DQNAgent, MVDQNAgent
from logger import MetricLogger


class Solver(object):
    def __init__(self,args):
        self.mode = args.mode
        self.scene = 'FloorPlan222'
        self.other_name = ''
        self.seed = 11
        self.agent_horizon = 25
        
        self.max_iteration = 300
        self.episodes = 200


        self.data_path = args.data_path
        self.model_path = args.model_path

        # enviroment settings
        self.target_objects = ["Box"]
        self.actions = ["MoveAhead","MoveBack","RotateLeft","RotateRight"]
        self.state_dim = (3, 256, 256)
        self.controller_setting = dict(
            visibilityDistance=0.75,
            scene= self.scene,             
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
        discover_flag = False
        if len(seen_objects) != 0:
            discover_flag = True
        return discover_flag
    
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
        return dict(ego=ego, alo=alo)

    def env_step(self, controller, action):
        event = controller.step(action=self.actions[action])
        object_detected = self.end_episode(event)
        action_failure = not event.metadata['lastActionSuccess']
        reward = self.reward(action, object_detected, action_failure)
        next_state = self.extract_state(controller, event)
        return next_state, reward, object_detected


    def sandbox(self):

        # setup
        dataPath = Path(self.data_path) 
        modelPath = Path(self.model_path)
        logger = MetricLogger(modelPath)
        dqn_agent = DQNAgent(self.state_dim, len(self.actions),modelPath)
        episodes = 100
        seed = 0
       
        # code for generate topdown view 
        # while seed < 50:
        #     with Controller(**self.controller_setting) as c:
        #         event = c.step(action='InitialRandomSpawn', randomSeed=seed, forceVisible=True)
        #         map_view = get_agent_map_data(c)['frame']
        #         im = Image.fromarray(map_view)
        #         im.save('../data/topdown222/'+str(seed)+'.png')
        #         seed += 1

        # ## code for debug images
        # with Controller(**self.controller_setting) as c:
        #     event = c.step(action='InitialRandomSpawn',randomSeed=11,forceVisible=True)
        #     event = c.step(action='GetReachablePositions')
        #     positions = event.metadata['actionReturn']
        #     n = 0
        #     for pos in positions:
        #         event = c.step(action='Teleport', position=pos, horizon=25)
        #         # event = c.step("LookDown", degrees=0)
        #         for i in range(360//90): 
        #             event = c.step('RotateRight')
        #             visible_objects = [o["objectId"] for o in event.metadata["objects"] if o['visible']]
        #             detected_objects = {k:v for k,v in event.instance_detections2D.items() if k in visible_objects}
        #             target_objects = {k:v for k,v in detected_objects.items() if k.split('|')[0] in self.target_objects}
        #             # print(visible_objects)
        #             # print(detected_objects)
        #             # print(target_objects)
        #             # raise ValueError
        #             if len(target_objects) != 0:
        #                 print(target_objects)
        #                 image = Image.fromarray(event.frame)
        #                 image.save('/home/jackliu/School/CISC856/MultiViewRL/data/image_test222_1/'+str(n)+'.jpeg','JPEG')
        #                 map_view = get_agent_map_data(c)['frame']
        #                 map_view = Image.fromarray(map_view)
        #                 map_view.save('/home/jackliu/School/CISC856/MultiViewRL/data/image_test222_1/map'+str(n)+'.jpeg','JPEG')
        #                 n += 1
                

    def train_ego(self):
        # setup
        max_iteration = self.max_iteration
        episodes = self.episodes
        seed = self.seed
        dataPath = Path(self.data_path) / 'ego_view' / (self.scene+self.other_name)/ ("seed_"+str(seed)) / str(episodes)
        modelPath = Path(self.model_path) / 'ego_view' / (self.scene+self.other_name) / ("seed_"+str(seed)) / str(episodes)

        # make folder if not exist
        if not dataPath.exists():
            dataPath.mkdir(parents=True)
        if not modelPath.exists():
            modelPath.mkdir(parents=True)
        # logger and agent init
        logger = MetricLogger(modelPath)
        dqn_agent = DQNAgent(self.state_dim, len(self.actions),modelPath)
        
        with Controller(**self.controller_setting) as c:
            # initlaize agent
            event = c.step(action='InitialRandomSpawn', randomSeed=seed, forceVisible=True)
            agent_init = event.metadata["agent"]
            agent_position = event.metadata["agent"]['position']
            agent_horizon = event.metadata["agent"]['cameraHorizon']
            agent_horizon = self.agent_horizon
            agent_rotation = event.metadata["agent"]['rotation']
            # training
            for e in range(episodes):
                # initlize enviorment
                event = c.step(action="Teleport",  
                    position=agent_position, 
                    rotation=agent_rotation, 
                    horizon=agent_horizon, 
                    standing=True)
                state = self.extract_state(c, event)
                ego_state = state['ego']
                alo_state = state['alo']
                n = 0
                while True:                   
                    # get state tuple
                    action = dqn_agent.act(ego_state)
                    next_state, reward, done = self.env_step(c, action)
                    next_ego_state = next_state['ego']
                    next_alo_state = next_state['alo']
                    # memorize
                    dqn_agent.cache(ego_state,next_ego_state, action, reward, done)
                    # learn
                    q, loss = dqn_agent.learn()
                    # log
                    logger.log_step(reward, loss, q)
                    # prepare for next state
                    state = next_state
                    ego_state = state['ego']
                    alo_state = state['alo']

                    # break out if movemnt exceed max iteartion
                    n += 1
                    if done or n > max_iteration:
                        break                   
                
                #log info
                logger.log_episode()
                logger.record(episode=e, epsilon=dqn_agent.exploration_rate, step=dqn_agent.curr_step)

            # save last model
            dqn_agent.save_last()

    def train_alo(self):
        # setup
        max_iteration = self.max_iteration
        episodes = self.episodes
        seed = self.seed
        dataPath = Path(self.data_path) / 'alo_view' / (self.scene+self.other_name) / ("seed_"+str(seed)) / str(episodes)
        modelPath = Path(self.model_path) / 'alo_view' / (self.scene+self.other_name) / ("seed_"+str(seed)) / str(episodes)

        # make folder if not exist
        if not dataPath.exists():
            dataPath.mkdir(parents=True)
        if not modelPath.exists():
            modelPath.mkdir(parents=True)
        # logger and agent init
        logger = MetricLogger(modelPath)
        dqn_agent = DQNAgent(self.state_dim, len(self.actions),modelPath)
        
        
        with Controller(**self.controller_setting) as c:
            # initlaize agent
            event = c.step(action='InitialRandomSpawn', randomSeed=seed, forceVisible=True)
            agent_init = event.metadata["agent"]
            agent_position = event.metadata["agent"]['position']
            agent_horizon = event.metadata["agent"]['cameraHorizon']
            agent_horizon = self.agent_horizon
            agent_rotation = event.metadata["agent"]['rotation']
            # training
            for e in range(episodes):
                # initlize enviorment
                event = c.step(action="Teleport",  
                    position=agent_position, 
                    rotation=agent_rotation, 
                    horizon=agent_horizon, 
                    standing=True)
                state = self.extract_state(c, event)
                ego_state = state['ego']
                alo_state = state['alo']
                n = 0
                while True:                   
                    # get state tuple
                    action = dqn_agent.act(alo_state)
                    next_state, reward, done = self.env_step(c, action)
                    next_ego_state = next_state['ego']
                    next_alo_state = next_state['alo']
                    # memorize
                    dqn_agent.cache(alo_state,next_alo_state, action, reward, done)
                    # learn
                    q, loss = dqn_agent.learn()
                    # log
                    logger.log_step(reward, loss, q)
                    # prepare for next state
                    state = next_state
                    ego_state = state['ego']
                    alo_state = state['alo']

                    # break out if movemnt exceed max iteartion
                    n += 1
                    if done or n > max_iteration:
                        break                   
                
                #log info
                logger.log_episode()
                logger.record(episode=e, epsilon=dqn_agent.exploration_rate, step=dqn_agent.curr_step)

            # save last model
            dqn_agent.save_last()


    def train_double(self):
        # setup
        max_iteration = self.max_iteration
        episodes = self.episodes
        seed = self.seed
        dataPath = Path(self.data_path) / 'double_view' / (self.scene+self.other_name) / ("seed_"+str(seed)) / str(episodes)
        modelPath = Path(self.model_path) / 'double_view' / (self.scene+self.other_name) / ("seed_"+str(seed)) / str(episodes)

        # make folder if not exist
        if not dataPath.exists():
            dataPath.mkdir(parents=True)
        if not modelPath.exists():
            modelPath.mkdir(parents=True)
        # logger and agent init
        logger = MetricLogger(modelPath)
        dqn_agent = MVDQNAgent(self.state_dim, len(self.actions),modelPath)

        with Controller(**self.controller_setting) as c:
            # initlaize agent
            event = c.step(action='InitialRandomSpawn', randomSeed=seed, forceVisible=True)
            agent_init = event.metadata["agent"]
            agent_position = event.metadata["agent"]['position']
            agent_horizon = event.metadata["agent"]['cameraHorizon']
            agent_horizon = self.agent_horizon
            agent_rotation = event.metadata["agent"]['rotation']
            # training
            for e in range(episodes):
                # initlize enviorment
                event = c.step(action="Teleport",  
                    position=agent_position, 
                    rotation=agent_rotation, 
                    horizon=agent_horizon, 
                    standing=True)
                state = self.extract_state(c, event)
                ego_state = state['ego']
                alo_state = state['alo']
                n = 0
                while True:                   
                    # get state tuple
                    action = dqn_agent.act([ego_state, alo_state])
                    next_state, reward, done = self.env_step(c, action)
                    next_ego_state = next_state['ego']
                    next_alo_state = next_state['alo']
                    # memorize
                    dqn_agent.cache(ego_state, alo_state, next_ego_state,next_alo_state, action, reward, done)
                    # learn
                    q, loss = dqn_agent.learn()
                    # log
                    logger.log_step(reward, loss, q)
                    # prepare for next state
                    state = next_state
                    ego_state = state['ego']
                    alo_state = state['alo']
                    # break out if movemnt exceed max iteartion
                    n += 1
                    if done or n > max_iteration:
                        break                   
                #log info
                logger.log_episode()
                logger.record(episode=e, epsilon=dqn_agent.exploration_rate, step=dqn_agent.curr_step)
            # save last model
            dqn_agent.save_last()

            
    def test(self):

        # setup
        mode = "alo"
        model_path = '/home/jackliu/School/CISC856/MultiViewRL/model/alo_view/FloorPlan222/seed_11/100/last.chkpt'
        max_iteration = 50
        episodes = 1
        seed = self.seed
        dataPath = Path(self.data_path) / 'ego_view' / (self.scene+self.other_name)/ ("seed_"+str(seed)) / str(episodes)
        modelPath = Path(self.model_path) / 'ego_view' / (self.scene+self.other_name) / ("seed_"+str(seed)) / str(episodes)

        # load agent and model
        dqn_agent = DQNAgent(self.state_dim, len(self.actions),modelPath)
        dqn_agent.load(model_path)
        
        gif_array = []
        movie_array = []
        gif_location = '../alo_best_11.gif'
        movie_location = '../alo_best_11.mp4'


        with Controller(**self.controller_setting) as c:
            # initlaize agent
            event = c.step(action='InitialRandomSpawn', randomSeed=seed, forceVisible=True)
            agent_init = event.metadata["agent"]
            agent_position = event.metadata["agent"]['position']
            agent_horizon = event.metadata["agent"]['cameraHorizon']
            agent_horizon = self.agent_horizon
            agent_rotation = event.metadata["agent"]['rotation']

            for e in range(episodes):
                # initlize enviorment
                event = c.step(action="Teleport",  
                    position=agent_position, 
                    rotation=agent_rotation, 
                    horizon=agent_horizon, 
                    standing=True)
                state = self.extract_state(c, event)
                ego_state = state['ego']
                alo_state = state['alo']
                n = 0
                while True:                   
                    # get state tuple
                    if mode == "ego":
                        chosen_state = ego_state
                    elif mode == "alo":
                        chosen_state = alo_state
                    action = dqn_agent.act_no_grad(chosen_state)
                    next_state, reward, done = self.env_step(c, action)
                    ego_next_state = next_state['ego']
                    alo_next_state = next_state['alo']

                    ###### convert array to gif and movie
                    im_ego = np.transpose(chosen_state, (1, 2, 0)).numpy()
                    im_ego = transforms.ToPILImage()(np.uint8(im_ego))
                    gif_array.append(im_ego)
                    # movie_array.append(c.step(action="Done").frame)
                    movie_array.append(np.transpose(chosen_state, (1, 2, 0)).numpy())


                    # prepare for next state
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
                    if done or n > max_iteration:
                        imageio.mimsave(gif_location, gif_array)
                        clip = ImageSequenceClip(movie_array, fps=2)
                        clip.write_videofile(movie_location)
                        break






