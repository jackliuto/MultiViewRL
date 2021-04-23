# Object Navigation in a 3D Rendered Home Environment with Multi-View Reinforcement Learning
CISC 856 Final Project
Group 27


## Example of a trained agent
![ego_best_1](https://user-images.githubusercontent.com/14166685/115815051-4155ed80-a3c4-11eb-964d-9fee9e6e31fa.gif)
![alo_best_1](https://user-images.githubusercontent.com/14166685/115814969-18cdf380-a3c4-11eb-91e0-06e6e0f59b76.gif)

## Intsructions
To train Egocentric Agent:
```
python src/main.py --mode train_ego
```
To train Allocentric Agent:
```
python src/main.py --mode train_alo
```
To train Double Agent:
```
python src/main.py --mode train_double
```
To test a policy:
```
python src/main.py --mode test
```
To output saliency maps:
```
python visuals/saliency_map.py
```

The repository contains the following:
- _src_ contains the necessary python files used to train the algorithm. Follow the instructions above to run. 
  -  agent.py contains DQN agent class
  -  network.py contrains all Neuarl Network class
  -  main.py contains a commandline program for our code
  -  solve.py contrains the traing and enviorment code for AI2-THOR
- _model_samples_ contains model checkpoints for the easy and hard conditions each trained on egocentric only, allocentric only and combined perspective.
- _visuals_ contains the necessary files to generate saliency maps using the model checkpoints 



