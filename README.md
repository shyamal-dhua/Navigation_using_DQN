# Navigation_using_DQN

### Introduction

For this project, we will train an agent to navigate (and collect bananas!) in a large, square world using `Deep Q Learning Algorithm`.

<img src="images/description.gif">

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of our agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, our agent must get an average score of +13 over 100 consecutive episodes.

### Dependencies

To set up your python environment to run the code in this repository, follow the instructions below. Please note that we will be using `Anaconda`
for executing the code. 

1. The project environment is similar to, but not identical to the Banana Collector environment on the [Unity ML-Agents GitHub page](https://github.com/Unity-Technologies/ml-agents/blob/main/docs/Learning-Environment-Examples.md#banana-collector). So first we need to download the Unity environment from one of the links below based on the operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
2. Enter Anaconda terminal and create (and activate) a new environment with Python 3.6.

	- __Linux__ or __Mac__: 
	```bash
	conda create --name drlnd python=3.6
	source activate drlnd
	```
	- __Windows__: 
	```bash
	conda create --name drlnd python=3.6 
	conda activate drlnd
	```
	  
3. Perform a minimal install of OpenAI gym.
  ```bash
  pip install gym
  ```

4. Clone this repository, and navigate to the `python/` folder.  Then, install several dependencies.
  ```bash
  cd Navigation_using_DQN/python
  pip install .
  ```

5. Download whl file for `torch 0.4.0` from `http://download.pytorch.org/whl/cpu/torch-0.4.0-cp36-cp36m-win_amd64.whl`. Then install it using the command below. Also install `pywinpty` using `pipwin`, then install `jupyter` [Note: during installation of `jupyter` it tries to install a higher version of `pywinpty` and can give some error which we can ignore.]
  ```bash
  pip install --no-deps path_to_torch_whl_file\torch-0.4.0-cp36-cp36m-win_amd64.whl
  pip install pipwin
  pipwin install pywinpty
  pip install jupyter
  ```
6. Create an IPython kernel for the drlnd environment. We can use this kernel if we use `jupyter notebook` to run. Before running code in a notebook, change the kernel to match the `drlnd` environment by using the drop-down `Kernel` menu.
  ```bash
  python -m ipykernel install --user --name drlnd --display-name "drlnd"
  ```

7. Extract the previously downloaded zip file for Environment [in step 1], to the root folder i.e. `Navigation_using_DQN/`.

8. We need to run `main.py` for both training and testing. Follow the below instruction for training. We can set the total number of episodes, iterations per episode, starting value of epsilon, minimum value of epsilon and epsilon decay [for epsilon-greedy  algorithm] during the training through command line. 
  ```bash
  python main.py --n_episodes 2000 --max_t 1000 --eps_start 1 --eps_end 0.01 --eps_decay 0.995
  ```

9. Similarly follow the below instruction for testing. We need to provide a checkpoint file for this. 
  ```bash
  python main.py --evaluate checkpoints/best_checkpoint.pth
  ```
  
10. To get better understanding on the available parameters to be passed in command line, use the help function as below, 
  ```bash
  python main.py -h
  ```

11. The model parameters get stored in the folder `Navigation_using_DQN/checkpoints/`. During training we store the best checkpoints in `best_checkpoint.pth` and we 
also store the intermediate checkpoints [all scoring > 13.0] in same folder using timestamp as well as score appended to its file name. The plots for the training get stored in `Navigation_using_DQN/plots/`. 

12. A sample terminal output logs is present in `Navigation_using_DQN/terminal_ouput_logs.txt`.

13. Below is how the agent was acting without training. The agent takes random actions. 
<img src="images/untrained.gif">

14. After training, it looks as below. The agent has learnt to pick the yellow bananas while avoiding the blue ones. 
<img src="images/trained.gif">



