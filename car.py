# pip install gym
'''

Grupo: Beatriz dos Santos Nunes, Enzo Fadini Siqueira, Davi Coswosk de Oliveira, Raphaela Venturim e Willian James Santos de Souza

'''

# 1. Dependencias:
import gym
import numpy as np

# 2. Ambiente
# Ambiente: https://gym.openai.com/envs/MountainCar-v0/
# fonte: https://github.com/openai/gym/blob/master/gym/envs/classic_control/mountain_car.py

#2.1 Carga
env = gym.make("MountainCar-v0")

# 2.2 Descrição do ambiente
# The agent (a car) is started at the bottom of a valley.
# For any given state the agent may choose to accelerate to the left, right or cease any acceleration.

# 2.3 Entendendo o ambiente

#    2.3.1 Estado Inicial:
#    The position of the car is assigned a uniform random value in [-0.6 , -0.4].
#    The starting velocity of the car is always assigned to 0.
print(env.reset())

#    2.3.2 Ações:
print(env.action_space)

#        Type: Discrete(3)
#        Num    Action
#        0      Accelerate to the Left
#        1      Don't accelerate
#        2      Accelerate to the Right

#    2.3.2 Space observations:
print(env.observation_space.high)
print(env.observation_space.low)

#        Type: Box(2)
#        Num    Observation               Min            Max
#        0      Car Position              -1.2           0.6
#        1      Car Velocity              -0.07          0.07


#    2.3.3 Reward:
#        Reward of 0 is awarded if the agent reached the flag (position = 0.5) on top of the mountain.
#        Reward of -1 is awarded if the position of the agent is less than 0.5.

# 3. QLearning
LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 2_000
SHOW_EVERY = 2_00
DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE
q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))


def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(np.int64))


for episode in range(EPISODES):
    if episode % SHOW_EVERY == 0:
        print(episode)
        render = True
    else:
        render = False
    discrete_state = get_discrete_state(env.reset())
    done = False
    while not done:
        action = np.argmax(q_table[discrete_state])
        new_state, reward, done, _ = env.step(action)
        new_discrete_state = get_discrete_state(new_state)
        if render:
            env.render()
        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action,)]
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[discrete_state + (action,)] = new_q
        elif new_state[0] >= env.goal_position:
            print(f"We made it at {episode}")
            q_table[discrete_state + (action,)] = 0
        discrete_state = new_discrete_state
env.close()


# 4 - Referências
'''
AQ-Learning introduction and Q Table - Reinforcement Learning w/ Python Tutorial p.1. Pythonprogramming, 2019.
Disponível em: < https://pythonprogramming.net/q-learning-reinforcement-learning-python-tutorial/ >.
Acesso em: 04 de nov. de 2021.

AQ-Learning introduction and Q Table - Reinforcement Learning w/ Python Tutorial p.2. Pythonprogramming, 2019.
Disponível em: < https://pythonprogramming.net/q-learning-algorithm-reinforcement-learning-python-tutorial/?completed=/q-learning-reinforcement-learning-python-tutorial/ >.
Acesso em: 05 de nov. de 2021.
'''
