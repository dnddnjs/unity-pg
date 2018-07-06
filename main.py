import torch
import argparse
import numpy as np
import torch.optim as optim
from model import Actor, Critic
from utils import get_action
from collections import deque
from running_state import ZFilter
from hparams import HyperParams as hp


parser = argparse.ArgumentParser()
parser.add_argument('--algorithm', type=str, default='PPO',
                    help='select one of algorithms among Vanilla_PG, NPG, TPRO, PPO')
parser.add_argument('--env', type=str, default="Hopper-v2",
                    help='name of Mujoco environement')
parser.add_argument('--render', default=False)
args = parser.parse_args()

if args.algorithm == "PG":
    from vanila_pg import train_model
elif args.algorithm == "NPG":
    from npg import train_model
elif args.algorithm == "TRPO":
    from trpo import train_model
elif args.algorithm == "PPO":
    from ppo import train_model


if __name__=="__main__":
    env_name = "./Env/walker2"
    train_mode = True
    torch.manual_seed(500)

    from unityagents import UnityEnvironment

    env = UnityEnvironment(file_name=env_name)

    default_brain = env.brain_names[0]
    brain = env.brains[default_brain]

    num_inputs = brain.vector_observation_space_size
    num_actions = brain.vector_action_space_size

    print('state size:', num_inputs)
    print('action size:', num_actions)

    actor = Actor(num_inputs, num_actions)
    critic = Critic(num_inputs)
    torch.load('save_model/actor')
    torch.load('save_model/critic')

    actor_optim = optim.Adam(actor.parameters(), lr=hp.actor_lr)
    critic_optim = optim.Adam(critic.parameters(), lr=hp.critic_lr,
                              weight_decay=hp.l2_rate)

    running_state = ZFilter((num_inputs,), clip=5)
    episodes = 0
    for iter in range(1000):
        actor.eval(), critic.eval()
        memory = deque()

        steps = 0
        scores = []
        while steps < 2048:
            episodes += 1
            env_info = env.reset(train_mode=train_mode)[default_brain]
            state = env_info.vector_observations[0]
            state = running_state(state)
            score = 0
            for _ in range(10000):
                steps += 1
                mu, std, _ = actor(torch.Tensor(state).unsqueeze(0))
                action = get_action(mu, std)[0]
                actions = np.zeros([len(env_info.agents), num_actions])
                actions[0] = action

                env_info = env.step(actions)[default_brain]
                next_state = env_info.vector_observations[0]
                reward = env_info.rewards[0]
                done = env_info.local_done[0]

                next_state = running_state(next_state)

                if done:
                    mask = 0
                else:
                    mask = 1

                memory.append([state, action, reward, mask])

                score += reward
                state = next_state

                if done:
                    break

            scores.append(score)
        score_avg = np.mean(scores)
        print('{} episode score is {:.2f}'.format(episodes, score_avg))
        actor.train(), critic.train()
        train_model(actor, critic, memory, actor_optim, critic_optim)
        if iter % 10:
            torch.save(actor, 'save_model/actor')
            torch.save(critic, 'save_model/critic')

    env.close()