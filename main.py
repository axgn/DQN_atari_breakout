import os

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import ale_py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, "save")
IMG_DIR = os.path.join(BASE_DIR, "img")


class BreakEnvWrapper(gym.Wrapper):
	def __init__(self, env, k, img_size=(84, 84)):
		super().__init__(env)
		self.k = k
		self.img_size = img_size
		obs_shape = env.observation_space.shape
		self.observation_space = gym.spaces.Box(
			low=0.0,
			high=1.0,
			shape=(k, img_size[0], img_size[1]),
			dtype=np.float32,
		)

	def _preprocess(self, state, th=0.26):
		state = np.array(
			Image.fromarray(state).resize(self.img_size, Image.BILINEAR)
		)
		state = state.astype(np.float64).mean(2) / 255.0
		return state

	def reset(self):
		state, _ = self.env.reset()
		state = self._preprocess(state)
		state = state[np.newaxis, ...].repeat(self.k, axis=0)
		return state

	def step(self, action):
		state_next = []
		info_list = []
		reward = 0.0
		ter = False
		tru = False
		for _ in range(self.k):
			if not ter:
				state_next_f, reward_f, ter_f, tru_f, info_f = self.env.step(action)
				state_next_f = self._preprocess(state_next_f)
				reward += reward_f
				ter = ter_f
				tru = tru or tru_f
				info_list.append(info_f)
			state_next.append(state_next_f[np.newaxis, ...])
		state_next = np.concatenate(state_next, 0)
		return state_next, reward, ter, tru, info_list


class QNet(nn.Module):
	def __init__(self, input_shape, n_actions):
		super().__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
			nn.ReLU(),
			nn.Conv2d(32, 64, kernel_size=4, stride=2),
			nn.ReLU(),
			nn.Conv2d(64, 64, kernel_size=3, stride=1),
			nn.ReLU(),
		)
		conv_out_size = self._get_conv_out(input_shape)

		self.fc = nn.Sequential(
			nn.Flatten(),
			nn.Linear(conv_out_size, 512),
			nn.ReLU(),
			nn.Linear(512, n_actions),
		)

	def _get_conv_out(self, shape):
		o = self.conv(torch.zeros(1, *shape))
		return int(np.prod(o.size()))

	def forward(self, x):
		conv_out = self.conv(x)
		out = self.fc(conv_out)
		return out


date_old = "50501"
date_new = "50501"


path_td = os.path.join(IMG_DIR, date_new, f"{date_new}_tdloss.png")
path_re = os.path.join(IMG_DIR, date_new, f"{date_new}_rw.png")
path_t_re = os.path.join(IMG_DIR, date_new, f"{date_new}_t_rw.png")

path_npy_old = os.path.join(SAVE_DIR, date_old)
path_npy_new = os.path.join(SAVE_DIR, date_new)
os.makedirs(path_npy_new, exist_ok=True)


class DeepQNetwork:
	def __init__(
		self,
		n_actions,
		input_shape,
		qnet,
		device,
		learning_rate=2e-4,
		reward_decay=0.99,
		replace_target_iter=5000,
		memory_size=100000,
		batch_size=32,
		png_size=100000,
	):
		self.losses = []
		self.aver_l = []
		self.rewards = []
		self.aver_r = []
		self.t_r = []

		self.n_actions = n_actions
		self.input_shape = input_shape
		self.lr = learning_rate
		self.gamma = reward_decay
		self.replace_target_iter = replace_target_iter
		self.memory_size = memory_size
		self.batch_size = batch_size
		self.device = device
		self.learn_step_counter = 0
		self.png_size = png_size
		self.init_memory()
		self.png_index = len(self.memory["aver_td"])

		self.qnet_eval = qnet(self.input_shape, self.n_actions).to(self.device)
		self.qnet_target = qnet(self.input_shape, self.n_actions).to(self.device)
		self.qnet_target.eval()
		self.optimizer = optim.RMSprop(self.qnet_eval.parameters(), lr=self.lr)

	def choose_action(self, state, epsilon=0.0):
		state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
		actions_value = self.qnet_eval.forward(state)
		if np.random.uniform(0, 1) > epsilon:
			action = torch.max(actions_value, 1)[1].data.cpu().numpy()[0]
		else:
			action = np.random.randint(0, self.n_actions)
		return int(action)

	def learn(self):
		if self.learn_step_counter % self.replace_target_iter == 0:
			self.qnet_target.load_state_dict(self.qnet_eval.state_dict())

		if self.memory_counter > self.memory_size:
			sample_index = np.random.choice(self.memory_size, size=self.batch_size)
		else:
			sample_index = np.random.choice(self.memory_counter, size=self.batch_size)

		b_s = torch.FloatTensor(self.memory["s"][sample_index]).to(self.device)
		b_a = torch.LongTensor(self.memory["a"][sample_index]).to(self.device)
		b_r = torch.FloatTensor(self.memory["r"][sample_index]).to(self.device)
		b_s_ = torch.FloatTensor(self.memory["s_"][sample_index]).to(self.device)
		b_d = torch.FloatTensor(self.memory["done"][sample_index]).to(self.device)

		q_curr_eval = self.qnet_eval(b_s).gather(1, b_a)
		q_next_target = self.qnet_target(b_s_).detach()

		q_next_eval = self.qnet_eval(b_s_).detach()
		next_state_values = q_next_target.gather(
			1, q_next_eval.max(1)[1].unsqueeze(1)
		)
		q_curr_recur = b_r + (1 - b_d) * self.gamma * next_state_values

		self.loss = F.smooth_l1_loss(q_curr_eval, q_curr_recur)
		self.optimizer.zero_grad()
		self.loss.backward()
		self.optimizer.step()
		self.learn_step_counter += 1
		self.losses.append(self.loss.detach().cpu().numpy())

		return self.loss.detach().cpu().numpy()

	def init_memory(self):
		if not hasattr(self, "memory"):
			self.memory = {
				"s": np.zeros((self.memory_size, *self.input_shape)),
				"a": np.zeros((self.memory_size, 1)),
				"r": np.zeros((self.memory_size, 1)),
				"s_": np.zeros((self.memory_size, *self.input_shape)),
				"done": np.zeros((self.memory_size, 1)),
				"aver_td": [],
				"aver_re": [],
				"total_re": [],
			}

	def store_transition(self, s, a, r, s_, d):
		if not hasattr(self, "memory_counter"):
			self.memory_counter = 0
		if self.memory_counter <= self.memory_size:
			index = self.memory_counter % self.memory_size
		else:
			index = np.random.randint(self.memory_size)

		self.memory["s"][index] = s
		self.memory["a"][index] = np.array(a).reshape(-1, 1)
		self.memory["r"][index] = np.array(r).reshape(-1, 1)
		self.memory["s_"][index] = s_
		self.memory["done"][index] = np.array(d).reshape(-1, 1)
		self.memory_counter += 1

		self.rewards.append(r)

	def store_t_r(self, total_reward):
		self.t_r.append(total_reward)
		if self.t_r:
			if len(self.memory["total_re"]) == 0:
				self.memory["total_re"] = np.array([self.t_r[-1]])
			else:
				self.memory["total_re"] = np.concatenate(
					[self.memory["total_re"], [self.t_r[-1]]]
				)

	def add_r_l(self):
		r_value = self.rewards[-100:]
		aver_rv = sum(r_value) / 100
		self.aver_r.append(aver_rv)

		l_value = self.losses[-100:]
		aver_lv = sum(l_value) / 100
		self.aver_l.append(aver_lv)

	def store_png(self):
		if not hasattr(self, "png_memory_counter"):
			self.png_memory_counter = 0
		if self.png_memory_counter <= self.png_size:
			self.png_index = self.png_memory_counter % self.png_size

		self.add_r_l()

		if self.png_index > self.png_memory_counter:
			self.memory["aver_td"] = []
			self.memory["aver_re"] = []
			if self.png_memory_counter > 0:
				self.png_index = self.png_index % self.png_memory_counter

		if self.aver_l:
			if len(self.memory["aver_td"]) == 0:
				self.memory["aver_td"] = np.array([self.aver_l[-1]])
			else:
				self.memory["aver_td"] = np.concatenate(
					[self.memory["aver_td"], [self.aver_l[-1]]]
				)
		if self.aver_r:
			if len(self.memory["aver_re"]) == 0:
				self.memory["aver_re"] = np.array([self.aver_r[-1]])
			else:
				self.memory["aver_re"] = np.concatenate(
					[self.memory["aver_re"], [self.aver_r[-1]]]
				)

		self.png_index += 1
		self.png_memory_counter += 1

	def save_load_model(self, op, path=SAVE_DIR, fname="qnet.pt"):
		os.makedirs(path, exist_ok=True)
		file_path = os.path.join(path, fname)
		if op == "save":
			torch.save(self.qnet_eval.state_dict(), file_path)
			os.makedirs(path_npy_new, exist_ok=True)
			if len(self.memory["aver_td"]):
				np.save(
					os.path.join(path_npy_new, f"aver_td_{date_new}.npy"),
					self.memory["aver_td"],
				)
			if len(self.memory["aver_re"]):
				np.save(
					os.path.join(path_npy_new, f"aver_re_{date_new}.npy"),
					self.memory["aver_re"],
				)
			if len(self.memory["total_re"]):
				np.save(
					os.path.join(path_npy_new, f"total_re_{date_new}.npy"),
					self.memory["total_re"],
				)
		elif op == "load":
			self.qnet_eval.load_state_dict(
				torch.load(file_path, map_location=self.device)
			)
			self.qnet_target.load_state_dict(
				torch.load(file_path, map_location=self.device)
			)
			aver_td_path = os.path.join(path_npy_old, f"aver_td_{date_old}.npy")
			aver_re_path = os.path.join(path_npy_old, f"aver_re_{date_old}.npy")
			total_re_path = os.path.join(path_npy_old, f"total_re_{date_old}.npy")
			if os.path.exists(aver_td_path):
				self.memory["aver_td"] = np.load(aver_td_path)
			if os.path.exists(aver_re_path):
				self.memory["aver_re"] = np.load(aver_re_path)
			if os.path.exists(total_re_path):
				self.memory["total_re"] = np.load(total_re_path)

	def plot_loss(self):
		if len(self.memory["aver_td"]) == 0:
			return
		os.makedirs(os.path.dirname(path_td), exist_ok=True)
		x_loss = np.arange(len(self.memory["aver_td"]))
		plt.plot(x_loss, self.memory["aver_td"], label="TD Loss")
		plt.xlabel("Iteration")
		plt.ylabel("Loss")
		plt.title("TD Loss over Training")
		plt.legend()
		plt.savefig(path_td)
		plt.close()

	def plot_rewards(self):
		if len(self.memory["aver_re"]) == 0:
			return
		os.makedirs(os.path.dirname(path_re), exist_ok=True)
		x_reward = np.arange(len(self.memory["aver_re"]))
		plt.plot(x_reward, self.memory["aver_re"], label="Average Reward")
		plt.xlabel("Episode")
		plt.ylabel("Average Reward")
		plt.title("Average Reward over Episodes")
		plt.legend()
		plt.savefig(path_re)
		plt.close()

	def plot_t_r(self):
		if len(self.memory["total_re"]) == 0:
			return
		os.makedirs(os.path.dirname(path_t_re), exist_ok=True)
		x_t_reward = np.arange(len(self.memory["total_re"]))
		plt.plot(x_t_reward, self.memory["total_re"], label="Score")
		plt.xlabel("Episode")
		plt.ylabel("Score")
		plt.title("Score over Episodes")
		plt.legend()
		plt.savefig(path_t_re)
		plt.close()


def play(env, agent, stack_frames, img_size):
	state = env.reset()
	img_buffer = [Image.fromarray(state[0] * 255)]

	step = 0
	total_reward = 0.0
	live_now = 5

	while True:
		if step == 0:
			action = 1
		else:
			action = agent.choose_action(state)

		state_next, reward, ter, tru, info = env.step(action)
		live_num = info[0].get("lives")

		if live_num != live_now:
			action = 1
			state_next, reward, ter, tru, info = env.step(action)
			live_now = live_num

		if step % 2 == 0:
			img_buffer.append(Image.fromarray(state_next[0] * 255))

		state = state_next.copy()
		step += 1
		total_reward += reward

		if ter or step > 40000:
			break

	return img_buffer


def save_gif(img_buffer, fname, gif_path=None):
	if gif_path is None:
		gif_path = os.path.join(IMG_DIR, "gif", date_new)
	os.makedirs(gif_path, exist_ok=True)
	img_buffer[0].save(
		os.path.join(gif_path, fname),
		save_all=True,
		append_images=img_buffer[1:],
		duration=1,
		loop=0,
	)


def epsilon_compute(frame_id, epsilon_max=1.0, epsilon_min=0.1, epsilon_decay=1_000_000):
	return epsilon_min + (epsilon_max - epsilon_min) * np.exp(-frame_id / epsilon_decay)


def train(env, agent, stack_frames, img_size, save_path=SAVE_DIR, max_steps=200_000):
	total_step = 0
	episode = 0
	max_total_reward = 0.0
	max_score = 0.0

	while True:
		state = env.reset()

		step = 0
		total_reward = 0.0
		loss = 0.0
		live_now = 5
		max_reward = 2.0
		score = 0.0

		while True:
			epsilon = epsilon_compute(total_step)
			action = agent.choose_action(state, epsilon)
			state_next, reward, ter, tru, info = env.step(action)
			live_num = info[0].get("lives")

			score += reward

			if max_reward < reward:
				max_reward = reward + 1

			if live_num != live_now:
				action = 1
				state_next, reward, ter, tru, info = env.step(action)
				live_now = live_num
				reward -= 1

			agent.store_transition(state, action, reward, state_next, ter)
			if total_step > 4 * agent.batch_size:
				loss = agent.learn()

			state = state_next.copy()
			step += 1
			total_step += 1
			total_reward += reward

			if total_step % 100 == 0:
				agent.store_png()
			if ter:
				agent.store_t_r(score)
				if total_reward > max_total_reward:
					max_total_reward = total_reward

			if ter:
				break

			if total_step % 10_000 == 0:
				num = (total_step % 100_000) // 10_000
				agent.save_load_model(
					op="save",
					path=save_path,
					fname=f"qnet_{date_new}_{num}.pt",
				)
				img_buffer = play(env, agent, stack_frames, img_size)
				save_gif(img_buffer, f"train_{date_new}_{total_step:06d}.gif")
				agent.plot_loss()
				agent.plot_t_r()

			if ter or step > 2000:
				episode += 1
				break

		if max_score < score:
			max_score = score

		if total_step > max_steps:
			break


def create_agent(env_break, device):
	stack_frames = 4
	img_size = (84, 84)
	agent = DeepQNetwork(
		n_actions=env_break.action_space.n,
		input_shape=[stack_frames, *img_size],
		qnet=QNet,
		device=device,
		learning_rate=2e-4,
		reward_decay=0.99,
		replace_target_iter=5000,
		memory_size=100000,
		batch_size=32,
	)
	return agent, stack_frames, img_size


def main():
	gym.register_envs(ale_py)
	env_name = "ALE/Breakout-v5"
	env = gym.make(env_name)
	env_break = BreakEnvWrapper(env, k=4, img_size=(84, 84))
	print("=== raw env ===")
	print("env:", env)
	print("action_space:", env.action_space)
	print("observation_space:", env.observation_space)
	print("action_meanings:", env.unwrapped.get_action_meanings())
	print("\n=== wrapped env_break ===")
	print("env_break:", env_break)
	print("action_space:", env_break.action_space)
	print("observation_space:", env_break.observation_space)
    # 看一下 reset 后的状态形状
	state_raw, _ = env.reset()
	state_wrapped = env_break.reset()
	print("raw state shape:", state_raw.shape)
	print("wrapped state shape:", state_wrapped.shape)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	agent, stack_frames, img_size = create_agent(env_break, device)
	print(agent.qnet_eval)

	# train(env_break, agent, stack_frames, img_size, save_path=SAVE_DIR, max_steps=200_000)


if __name__ == "__main__":
	main()

