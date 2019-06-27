#libaries importieren
import numpy as np

def calc_state(env):
	state = 0
	for i in range(3):
		for j in range(3):
			state += (3**3)**i * 3**j * env.playField[i][j]
	return state

class RLPlayer:
	def __init__(self):
		self.actionspace = 9
		self.observationspace = 3**9
		self.qtable = np.zeros((self.observationspace, self.actionspace))

	def train(self, episodes=50000, alpha=0.1, epsilon=0.8, gamma=0.6):
		env = TicTacToe()
		for i in range(episodes):
			env.restart()
			state = calc_state(env) #state 0
			done = False
			while not done:
				if np.random.rand() < epsilon:
					action = np.random.randint(self.actionspace)
				else:
					action = np.argmax(self.qtable[state])
				x, y = action // 3, action % 3
				env.place(x, y)
				next_state = calc_state(env)
				if env.player_won() == 1:
					done = True
					reward = -1 #loss
				elif env.player_won() == 2:
					done = True
					reward = 1 #win
				elif env.player_won() == 3:
					 done = True
					 reward = 0 #draw
				else: 
					reward = 0
				qtable[state, action] = (1-alpha)*qtable[state, action] + alpha*(reward + gamma*np.argmax(qtable(next_state)))


