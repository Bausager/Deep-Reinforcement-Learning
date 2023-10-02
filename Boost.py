import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

import NNFS



class Boost(object):
	"""docstring for Boost"""
	def __init__(self, *, d=0.6, f=10e3, samples=250, store_periods=100):
		self.d = d
		self.f = f
		self.Ts = 1/f
		self.dt = self.Ts/samples
		self.store_periods = store_periods

		self.timeseries = np.zeros([self.store_periods])
		self.timeseries_temp = np.zeros([self.store_periods])

		self.rewards = np.zeros([self.store_periods])
		self.rewards_temp = np.zeros([self.store_periods])

		self.states = np.zeros([self.store_periods])
		self.states_temp = np.zeros([self.store_periods])



		self.time = 0		
		self.state = 0
		self.x0 = np.array([[0], [0]])
		self.x0_dot = np.array([[0], [0]])

		self.state_space_equations()

		#%%
		# Instantiate the model
		self.model = NNFS.Model(N_Step_TD=50, N_actions=2, gamma = 0.9)
		# Input Layer
		self.model.add(NNFS.Layer_Dense(6, 100, weight_regularizer_L1=5e-5, bias_regularizer_L1=5e-5))
		self.model.add(NNFS.Activation_ReLU())
		# First Hidden Layer
		self.model.add(NNFS.Layer_Dense(100, 100, weight_regularizer_L1=5e-5, bias_regularizer_L1=5e-5))
		self.model.add(NNFS.Activation_ReLU())
		# Second Hidden Layer
		self.model.add(NNFS.Layer_Dense(100, 100, weight_regularizer_L1=5e-5, bias_regularizer_L1=5e-5))
		self.model.add(NNFS.Activation_ReLU())
		# Output Layer
		self.model.add(NNFS.Layer_Dense(100, 2))
		self.model.add(NNFS.Activation_Linear())
		# Set loss, optimizer and accuracy objects
		self.model.set(
		    loss=NNFS.Loss_MeanSquredError(),
		    optimizer=NNFS.Optimizer_Adam(learning_rate=0.0001, decay=1e-3),
		    accuracy=NNFS.Accuracy_Regression()
		)
		# Finalize the model
		self.model.finalize()



	def state_space_equations(self, *, L=1e-3, C=470.0E-9, Vin=10, Rout=20):

		self.L = L
		self.C = C
		self.Vin = Vin
		self.Rout = Rout

		# -------- State Space Description ---------

		# Interval I (D)
		self.A1 = np.array([[0, 0], [0, -1/(self.Rout*self.C)]])       	
		self.B1 = np.array([[1/self.L, 0], [0, 0]]) 

		# Interval II (Dp)
		self.A2 = np.array([[0, -1/self.L], [1/self.C, -1/(self.C*self.Rout)]]) 	
		self.B2 = np.array([[1/self.L, 0], [0, 0]]) 

		# Both Intervals
		self.U =  np.array([[self.Vin], [0]]) 
		self.C = np.array([[1, 0], [0, 1], [0, 1/self.Rout]])                       
		self.D = np.array([[0, 0], [0, 0], [0, 0]]) 
		# ------------------------------------------

		self.y = np.zeros([self.C[:,0].shape[0], self.store_periods])
		self.y_temp = np.zeros([self.C[:,0].shape[0], self.store_periods])

	def step(self, v_ref):

		
		self.time += self.dt

		if((self.time % self.Ts) < ((self.time-self.dt) % self.Ts)):
			self.control(v_ref=v_ref)


		# Interval I (D)
		      
		#if(self.state):  
		if((self.time % self.Ts) < (self.Ts*self.d)):
			dxdt = np.add(np.matmul(self.A1,self.x0), np.matmul(self.B1, self.U))
		# Interval II (Dp)
		else:
			dxdt = np.add(np.matmul(self.A2,self.x0), np.matmul(self.B2, self.U))


		# Update for next iteration.
		self.x0 = dxdt*self.dt + self.x0 

	def control(self, v_ref):

		reward = (np.abs(v_ref - self.x0_dot[1]) - np.abs(v_ref - self.x0[1])) - (np.abs(v_ref - self.x0[1]))

		states = [self.x0[0,0], self.x0[1,0], self.x0_dot[0,0], self.x0_dot[1,0], v_ref, self.Vin]
		self.state = self.model.RL_train(X=states, reward=reward)
		if self.state:
			self.d += 0.001
		else:
			self.d -= 0.001

		if self.d < 0.05:
			self.d = 0.05
		elif self.d >= 0.95:
			self.d = 0.95


		#self.state = -1
		self.x0_dot = self.x0

		self.rewards_temp[:-1] = self.rewards[1:]
		self.rewards_temp[-1] = reward[0]
		self.rewards = self.rewards_temp

		self.y_temp[:,:-1] = self.y_temp[:,1:]
		self.y_temp[:,-1] = np.matmul(self.C, self.x0).flatten(order='F')
		self.y = self.y_temp

		self.states_temp[:-1] = self.states[1:]
		self.states_temp[-1] = self.d
		self.states = self.states_temp

		self.timeseries_temp[:-1] = self.timeseries[1:]
		self.timeseries_temp[-1] = self.time
		self.timeseries = self.timeseries_temp

		



if __name__ == "__main__":

	converter = Boost()
	v_ref = 30
	v_in = 10
	R_out = 20


	try:
		print(f'Loading Model 1')
		converter.model = NNFS.Model.load('RL1.model')
	except:
		print(f'Loading Model 2')
		converter.model = NNFS.Model.load('RL2.model')

	

	converter.state_space_equations(Vin=v_in, Rout=R_out)
	for x in tqdm(range(int(100e3))):
		converter.step(v_ref=v_ref)

	#converter.model.save('RL1.model')
	#converter.model.save('RL2.model')


#	if np.random.rand() < 0.01:
#		i = np.random.rand()
#		if  i > 0.6:
#			v_in = np.random.uniform(low=1, high=40)
#			if v_ref < v_in:
#				v_ref = np.random.uniform(low=v_in, high=v_in*3)
#		elif i > 0.3:
#			v_ref = np.random.uniform(low=v_in, high=v_in*3)
#		else:
#			R_out = np.random.uniform(low=v_in, high=v_ref)

	plt.close()

	fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(9, 8), sharex=True)
	#fig.suptitle(f'V_in: {round(v_in,2)}, v_ref: {round(v_ref,2)}, R: {round(R_out,2)}')
	axs[0,0].plot(converter.timeseries, converter.y[1,:], label=r"$V_{out}$" + f' (Avg. {np.round(np.mean(converter.y[1,:]),2)}V)', color = 'k')
	axs[0,0].axhline(y = (v_ref), color = 'k', linestyle = '--', label=r"$v_{ref}$")
	axs[0,0].set_ylabel(f"Output Voltage [V]")
	axs[0,0].set_ylim([0, np.max(converter.y[1,:]) * 1.1])
	axs[0,0].grid()
	axs[0,0].legend(loc='lower right')

	axs[1,0].plot(converter.timeseries, converter.rewards, label=r'$R_{t}$' + f' (Avg. {np.round(np.mean(converter.rewards),2)})', color = 'k')
	axs[1,0].set_ylabel(f"Reward")
	axs[1,0].grid()
	axs[1,0].set_xlabel("Time [s]")
	axs[1,0].legend(loc='lower right')

	axs[0,1].plot(converter.timeseries, converter.states, marker="o", linestyle = 'None' , markersize=4, color = 'k', label=r'$\delta$' + f' (Avg. {np.round(np.mean(converter.states),2)})')
	axs[0,1].set_ylabel(f'Duty Cycle [%]')
	axs[0,1].set_ylim([-0.01, 1.01])
	axs[0,1].grid()
	axs[0,1].legend(loc='lower right')

	axs[1,1].plot(converter.timeseries, converter.y[2,:], color = 'k' , label=r'$I_{L}$' + f' (Avg. {np.round(np.mean(converter.y[2,:]),2)})')
	axs[1,1].set_ylabel(f"Inductor Current [I]")
	axs[1,1].grid()
	axs[1,1].set_xlabel("Time [s]")
	axs[1,1].legend(loc='lower right')

	fig.tight_layout()

	plt.savefig("Everything.png")



	plt.figure(figsize=(5, 5))
	plt.plot(converter.timeseries, converter.y[1,:], label=r"$V_{out}$" + f' (Avg. {np.round(np.mean(converter.y[1,:]),2)}V)', color = 'k')
	plt.axhline(y = (v_ref), color = 'k', linestyle = '--', label=r"$V_{ref}$")
	plt.ylabel(f"Output Voltage [V]")
	plt.xlabel("Time [s]")
	plt.ylim([0, np.max(converter.y[1,:]) * 1.1])
	plt.grid()
	plt.legend(loc='lower right')
	fig.tight_layout()
	plt.savefig("Voltage.png")

	plt.figure(figsize=(5, 5))
	plt.plot(converter.timeseries, converter.rewards, label=r'$R_{t}$' + f' (Avg. {np.round(np.mean(converter.rewards),2)})', color = 'k')
	plt.ylabel(f"Reward")
	plt.xlabel("Time [s]")
	plt.grid()
	plt.legend(loc='lower right')
	fig.tight_layout()
	plt.savefig("Reward.png")

	plt.figure(figsize=(5, 5))
	plt.plot(converter.timeseries, converter.states, marker="o", linestyle='None', markersize=4, color = 'k', label=r'$\delta$' + f' (Avg. {np.round(np.mean(converter.states),2)})')
	plt.ylabel(f"Duty Cycle")
	plt.xlabel("Time [s]")
	plt.ylim([-0.01, 1.01])
	plt.grid()
	plt.legend(loc='lower right')
	fig.tight_layout()
	plt.savefig("Duty Cycle.png")

	plt.figure(figsize=(5, 5))
	plt.plot(converter.timeseries, converter.y[2,:], color = 'k', label=r'$I_{L}$' + f' (Avg. {np.round(np.mean(converter.y[2,:]),2)})')
	plt.ylabel(f"Inductor Current [I]")
	plt.xlabel("Time [s]")
	plt.grid()
	plt.legend(loc='lower right')
	fig.tight_layout()
	plt.savefig("Inductor Current.png")

