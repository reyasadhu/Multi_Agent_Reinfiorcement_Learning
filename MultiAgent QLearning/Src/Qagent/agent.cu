/*************************************************************************
/* ECE 277: GPU Programmming 2021 WINTER quarter
/* Author and Instructor: Cheolhong An
/* Copyright 2019
/* University of California, San Diego
/*************************************************************************/
#include <cuda_fp16.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <curand.h>
#include <curand_kernel.h>

float epsilon;
short *d_action;
curandState* state;
float *d_epsilon;
float *d_qtable;
bool *d_active;

#define rows 32
#define cols 32
#define num_agents 128

#define gamma 0.9
#define alpha 0.1
#define delta_eps 0.001

__global__ void Agent_init_kernel(curandState* state, short* d_action, bool* d_active)
{
	
	int agent_id = threadIdx.x + blockIdx.x * blockDim.x;
	curand_init(clock() + agent_id, agent_id, 0, &state[agent_id]);
	d_active[agent_id] = 1;
	d_action[agent_id] = -0.500f;
}

__global__ void Qtable_init_kernel(float* d_qtable)
{
	int ix = threadIdx.x + blockDim.x * blockIdx.x;
	int iy = threadIdx.y + blockDim.y * blockIdx.y;
	int tid = iy * cols * 4 + ix;
	d_qtable[tid] = 0;
}
__global__ void Epsilon_init_kernel(float* d_epsilon)
{
	*d_epsilon = 1.0f;
}

void agent_init()
{
	// add your codes
	int qsize = 4 * rows * cols;
	cudaMalloc((void**)&d_qtable, sizeof(float)*qsize);
	//dim3 grid(rows, cols);
	//dim3 block(4);
	Qtable_init_kernel << <rows*cols, 4 >> > (d_qtable);

	cudaMalloc((void**)&state, sizeof(curandState) * num_agents);
	cudaMalloc((void**)&d_action, sizeof(short)*num_agents);
	cudaMalloc((void**)&d_active, sizeof(bool) * num_agents);
	Agent_init_kernel << <1, num_agents >> > (state, d_action, d_active);

	cudaMalloc((void**)&d_epsilon, sizeof(float));
	Epsilon_init_kernel << <1, 1 >> > (d_epsilon);
}

__global__ void agent_init_episode_kernel(bool* d_active)
{
	int agent_id= threadIdx.x + blockIdx.x * blockDim.x;
	d_active[agent_id] = 1;
}

void agent_init_episode() 
{
	// add your codes
	agent_init_episode_kernel << <1, num_agents >> > (d_active);

}


__global__ void adjust_epsilon_kernel(float* d_epsilon) {

	*d_epsilon -= delta_eps;
	if (*d_epsilon > 1.000f) {
		*d_epsilon = 1.000f;
	}
	else if (*d_epsilon < 0.100f) {
		*d_epsilon = 0.100f;
	}

}

float agent_adjustepsilon() 
{
	// add your codes
	adjust_epsilon_kernel << <1, 1 >> > (d_epsilon);
	cudaMemcpy(&epsilon, d_epsilon, sizeof(float), cudaMemcpyDeviceToHost);
	return epsilon;

	
}

__global__ void agent_action_kernel(int2* cstate, short* d_action, curandState* state, float* d_qtable, float* d_epsilon,bool* d_active)
{

	int agent_id = threadIdx.x + blockIdx.x * blockDim.x;
	if (d_active[agent_id] == 1) 
	{
		int c_x = cstate[agent_id].x;
		int c_y = cstate[agent_id].y;
		int action;
		float rand_state = curand_uniform(&state[agent_id]);

		if (rand_state < *d_epsilon) 
		{
			//Exploration
			action = (int)(curand_uniform(&state[agent_id]) * 4);
			if (action == 4) action = 0;
		}
		else 
		{
			//Exploitation
			int q_idx = (c_y * cols + c_x) * 4;
			float qval_max = d_qtable[q_idx];
			action = 0;
			for (int i = 1; i < 4; i++) 
			{
				if (d_qtable[q_idx] + i > qval_max) 
				{
					qval_max = d_qtable[q_idx + i];
					action = i;
				}
			}
		}

		d_action[agent_id] = (short)action;
	}
}
short* agent_action(int2* cstate)
{
	// add your codes
	agent_action_kernel << <1, num_agents >> > (cstate, d_action, state, d_qtable, d_epsilon, d_active);
	return d_action;
}

__global__ void agent_update_kernel(int2* cstate, int2* nstate, float* rewards, float* d_qtable, short* d_action, bool* d_active)
{
	int agent_id = threadIdx.x + blockIdx.x * blockDim.x;
	if (d_active[agent_id] == 1)
	{

		int c_x = cstate[agent_id].x;
		int c_y = cstate[agent_id].y;
		int n_x = nstate[agent_id].x;
		int n_y = nstate[agent_id].y;


		// To find max Q(S_n+1,a')
		int n_q_idx = (n_y * cols + n_x) * 4;
		float qval_max = d_qtable[n_q_idx];
		for (int i = 1; i < 4; i++) {
			if (d_qtable[n_q_idx + i] > qval_max) {
				qval_max = d_qtable[n_q_idx + i];
			}
		}

		//Update Q(S_n,a_n)
		int c_q_idx = (c_y * cols * c_x) + (int)d_action[agent_id];
		

		if (rewards[agent_id] == 0)
		{
			d_qtable[c_q_idx] += alpha * (rewards[agent_id] + gamma * qval_max - d_qtable[c_q_idx]);
			cstate[agent_id] = nstate[agent_id];
		}
		else
		{
			// agents becoming inactive
			d_qtable[c_q_idx] += alpha * (rewards[agent_id] - d_qtable[c_q_idx]);
			cstate[agent_id].x = 0;
			cstate[agent_id].y = 0;
			d_active[agent_id] = 0;
		}
	}
}

void agent_update(int2* cstate, int2* nstate, float* rewards)
{
	// add your codes
	agent_update_kernel << <1, num_agents >> > (cstate, nstate, rewards, d_qtable, d_action,d_active);

}



