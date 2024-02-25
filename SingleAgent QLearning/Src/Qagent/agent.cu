/*************************************************************************
/* ECE 277: GPU Programmming 2021 WINTER
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
float* d_epsilon;
float* d_qtable;

#define rows 4
#define cols 4

#define gamma 0.9
#define alpha 0.1
#define delta_eps 0.001

__global__ void Agent_init_kernel(float* d_epsilon, short* d_action)
{
	*d_epsilon = 1.000f;
	*d_action = -0.500f;
}

__global__ void Qtable_init_kernel(float* d_qtable)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	d_qtable[tid] = 0;
}
__global__ void init_randstate(curandState* state)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	curand_init(clock() + tid, tid, 0, &state[tid]);
}
void agent_init()
{
	// add your codes
	int qsize = 4 * rows * cols;
	epsilon = 1.000f;

	cudaMalloc((void**)&d_qtable, sizeof(float) * qsize);
	cudaMalloc((void**)&d_action, sizeof(short));
	cudaMalloc((void**)&d_epsilon, sizeof(float));
	cudaMalloc((void**)&state, sizeof(curandState));

	Agent_init_kernel << <1, 1 >> > (d_epsilon,d_action);
	Qtable_init_kernel << <rows * cols, 4 >> > (d_qtable);
	init_randstate << <1, 1 >> > (state);

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
	adjust_epsilon_kernel<< <1, 1 >> > (d_epsilon);
	cudaMemcpy(&epsilon, d_epsilon, sizeof(float), cudaMemcpyDeviceToHost);
	return epsilon;
}

__global__ void agent_action_kernel(int2* cstate,short* d_action, curandState* state, float* d_qtable, float* d_epsilon) 
{
	
	int agent_id = 0;
	int c_x = cstate[agent_id].x;
	int c_y = cstate[agent_id].y;
	int action;
	float rand_state = curand_uniform(&state[agent_id]);

	if (rand_state < *d_epsilon) {
	//Exploration
		action = (int)(curand_uniform(&state[agent_id]) * 4);
	}
	else {
		//exploitation
		int q_idx = (c_y * cols + c_x) * 4;
		float qval_max = d_qtable[q_idx];
		action = 0;
		for (int i = 1; i < 4; i++) {
			if (d_qtable[q_idx] + i > qval_max) {
				qval_max = d_qtable[q_idx + i];
				action = i;
			}
		}
	}

	*d_action = (short)action;
}
short* agent_action(int2* cstate)
{
	// add your codes
	agent_action_kernel << <1, 1 >> > (cstate,d_action,state,d_qtable,d_epsilon);
	return d_action;
}

__global__ void agent_update_kernel(int2* cstate, int2* nstate, float* rewards, float* d_qtable, short* d_action) 
{
	int agent_id = 0;
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
	int c_q_idx = (c_y * cols * c_x) + *d_action;
	d_qtable[c_q_idx] += alpha * (rewards[agent_id]+gamma * qval_max - d_qtable[c_q_idx]);

	if (rewards[agent_id] == 0) 
	{
		cstate[agent_id] = nstate[agent_id];
	}
	else
	{
		// agents becoming inactive
		cstate[agent_id].x = 0;
		cstate[agent_id].y = 0;
	}
}

void agent_update(int2* cstate, int2* nstate, float *rewards)
{
	// add your codes
	agent_update_kernel << <1, 1 >> > (cstate, nstate, rewards, d_qtable, d_action);

}
