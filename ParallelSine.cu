//
// Assignment 1: ParallelSine
// CSCI 415: Networking and Parallel Computation
// Spring 2017
// Name(s): Ben Hapip, Damon Hage, Thomas Ames
//
// Sine implementation derived from slides here: http://15418.courses.cs.cmu.edu/spring2016/lecture/basicarch


// standard imports
#include <stdio.h>
#include <math.h>
#include <iomanip>
#include <iostream>
#include <string>
#include <sys/time.h>

// problem size (vector length) N
static const int N = 12345678;

// Number of terms to use when approximating sine
static const int TERMS = 6;

// kernel function (CPU - Do not modify)
void sine_serial(float *input, float *output)
{
  int i;

  for (i=0; i<N; i++) {
      float value = input[i]; 
      float numer = input[i] * input[i] * input[i]; 
      int denom = 6; // 3! 
      int sign = -1; 
      for (int j=1; j<=TERMS;j++) 
      { 
         value += sign * numer / denom; 
         numer *= input[i] * input[i]; 
         denom *= (2*j+2) * (2*j+3); 
         sign *= -1; 
      } 
      output[i] = value; 
    }
}


// kernel function (CUDA device)
// TODO: Implement your graphics kernel here. See assignment instructions for method information
const int block_size = 1024;
// typedef struct{
//  int N;
// }
__global__ void sine_parallel(float *input, float *output)
{
// if this does not work HOW we want there is an example of this here: http://15418.courses.cs.cmu.edu/spring2016/lecture/basicarch/slide_018  
//   pthread_t thread_id;
//   my_args args;
//   args.N = N/2;
  /*
  creates a thread_id for each thread based upon its position in every block.
  block_size defined at 1024.  
  this statement assigns every thread a value 0-(N-1) that will access its corresponding element of the array at the id.
  */
  int thread_id = blockIdx.x * block_size + threadIdx.x;
	
	
      /*
      This if-else statement is meant to prevent accesses past the end of the array.
      */
     //WE MIGHT WANT TO CONSIDER USING A FORALL LOOP HERE FOR A DATA-PARALLEL EXAMPLE?
      //forall(thread_id from 0 to N-1)
      if(thread_id =< N)
      {
	  float value = input[thread_id]; 
          float numer = input[thread_id] * input[thread_id] * input[thread_id]; 
          int denom = 6; // 3! 
      	  int sign = -1; 
          for (int j=1; j<=TERMS;j++) 
          { 
               value += sign * numer / denom; 
               numer *= input[thread_id] * input[thread_id]; 
               denom *= (2*j+2) * (2*j+3); 
               sign *= -1; 
          } 
          output[thread_id] = value; 
      }
      else
      {
	  //die ("Thread_id greater than N value. No calculation to run.");
	  //Could probably leave this Else() empty so the thread just does nothing.
      }    
}
// BEGIN: timing and error checking routines (do not modify)

// Returns the current time in microseconds
long long start_timer() {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec * 1000000 + tv.tv_usec;
}


// Prints the time elapsed since the specified time
long long stop_timer(long long start_time, std::string name) {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	long long end_time = tv.tv_sec * 1000000 + tv.tv_usec;
        std::cout << std::setprecision(5);	
	std::cout << name << ": " << ((float) (end_time - start_time)) / (1000 * 1000) << " sec\n";
	return end_time - start_time;
}

void checkErrors(const char label[])
{
  // we need to synchronise first to catch errors due to
  // asynchroneous operations that would otherwise
  // potentially go unnoticed

  cudaError_t err;

  err = cudaThreadSynchronize();
  if (err != cudaSuccess)
  {
    char *e = (char*) cudaGetErrorString(err);
    fprintf(stderr, "CUDA Error: %s (at %s)", e, label);
  }

  err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    char *e = (char*) cudaGetErrorString(err);
    fprintf(stderr, "CUDA Error: %s (at %s)", e, label);
  }
}

// END: timing and error checking routines (do not modify)



int main (int argc, char **argv)
{
  //BEGIN: CPU implementation (do not modify)
  float *h_cpu_result = (float*)malloc(N*sizeof(float));
  float *h_input = (float*)malloc(N*sizeof(float));
  //Initialize data on CPU
  int i;
  for (i=0; i<N; i++)
  {
    h_input[i] = 0.1f * i;
  }

  //Execute and time the CPU version
  long long CPU_start_time = start_timer();
  sine_serial(h_input, h_cpu_result);
  long long CPU_time = stop_timer(CPU_start_time, "\nCPU Run Time");
  //END: CPU implementation (do not modify)


  //TODO: Prepare and run your kernel, make sure to copy your results back into h_gpu_result and display your timing results
   
  //timer to calculate total GPU run time	
  long long Total_GPU_run_Time= start_timer();
	
   //timer for memory allocation	
  long long GPU_mem_allocation = start_timer();
  float *h_gpu_result = (float*)malloc(N*sizeof(float));
  if(cudaMalloc( (void **) &d_input, sizeof(&h_input))   != cudeSuccess ) die("Error allocating GPU memory");
  if(cudaMalloc( (void **) &d_output, sizeof(&h_input))   != cudeSuccess ) die("Error allocating GPU memory");
  long long GPU_mem_allocatoin_result = stop_timer(GPU_mem_allocation, "\nGPU Memory Allocation");	
  
  //timer for memory copy to device
  long long GPU_copy_to_device = start_timer();
  cudaMemcpy(d_input, h_input, sizeof(&h_input), cudaMemcpyHostToDevice);	
  long long GPU_mem_allocatoin_result = stop_timer(GPU_copy_to_device, "\nGPU Memory Copy to Device");
	
  //timer and execution of our GPU Kernel	
  long long GPU_start_time = start_timer();
  sine_parallel <<<(N/block_size + 1),block_size>>>(h_input, h_gpu_result);
  cudaThreadSynchronize();
  long long GPU_time = stop_timer(GPU_start_time, "\nGPU Kernel Run Time");
  
  //timer for memory copy back to host
  long long GPU_copy_to_host = start_timer();
  cudaMemcpy(h_gpu_result, d_output, sizeof(&h_input), cudaMemcpyDeviceToHost);
  long long GPU_tohost_copy_result = stop_timer(GPU_copy_to_host, "\nGPU Memory Copy to Host");
	
  //printing total GPU run time
  long long Result_of_GPU_run_time = stop_timer(Total_GPU_run_Time, "\nTotal GPU Run Time");
  
  
  // Checking to make sure the CPU and GPU results match - Do not modify
  int errorCount = 0;
  for (i=0; i<N; i++)
  {
    if (abs(h_cpu_result[i]-h_gpu_result[i]) > 1e-6)
      errorCount = errorCount + 1;
  }
  if (errorCount > 0)
    printf("Result comparison failed.\n");
  else
    printf("Result comparison passed.\n");

  // Cleaning up memory
  free(h_input);
  free(h_cpu_result);
  free(h_gpu_result);
  return 0;
}






