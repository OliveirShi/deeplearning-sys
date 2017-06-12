#include "./c_runtime_api.h"
#include <cassert>
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>


 /* return uprounding of division */
int ceil(int a, int b){
  if (a % b == 0) return a / b;
  return a / b + 1;
}

/* given an array, return its length */
int64_t totallength(DLArrayHandle array){
  int64_t length = 1;
  for (int i = 0; i < array->ndim; i++){
    length *= array->shape[i];
  }
  return length;
}

/* TODO: Your code here */
/* all your GPU kernel code*/

/* broadcast_kernel */
__global__ void broadcast_kernel(int64_t in_length,
                                 int64_t bc_length,
                                 const float *input_data,
                                 float *output_data){
  int y = blockIdx.x * blockDim.x + threadIdx.x;
  if (y >= bc_length) return;
  output_data += y * in_length;
  for (int k = 0; k < in_length; ++k){
    output_data[k] = input_data[k];
  }
}

int DLGpuBroadcastTo(const DLArrayHandle input, DLArrayHandle output){
  int64_t in_length = totallength(input);
  int64_t out_length = totallength(output);
  int64_t bc_length = output->shape[0];
  assert(in_length == out_length / bc_length);
  float *input_data = (float *)input->data;
  float *output_data = (float *)output->data;
  dim3 blocks, threads;
  if (bc_length < 1024){
    blocks.x = 1;
    threads.x = bc_length;
  }
  else{
    blocks.x = ceil(bc_length, 1024);
    threads.x = 1024;
  }
  broadcast_kernel<<<blocks, threads>>>(in_length, bc_length, input_data, output_data);
  return 0;
}

/* array_set_kernel */
__global__ void array_set_kernel(int64_t length,
                                 float *array_data,
                                 float value){
  int y = blockIdx.x * blockDim.x + threadIdx.x;
  if (y >= length) return;
  array_data[y] = value;
}

int DLGpuArraySet(DLArrayHandle arr, float value) { 
  /* TODO: Your code here */
  int64_t length = totallength(arr);
  float *array_data = (float *)arr->data;
  dim3 blocks, threads;
  if (length < 1024){
    blocks.x = 1;
    threads.x = length;
  }
  else{
    blocks.x = ceil(length, 1024);
    threads.x = 1024;
  }
  array_set_kernel<<<blocks, threads>>>(length, array_data, value);
  return 0;
}

/* reduce_sum_axis_zero_kernel*/
// output[c] = sum(input[:, c])
__global__ void reduce_sum_axis_zero_kernel(int64_t output_len, int64_t input_len,
                                            const float *input_data,
                                            float *output_data){
  int y = blockIdx.x * blockDim.x + threadIdx.x;
  if (y >= output_len) return;
  output_data[y] = 0;
  for (int x = y; x < input_len; x+=output_len){
    output_data[y] += input_data[x];
  }
}

int DLGpuReduceSumAxisZero(const DLArrayHandle input, DLArrayHandle output) {
  /* TODO: Your code here */
  assert (input->ndim  - output->ndim == 1);
  float *input_data = (float *)input->data;
  float *output_data = (float *)output->data;
  int64_t input_len = totallength(input);
  int64_t output_len = totallength(output);
  dim3 blocks, threads;
  if (output_len < 1024){
    blocks.x = 1;
    threads.x = output_len;
  }
  else{
    blocks.x = ceil(output_len, 1024);
    threads.x = 1024;
  }
  reduce_sum_axis_zero_kernel<<<blocks, threads>>>(output_len, input_len, input_data, output_data);
  return 0;
}

/* matrix_elementwise_add*/
// output[r, c] = matA[r, c] + mat_B[r, c]
__global__ void matrix_elementwise_add_kernel(int64_t length,
                                              const float *matA_data,
                                              const float *matB_data,
                                              float *output_data){
  int y = blockIdx.x * blockDim.x + threadIdx.x;
  if (y >= length) return;
  output_data[y] = matA_data[y] + matB_data[y];
}

int DLGpuMatrixElementwiseAdd(const DLArrayHandle matA,
                              const DLArrayHandle matB, DLArrayHandle output) {
  /* TODO: Your code here */
  assert (matA->ndim == matB->ndim);
  for (int i = 0; i < matA->ndim; i++){
    assert (matA->shape[i] == matB->shape[i]);
  }
  int64_t length = totallength(matA);
  float *matA_data = (float *)matA->data;
  float *matB_data = (float *)matB->data;
  float *output_data = (float *)output->data;
  dim3 blocks, threads;
  if (length < 1024){
    blocks.x = 1;
    threads.x = length;
  }
  else{
    blocks.x = ceil(length, 1024);
    threads.x = 1024;
  }
  matrix_elementwise_add_kernel<<<blocks, threads>>>(length, matA_data, matB_data, output_data);
  return 0;
}

/* matrix_elementwise_add_by_const */
// output[r, c] = input[r, c] + val
__global__ void matrix_elementwise_add_by_const_kernel(int64_t length,
                                                       const float *input_data,
                                                       const float val,
                                                       float *output_data){
  int y = blockIdx.x * blockDim.x + threadIdx.x;
  if (y >= length) return;
  output_data[y] = input_data[y] + val;
}

int DLGpuMatrixElementwiseAddByConst(const DLArrayHandle input, float val,
                                     DLArrayHandle output) {
  /* TODO: Your code here */
  float *input_data = (float *)input->data;
  float *output_data = (float *)output->data;
  int64_t length = totallength(input);
  dim3 blocks, threads;
  if (length < 1024){
    blocks.x = 1;
    threads.x = length;
  }
  else{
    blocks.x = ceil(length, 1024);
    threads.x = 1024;
  }
  matrix_elementwise_add_by_const_kernel<<<blocks, threads>>>(length, input_data, val, output_data);
  return 0;
}

/* matrix_elementwise_multiply */
// output[r, c] = matA[r, c] * matB[r, c]
__global__ void matrix_elementwise_multiply_kernel(int64_t length,
                                                   const float *matA_data,
                                                   const float *matB_data,
                                                  float *output_data){
  int y = blockIdx.x * blockDim.x + threadIdx.x;
  if (y >= length) return;
  output_data[y] = matA_data[y] * matB_data[y];
}

int DLGpuMatrixElementwiseMultiply(const DLArrayHandle matA,
                                   const DLArrayHandle matB,
                                   DLArrayHandle output) {
  /* TODO: Your code here */
  assert (matA->ndim == matB->ndim);
  for (int i = 0; i < matA->ndim; i++){
    assert (matA->shape[i] == matB->shape[i]);
  }
  int64_t length = totallength(matA);
  float *matA_data = (float *)matA->data;
  float *matB_data = (float *)matB->data;
  float *output_data = (float *)output->data;
  dim3 blocks, threads;
  if (length < 1024){
    blocks.x = 1;
    threads.x = length;
  }
  else{
    blocks.x = ceil(length, 1024);
    threads.x = 1024;
  }
  matrix_elementwise_multiply_kernel<<<blocks, threads>>>(length, matA_data, matB_data, output_data);
  return 0;
}

/* matrix_multiply_by_const */
// output[r, c] = input[r, c] * val
__global__ void matrix_multiply_by_const_kernel(int64_t length,
                                                const float *input_data,
                                                const float val,
                                                float *output_data){
  int y = blockIdx.x * blockDim.x + threadIdx.x;
  if (y >= length) return;
  output_data[y] = input_data[y] * val;
}

int DLGpuMatrixMultiplyByConst(const DLArrayHandle input, float val,
                               DLArrayHandle output) {
  /* TODO: Your code here */
  float *input_data = (float *)input->data;
  float *output_data = (float *)output->data;
  int64_t length = totallength(input);
  dim3 blocks, threads;
  if (length < 1024){
    blocks.x = 1;
    threads.x = length;
  }
  else{
    blocks.x = ceil(length, 1024);
    threads.x = 1024;
  }
  matrix_multiply_by_const_kernel<<<blocks, threads>>>(length, input_data, val, output_data);
  return 0;
}

/* matrix_multiply*/
// output[r, c] = sum(matA[r, :] * matB[:, c])
cublasHandle_t cublas_handle = NULL;
int DLGpuMatrixMultiply(const DLArrayHandle matA, bool transposeA,
                        const DLArrayHandle matB, bool transposeB,
                        DLArrayHandle matC) {
  /* TODO: Your code here */
  // Hint: use cublas
  // cublas assume matrix is column major
  if(!cublas_handle) {
    cublasCreate(&cublas_handle);
  }

  float one = 1.0f;
  float zero = 0.0f;
  int m = matC->shape[1];
  int n = matC->shape[0];
  int k = transposeA ? matA->shape[0] : matA->shape[1];

  cublasSgemm(cublas_handle,
    transposeB ? CUBLAS_OP_T : CUBLAS_OP_N,
    transposeA ? CUBLAS_OP_T : CUBLAS_OP_N,
    m, n, k,
    &one,
    (const float*)matB->data, !transposeB ? m : k,
    (const float*)matA->data, !transposeA ? k : n,
    &zero,
    (float*)matC->data, m
  );
  return 0;
}

/* relu_kernel */
// output[r] = max(0., input[r])
__global__ void relu_kernel(int64_t length,
                            const float *input_data,
                            float *output_data){
  int y = blockIdx.x * blockDim.x + threadIdx.x;
  if (y >= length){
    return;
  }
  output_data[y] = max(0.0f, input_data[y]);
}

int DLGpuRelu(const DLArrayHandle input, DLArrayHandle output) {
  /* TODO: Your code here */
  int64_t length = totallength(input);
  float *input_data = (float *)input->data;
  float *output_data = (float *)output->data;
  dim3 blocks, threads;
  if (length < 1024){
    blocks.x = 1;
    threads.x = length;
  }
  else{
    blocks.x = ceil(length, 1024);
    threads.x = 1024;
  }
  relu_kernel<<<blocks, threads>>>(length, input_data, output_data);
  return 0;
}

/* relu_gradien_kernel */
__global__ void relu_gradien_kernel(int64_t length,
                                    const float *input_data,
                                    const float *in_grad_data,
                                    float *output_data){
  // One dimensional thread bolcks
  int y = blockIdx.x * blockDim.x + threadIdx.x;
  if (y >= length){
    return;
  }
  output_data[y] = input_data[y] > 0.0f ? in_grad_data[y] : 0.0f;
}

int DLGpuReluGradient(const DLArrayHandle input, const DLArrayHandle in_grad,
                      DLArrayHandle output) {
  /* TODO: Your code here */
  if (input->ndim == 1){
    assert(in_grad->ndim == 1 && input->shape[0] == in_grad->shape[0]);
  }
  else{
    assert(in_grad->ndim == 2 && input->shape[0] == in_grad->shape[0]
                              && input->shape[1] == in_grad->shape[1]);
  }
  int64_t length = totallength(input);
  float *input_data = (float *)input->data;
  float *in_grad_data = (float *)in_grad->data;
  float *output_data = (float *)output->data;
  dim3 blocks, threads;
  if (length < 1024){
    blocks.x = 1;
    threads.x = length;
  }
  else{
    blocks.x = ceil(length, 1024);
    threads.x = 1024;
  }
  relu_gradien_kernel<<<blocks, threads>>>(length, input_data, in_grad_data, output_data);
  return 0;
}

/* matrix_softmax_kernel */
// output[r, c] = exp(input[r,c]-max(input[r,:])) / sum(exp(input[r,:]-max(input[r,:])))
__global__ void matrix_softmax_kernel(int64_t nrow, int64_t ncol,
                                      const float *input_data,
                                      float *output_data){
  // two dimensional thread blocks.
  int y = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
  if (y >= nrow){
    return;
  }
  // y_th row of input data
  input_data += y * ncol;
  output_data += y * ncol;
  // find max for a row.
  float maxval = *input_data;
  for (int x = 1; x < ncol; ++x){
    maxval = max(maxval, input_data[x]);
  }
  // Deduct by max for a row, and raise to exp.
  // in case of too large of exp, and the result will not be affected
  float sum = 0;
  for (int x = 0; x < ncol; ++x) {
    sum += exp(input_data[x] - maxval);
  }
  // Compute per-row softmax.
  for (int x = 0; x < ncol; ++x) {
    output_data[x] = exp(input_data[x] - maxval) / sum;
  }
}

int DLGpuSoftmax(const DLArrayHandle input, DLArrayHandle output) {
  /* TODO: Your code here */
  assert(input->ndim == 2);
  assert(output->ndim == 2);
  int64_t nrow = input->shape[0];
  int64_t ncol = input->shape[1];
  float *input_data = (float *)input->data;
  float *output_data = (float *)output->data;
  dim3 threads;
  if (nrow < 1024){
    threads.x = nrow;
  }
  else{
    threads.x = 1024;
    threads.y = ceil(nrow, 1024);
  }
  matrix_softmax_kernel<<<1, threads>>>(nrow, ncol, input_data, output_data);
  return 0;
}

// matrix_softmax_cross_entropy_kernel
// y = inputs[0], y_ = inputs[1]
// np.mean(-np.sum(y_ * np.log(softmax(y)), axis=1), keepdims=True)
__global__ void matrix_softmax_cross_entropy_kernel(int nrow, int ncol,
                                                    const float *input_a,
                                                    const float *input_b,
                                                    float *output) {
  // Dynamic shared memory, size provided at kernel launch.
  extern __shared__ float loss_per_row[];
  // Two dimensional thread blocks.
  //for each thread indexed as y.
  int y = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x +
          threadIdx.x;
  if (y >= nrow) {
    return;
  }
  // y_th row of input data
  input_a += y * ncol;
  input_b += y * ncol;
  float maxval = *input_a;
  // Find max for a row.
  for (int x = 1; x < ncol; ++x) {
    maxval = max(maxval, input_a[x]);
  }
  // Deduct by max for a row, and raise to exp.
  // in case of too large of exp, and the result will not be affected
  float sum = 0;
  for (int x = 0; x < ncol; ++x) {
    sum += exp(input_a[x] - maxval);
  }
  // Compute per-row loss.
  float loss = 0;
  for (int x = 0; x < ncol; ++x) {
    loss -= input_b[x] * log(exp(input_a[x] - maxval) / sum);
  }
  loss_per_row[y] = loss;
  __syncthreads(); // synchronize all the threads in this block
  // Compute reduce_mean across rows.
  float mean_loss = 0;
  // Use a single thread to reduce mean across rows.
  if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
    for (int i = 0; i < nrow; ++i) {
      mean_loss += loss_per_row[i];
    }
    mean_loss /= nrow;
    output[0] = mean_loss;
  }
}

int DLGpuSoftmaxCrossEntropy(const DLArrayHandle input_a,
                             const DLArrayHandle input_b,
                             DLArrayHandle output) {
  assert(input_a->ndim == 2);
  assert(input_b->ndim == 2);
  assert(output->ndim == 1);
  assert(input_a->shape[0] == input_b->shape[0] &&
         input_a->shape[1] == input_b->shape[1]);
  int nrow = input_a->shape[0];
  // Maximum x- or y-dimension of a block = 1024
  // But we need 'nrow' shared memory, and max shared memory is 48KB.
  // Conservatively allow max 16KB shared memory.
  assert(nrow <= 1024 * 4);
  int ncol = input_a->shape[1];
  const float *input_data_a = (const float *)input_a->data;
  const float *input_data_b = (const float *)input_b->data;
  float *output_data = (float *)output->data;
  dim3 threads;
  if (nrow <= 1024) {
    threads.x = nrow;
  } else {
    threads.x = 1024;
    threads.y = (nrow + 1023) / 1024;
  }
  // 1 block, each block with 'threads' number of threads with 'nrow' shared
  // memory size
  matrix_softmax_cross_entropy_kernel<<<1, threads, nrow * sizeof(float)>>>(
      nrow, ncol, input_data_a, input_data_b, output_data);
  return 0;
}
