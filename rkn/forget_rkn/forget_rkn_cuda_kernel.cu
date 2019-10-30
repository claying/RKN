#include <ATen/ATen.h>
// #include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <stdio.h>
// template<typename T>
// using tpair_of = std::tuple<T, T>;

// template<size_t index>
// std::vector<Tensor> project(at::ArrayRef<tpair_of<Tensor>> tuples) {
//   std::vector<Tensor> result;
//   result.reserve(tuples.size());
//   for (auto & t : tuples) {
//     result.push_back(std::get<index>(t));
//   }
//   return result;
// }

// Tensor hidden_concat(at::ArrayRef<Tensor> hiddens) { return at::cat(hiddens, 0); }
// tpair_of<Tensor> hidden_concat(at::ArrayRef<tpair_of<Tensor>> hiddens) {
//   return std::make_tuple(hidden_concat(project<0>(hiddens)), hidden_concat(project<1>(hiddens)));

// Tensor hidden_slice(const Tensor& t, int64_t64_t start, int64_t64_t end) {
//   return t.narrow(0, start, end - start);
// }
// tpair_of<Tensor> hidden_slice(const tpair_of<Tensor>& t, int64_t64_t start, int64_t64_t end) {
//   return std::make_tuple(hidden_slice(std::get<0>(t), start, end),
//                          hidden_slice(std::get<1>(t), start, end));
// }
namespace {
template <typename scalar_t>
__global__ void forget_rkn_cuda_forward_kernel(
    // scalar_t* __restrict__ outputs,
    // scalar_t* __restrict__ hiddens,
    scalar_t* __restrict__ new_output,
    scalar_t* __restrict__ new_hidden,
    scalar_t* __restrict__ step_output,
    const scalar_t* __restrict__ step_input,
    const scalar_t* __restrict__ hidden,
    const scalar_t* __restrict__ forget,
    size_t batch_size,
    size_t state_size,
    size_t kmer_size,
    int64_t time_step,
    bool compute_la,
    bool additive) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int block_size = state_size * kmer_size;
    // const int64_t row = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch = blockIdx.y * blockDim.y + threadIdx.y;
    // if (time_step >= 3)
    //     printf("step %ld: %d\n", time_step, column);
        // printf("%d\n", column);
    const int64_t index = batch * block_size + column;
    // const int64_t low_index = index - 1;
    const auto hidden_low = (index % kmer_size == 0) ? 1.0 : hidden[index - 1];
    // const int64_t global_index = input_offset * block_size + index;

    if (column < block_size && batch < batch_size) {
        new_hidden[index] = forget[index] * hidden[index];
        if (additive)
            new_hidden[index] += (1 - forget[index]) * (step_input[index] + hidden_low);
        else
            new_hidden[index] += (1 - forget[index]) * step_input[index] * hidden_low;
        // hiddens[global_index] = new_hidden[index];
        if (compute_la) {
            if (additive)
                step_output[index] = time_step * step_output[index] + hidden_low + step_input[index];
            else
                step_output[index] = time_step * step_output[index] + hidden_low * step_input[index];
            step_output[index] /= time_step + 1.;
            // step_output[index] += hidden_low * inputs[global_index];
            // outputs[global_index] = step_output[index];
        }
        else {
            step_output[index] = new_hidden[index];
            // outputs[global_index] = step_output[index];
        }
        new_output[index] = step_output[index];
        // printf("step %ld; index %ld; hidden %f\n", time_step, index, hidden_low);
    }
}

template <typename scalar_t>
__global__ void forget_rkn_cuda_backward_kernel(
    // scalar_t* __restrict__ d_inputs,
    scalar_t* __restrict__ d_step_input,
    scalar_t* __restrict__ d_forget,
    scalar_t* __restrict__ d_hidden,
    scalar_t* __restrict__ new_d_hidden,
    // scalar_t* __restrict__ d_hiddens,
    scalar_t* __restrict__ d_output,
    const scalar_t* __restrict__ d_step_output,
    // const scalar_t* __restrict__ outputs,
    // const scalar_t* __restrict__ inputs,
    // const scalar_t* __restrict__ hiddens,
    const scalar_t* __restrict__ step_input,
    // const scalar_t* __restrict__ hidden,
    const scalar_t* __restrict__ hidden_minus1,
    const scalar_t* __restrict__ forget,
    int64_t batch_size,
    size_t state_size,
    size_t kmer_size,
    int64_t time_step,
    bool compute_la,
    bool additive) {

    const int64_t column = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t block_size = state_size * kmer_size;
    const int64_t batch = blockIdx.y * blockDim.y + threadIdx.y;
    const int64_t index = batch * block_size + column;
    const int64_t low_index = index - 1;

    auto hidden_minus1_val = 0.0;
    auto hidden_minus1_low = (index % kmer_size == 0) ? 1.0 : 0.0;
    if (hidden_minus1 != NULL) {
        hidden_minus1_val = hidden_minus1[index];
        hidden_minus1_low = (index % kmer_size == 0) ? 1.0 : hidden_minus1[index - 1];
    }

    if (column < block_size && batch < batch_size) {
        //printf("start step %ld; batch %ld; column %ld; index %ld; d_input %f; new_d_hidden %f; d_hidden %f; low_index %ld; hidden_minus1_low %f\n", time_step, batch, column, index, d_step_input[index], new_d_hidden[index], d_hidden[index], low_index, hidden_minus1_low);
        d_output[index] += d_step_output[index];
        if (compute_la) {
            if (additive)
                d_step_input[index] = d_output[index] / (time_step + 1.);
            else
                d_step_input[index] = hidden_minus1_low * d_output[index] / (time_step + 1.);
        } else {
            d_hidden[index] += d_output[index];
        }

        if (additive) {
            d_step_input[index] += (1 - forget[index]) * d_hidden[index];
            d_forget[index] += (hidden_minus1_val - step_input[index] - hidden_minus1_low) * d_hidden[index];
        } else{
            d_step_input[index] += (1 - forget[index]) * hidden_minus1_low * d_hidden[index];
            d_forget[index] += (hidden_minus1_val - step_input[index] * hidden_minus1_low) * d_hidden[index];
        }

        if (index % kmer_size != 0)
            if (additive)
                new_d_hidden[low_index] = (1 - forget[index]) * d_hidden[index];
            else
                new_d_hidden[low_index] = (1 - forget[index]) * step_input[index] * d_hidden[index];
        if (compute_la) {
            if (index % kmer_size != 0)
                if (additive)
                    new_d_hidden[low_index] += d_output[index] / (time_step + 1.);
                else
                    new_d_hidden[low_index] += step_input[index] * d_output[index] / (time_step + 1.);
            d_output[index] *= time_step / (time_step + 1.);
        } else {
            d_output[index] = 0.;
        }
        d_hidden[index] = forget[index] * d_hidden[index];
        //printf("step %ld; batch %ld; column %ld; index %ld; d_input %f; new_d_hidden %f; d_hidden %f; low_index %ld; hidden_minus1_low %f\n", time_step, batch, column, index, d_step_input[index], new_d_hidden[index], d_hidden[index], low_index, hidden_minus1_low);

    }
}

template <typename scalar_t>
__global__ void forget_rkn_packed_cuda_forward_kernel(
    scalar_t* __restrict__ outputs,
    scalar_t* __restrict__ hiddens,
    scalar_t* __restrict__ step_output,
    scalar_t* __restrict__ new_hidden,
    const scalar_t* __restrict__ inputs,
    const scalar_t* __restrict__ hidden,
    const scalar_t* __restrict__ forget,
    size_t batch_size,
    size_t state_size,
    size_t kmer_size,
    int64_t input_offset,
    int64_t time_step,
    bool compute_la,
    bool additive) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int block_size = state_size * kmer_size;
    // const int64_t row = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch = blockIdx.y * blockDim.y + threadIdx.y;
    // if (time_step >= 3)
    //     printf("step %ld: %d\n", time_step, column);
        // printf("%d\n", column);
    const int64_t index = batch * block_size + column;
    // const int64_t low_index = index - 1;
    const auto hidden_low = (index % kmer_size == 0) ? 1.0 : hidden[index - 1];
    const int64_t global_index = input_offset * block_size + index;

    if (column < block_size && batch < batch_size) {
        new_hidden[index] = forget[index] * hidden[index];
        if (additive)
            new_hidden[index] += (1 - forget[index]) * (inputs[global_index] + hidden_low);
        else
            new_hidden[index] += (1 - forget[index]) * inputs[global_index] * hidden_low;
        hiddens[global_index] = new_hidden[index];
        if (compute_la) {
            if (additive)
                step_output[index] = time_step * step_output[index] + hidden_low + inputs[global_index];
            else
                step_output[index] = time_step * step_output[index] + hidden_low * inputs[global_index];
            step_output[index] /= time_step + 1.;
            // step_output[index] += hidden_low * inputs[global_index];
            outputs[global_index] = step_output[index];
        }
        else {
            step_output[index] = new_hidden[index];
            outputs[global_index] = step_output[index];
        }
        // printf("step %ld; index %ld; hidden %f\n", time_step, index, hidden_low);
    }
}

template <typename scalar_t>
__global__ void forget_rkn_packed_cuda_backward_kernel(
    scalar_t* __restrict__ d_inputs,
    scalar_t* __restrict__ d_forget,
    scalar_t* __restrict__ d_hidden,
    scalar_t* __restrict__ new_d_hidden,
    // scalar_t* __restrict__ d_hiddens,
    scalar_t* __restrict__ d_output,
    const scalar_t* __restrict__ d_outputs,
    // const scalar_t* __restrict__ outputs,
    const scalar_t* __restrict__ inputs,
    const scalar_t* __restrict__ hiddens,
    const scalar_t* __restrict__ forget,
    int64_t batch_size,
    int64_t next_batch_size,
    size_t state_size,
    size_t kmer_size,
    int64_t input_offset,
    int64_t time_step,
    bool compute_la,
    bool additive) {
    const int64_t column = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t block_size = state_size * kmer_size;
    const int64_t batch = blockIdx.y * blockDim.y + threadIdx.y;
    const int64_t index = batch * block_size + column;
    const int64_t low_index = index - 1;
    // const auto hidden_low = (low_index < 0) ? 1.0 : hidden[low_index];
    const int64_t global_index = input_offset * block_size + index;
    // const int64_t global_index_minus1 = ()
    const int64_t global_index_minus1 = global_index - next_batch_size * block_size;
    // const int64_t low_global_index_minus1 = global_index_minus1 - 1;
    const int64_t threshold = (input_offset - next_batch_size) * block_size;
    // auto hidden_minus1 = 0.;
    // auto hidden_minus1_low = 0.;
    // if (next_batch_size > 0) {
    //     hidden_minus1 = hiddens[global_index_minus1];
    //     hidden_minus1_low = ((global_index_minus1 - threshold) % kmer_size == 0) ? 1.0 : hiddens[global_index_minus1 - 1];
    // } else {
    //     hidden_minus1_low = ((global_index_minus1 - threshold) % kmer_size == 0) ? 1.0 : 0.0;
    // }
    const auto hidden_minus1 = (next_batch_size > 0) ? hiddens[global_index_minus1] : 0.0;
    auto hidden_minus1_low = 1.0;
    if ((global_index_minus1 - threshold) % kmer_size != 0)
        hidden_minus1_low = (next_batch_size > 0) ? hiddens[global_index_minus1 - 1] : 0.0;
    // const auto hidden_minus1_low = ((global_index_minus1 - threshold) % kmer_size == 0) ? 1.0 : hiddens[global_index_minus1 - 1];

    if (column < block_size && batch < batch_size) {
        // printf("start step %ld; batch %ld; column %ld; index %ld; global_index %ld; d_inputs %f; new_d_hidden %f; d_hidden %f; low_index %ld\n", time_step, batch, column, index, global_index, d_inputs[global_index], new_d_hidden[index], d_hidden[index], low_index);
        // printf("step %ld; batch %ld; column %ld; index %ld; global_index %ld;\n", time_step, batch, column, index, global_index);
        d_output[index] += d_outputs[global_index];
        if (compute_la) {
            if (additive)
                d_inputs[global_index] = d_output[index] / (time_step + 1.);
            else
                d_inputs[global_index] = hidden_minus1_low * d_output[index] / (time_step + 1.);
            // d_hiddens[global_index_low] = inputs[global_index] * d_output[index] / (time_step + 1.);
        } else {
            d_hidden[index] += d_output[index];
        }
        if (additive) {
            d_inputs[global_index] += (1 - forget[index]) * d_hidden[index];
            d_forget[index] += (hidden_minus1 - inputs[global_index] - hidden_minus1_low) * d_hidden[index];
        } else{
            d_inputs[global_index] += (1 - forget[index]) * hidden_minus1_low * d_hidden[index];
            d_forget[index] += (hidden_minus1 - inputs[global_index] * hidden_minus1_low) * d_hidden[index];
        }

        // new_d_hidden[index] += forget[index] * d_hidden[index];
        // atomicAdd(&new_d_hidden[index], forget[index] * d_hidden[index]);

        if (index % kmer_size != 0)
            if (additive)
                new_d_hidden[low_index] = (1 - forget[index]) * d_hidden[index];
            else
                new_d_hidden[low_index] = (1 - forget[index]) * inputs[global_index] * d_hidden[index];
            // atomicAdd(&new_d_hidden[low_index], (1 - forget[index]) * inputs[global_index] * d_hidden[index]);
        if (compute_la) {
            if (index % kmer_size != 0)
                if (additive)
                    new_d_hidden[low_index] += d_output[index] / (time_step + 1.);
                else
                    new_d_hidden[low_index] += inputs[global_index] * d_output[index] / (time_step + 1.);
            d_output[index] *= time_step / (time_step + 1.);
        } else {
            d_output[index] = 0.;
        }
        d_hidden[index] = forget[index] * d_hidden[index];
        // printf("step %ld; batch %ld; column %ld; index %ld; global_index %ld; d_inputs %f; new_d_hidden: %f; d_hidden: %f; low_index %ld\n", time_step, batch, column, index, global_index, d_inputs[global_index], new_d_hidden[index], d_hidden[index], low_index);
        // printf("step %ld; index %ld; global_index %ld; d_hidden %f; d_output %f; d_inputs %f; new_d_hidden %f\n", time_step, index, global_index, d_hidden[index], d_output[index], d_inputs[global_index], new_d_hidden[index]);
    }
}

template <typename scalar_t>
__global__ void forget_rkn_max_cuda_forward_kernel(
    // scalar_t* __restrict__ outputs,
    // scalar_t* __restrict__ hiddens,
    scalar_t* __restrict__ new_output,
    scalar_t* __restrict__ new_hidden,
    scalar_t* __restrict__ step_output,
    const scalar_t* __restrict__ step_input,
    const scalar_t* __restrict__ hidden,
    const scalar_t* __restrict__ forget,
    uint8_t* __restrict__ mask_hidden,
    uint8_t* __restrict__ mask_output,
    size_t batch_size,
    size_t state_size,
    size_t kmer_size,
    int64_t time_step,
    bool compute_la,
    bool additive) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int block_size = state_size * kmer_size;
    // const int64_t row = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch = blockIdx.y * blockDim.y + threadIdx.y;
    // if (time_step >= 3)
    //     printf("step %ld: %d\n", time_step, column);
        // printf("%d\n", column);
    const int64_t index = batch * block_size + column;
    // const int64_t low_index = index - 1;
    const auto hidden_low = (index % kmer_size == 0) ? 1.0 : hidden[index - 1];
    // const int64_t global_index = input_offset * block_size + index;
    auto aux = step_input[index];

    if (column < block_size && batch < batch_size) {
        new_hidden[index] = forget[index] * hidden[index];
        if (additive)
            aux = step_input[index] + hidden_low;
        else
            aux = step_input[index] * hidden_low;
        mask_hidden[index] = new_hidden[index] > aux;
        new_hidden[index] = fmax(new_hidden[index], aux);

        if (compute_la) {
            if (additive)
                aux = hidden_low + step_input[index];
            else
                aux = hidden_low * step_input[index];
            mask_output[index] = step_output[index] > aux;
            step_output[index] = fmax(step_output[index], aux);
        }
        else {
            step_output[index] = new_hidden[index];
            // outputs[global_index] = step_output[index];
        }
        new_output[index] = step_output[index];
        // printf("step %ld; index %ld; hidden %f\n", time_step, index, hidden_low);
    }
}

template <typename scalar_t>
__global__ void forget_rkn_max_cuda_backward_kernel(
    // scalar_t* __restrict__ d_inputs,
    scalar_t* __restrict__ d_step_input,
    scalar_t* __restrict__ d_forget,
    scalar_t* __restrict__ d_hidden,
    scalar_t* __restrict__ new_d_hidden,
    // scalar_t* __restrict__ d_hiddens,
    scalar_t* __restrict__ d_output,
    const scalar_t* __restrict__ d_step_output,
    // const scalar_t* __restrict__ outputs,
    // const scalar_t* __restrict__ inputs,
    // const scalar_t* __restrict__ hiddens,
    const scalar_t* __restrict__ step_input,
    // const scalar_t* __restrict__ hidden,
    const scalar_t* __restrict__ hidden_minus1,
    const scalar_t* __restrict__ forget,
    const uint8_t* __restrict__ mask_hidden,
    const uint8_t* __restrict__ mask_output,
    int64_t batch_size,
    size_t state_size,
    size_t kmer_size,
    int64_t time_step,
    bool compute_la,
    bool additive) {

    const int64_t column = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t block_size = state_size * kmer_size;
    const int64_t batch = blockIdx.y * blockDim.y + threadIdx.y;
    const int64_t index = batch * block_size + column;
    const int64_t low_index = index - 1;

    auto hidden_minus1_val = 0.0;
    auto hidden_minus1_low = (index % kmer_size == 0) ? 1.0 : 0.0;
    if (hidden_minus1 != NULL) {
        hidden_minus1_val = hidden_minus1[index];
        hidden_minus1_low = (index % kmer_size == 0) ? 1.0 : hidden_minus1[index - 1];
    }

    if (column < block_size && batch < batch_size) {
        //printf("start step %ld; batch %ld; column %ld; index %ld; d_input %f; new_d_hidden %f; d_hidden %f; low_index %ld; hidden_minus1_low %f\n", time_step, batch, column, index, d_step_input[index], new_d_hidden[index], d_hidden[index], low_index, hidden_minus1_low);
        d_output[index] += d_step_output[index];
        if (compute_la) {
            if (additive)
                d_step_input[index] = mask_output[index] ? 0.0 : d_output[index] ;
            else
                d_step_input[index] = mask_output[index] ? 0.0 : hidden_minus1_low * d_output[index];
        } else {
            d_hidden[index] += d_output[index];
        }

        if (additive) {
            d_step_input[index] += mask_hidden[index] ? 0.0 : d_hidden[index];
            // d_forget[index] += (mask_hidden[index] ? hidden_minus1_val : (- step_input[index] - hidden_minus1_low)) * d_hidden[index];
        } else{
            d_step_input[index] += mask_hidden[index] ? 0.0 : hidden_minus1_low * d_hidden[index];
            // d_forget[index] += mask_hidden[index] ? hidden_minus1_val * d_hidden[index] : 0.0;
        }
        d_forget[index] += mask_hidden[index] ? hidden_minus1_val * d_hidden[index] : 0.0;

        if (index % kmer_size != 0)
            if (additive)
                new_d_hidden[low_index] = mask_hidden[index] ? 0.0 : d_hidden[index];
            else
                new_d_hidden[low_index] = mask_hidden[index] ? 0.0 : step_input[index] * d_hidden[index];
        if (compute_la) {
            if (index % kmer_size != 0)
                if (additive)
                    new_d_hidden[low_index] += mask_output[index] ? 0.0 : d_output[index];
                else
                    new_d_hidden[low_index] += mask_output[index] ? 0.0 : step_input[index] * d_output[index];
            d_output[index] *= mask_output[index] ? 1.0 : 0.0;
        } else {
            d_output[index] = 0.;
        }
        d_hidden[index] = mask_hidden[index] ? forget[index] * d_hidden[index] : 0.0;
        //printf("step %ld; batch %ld; column %ld; index %ld; d_input %f; new_d_hidden %f; d_hidden %f; low_index %ld; hidden_minus1_low %f\n", time_step, batch, column, index, d_step_input[index], new_d_hidden[index], d_hidden[index], low_index, hidden_minus1_low);

    }
}

template <typename scalar_t>
__global__ void forget_rkn_packed_max_cuda_forward_kernel(
    scalar_t* __restrict__ outputs,
    scalar_t* __restrict__ hiddens,
    scalar_t* __restrict__ step_output,
    scalar_t* __restrict__ new_hidden,
    const scalar_t* __restrict__ inputs,
    const scalar_t* __restrict__ hidden,
    const scalar_t* __restrict__ forget,
    uint8_t* __restrict__ mask_outputs,
    uint8_t* __restrict__ mask_hiddens,
    size_t batch_size,
    size_t state_size,
    size_t kmer_size,
    int64_t input_offset,
    int64_t time_step,
    bool compute_la,
    bool additive) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int block_size = state_size * kmer_size;
    const int batch = blockIdx.y * blockDim.y + threadIdx.y;

    const int64_t index = batch * block_size + column;
    const auto hidden_low = (index % kmer_size == 0) ? 1.0 : hidden[index - 1];
    const int64_t global_index = input_offset * block_size + index;
    auto aux = hidden_low;

    if (column < block_size && batch < batch_size) {
        new_hidden[index] = forget[index] * hidden[index];
        if (additive)
            aux = inputs[global_index] + hidden_low;
        else
            aux = inputs[global_index] * hidden_low;
        mask_hiddens[global_index] = new_hidden[index] > aux;
        new_hidden[index] = fmax(new_hidden[index], aux);
        hiddens[global_index] = new_hidden[index];
        if (compute_la) {
            if (additive)
                aux = hidden_low + inputs[global_index];
            else
                aux = hidden_low * inputs[global_index];
            mask_outputs[global_index] = step_output[index] > aux;
            step_output[index] = fmax(step_output[index], aux);
            outputs[global_index] = step_output[index];
        }
        else {
            step_output[index] = new_hidden[index];
            outputs[global_index] = step_output[index];
        }
    }
}

template <typename scalar_t>
__global__ void forget_rkn_packed_max_cuda_backward_kernel(
    scalar_t* __restrict__ d_inputs,
    scalar_t* __restrict__ d_forget,
    scalar_t* __restrict__ d_hidden,
    scalar_t* __restrict__ new_d_hidden,
    scalar_t* __restrict__ d_output,
    const scalar_t* __restrict__ d_outputs,
    const scalar_t* __restrict__ inputs,
    const scalar_t* __restrict__ hiddens,
    const scalar_t* __restrict__ forget,
    const uint8_t* __restrict__ mask_outputs,
    const uint8_t* __restrict__ mask_hiddens,
    int64_t batch_size,
    int64_t next_batch_size,
    size_t state_size,
    size_t kmer_size,
    int64_t input_offset,
    int64_t time_step,
    bool compute_la,
    bool additive) {
    const int64_t column = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t block_size = state_size * kmer_size;
    const int64_t batch = blockIdx.y * blockDim.y + threadIdx.y;
    const int64_t index = batch * block_size + column;
    const int64_t low_index = index - 1;
    const int64_t global_index = input_offset * block_size + index;
    const int64_t global_index_minus1 = global_index - next_batch_size * block_size;
    const int64_t threshold = (input_offset - next_batch_size) * block_size;
    const auto hidden_minus1 = (next_batch_size > 0) ? hiddens[global_index_minus1] : 0.0;
    auto hidden_minus1_low = 1.0;
    if ((global_index_minus1 - threshold) % kmer_size != 0)
        hidden_minus1_low = (next_batch_size > 0) ? hiddens[global_index_minus1 - 1] : 0.0;

    if (column < block_size && batch < batch_size) {
        d_output[index] += d_outputs[global_index];
        if (compute_la) {
            if (additive)
                d_inputs[global_index] = mask_outputs[global_index] ? 0.0 : d_output[index];
            else
                d_inputs[global_index] = mask_outputs[global_index] ? 0.0 : hidden_minus1_low * d_output[index];
        } else {
            d_hidden[index] += d_output[index];
        }
        if (additive) {
            d_inputs[global_index] += mask_hiddens[global_index] ? 0.0 : d_hidden[index];
            // d_forget[index] += (hidden_minus1 - inputs[global_index] - hidden_minus1_low) * d_hidden[index];
        } else{
            d_inputs[global_index] += mask_hiddens[global_index] ? 0.0 : hidden_minus1_low * d_hidden[index];
            // d_forget[index] += (hidden_minus1 - inputs[global_index] * hidden_minus1_low) * d_hidden[index];
        }
        d_forget[index] += mask_hiddens[global_index] ? hidden_minus1 * d_hidden[index] : 0.0; 

        if (index % kmer_size != 0)
            if (additive)
                new_d_hidden[low_index] = mask_hiddens[global_index] ? 0.0 : d_hidden[index];
            else
                new_d_hidden[low_index] = mask_hiddens[global_index] ? 0.0 : inputs[global_index] * d_hidden[index];
        if (compute_la) {
            if (index % kmer_size != 0)
                if (additive)
                    new_d_hidden[low_index] += mask_outputs[global_index] ? 0.0 : d_output[index];
                else
                    new_d_hidden[low_index] += mask_outputs[global_index] ? 0.0 : inputs[global_index] * d_output[index];
            d_output[index] *= mask_outputs[global_index] ? 1.0 : 0.0;
        } else {
            d_output[index] = 0.;
        }
        d_hidden[index] = mask_hiddens[global_index] ? forget[index] * d_hidden[index] : 0.0;
        //printf("start step %ld; batch %ld; column %ld; index %ld; global_index %ld; d_inputs %f; new_d_hidden %f; d_hidden %f; hidden minus low %f\n", time_step, batch, column, index, global_index, d_inputs[global_index], new_d_hidden[index], d_hidden[index], hidden_minus1_low);
    }
}

} // namespace

std::vector<at::Tensor> forget_rkn_packed_cuda_forward(
    at::Tensor inputs,
    at::Tensor input_batch_sizes,
    at::Tensor forget,
    at::Tensor hidden,
    bool compute_la,
    bool additive) {
    // const int64_t threads = 1024;
    // const dim3 blocks()
    // std::vector<at::Tensor> hiddens;
    // std::vector<at::Tensor> step_outputs;
    // std::vector<at::Tensor> step_hiddens;
    auto outputs = at::zeros_like(inputs);
    auto hiddens = at::zeros_like(inputs);
    auto new_hidden = at::zeros_like(hidden);
    auto step_output = at::zeros_like(hidden);

    int64_t input_offset = 0;
    int64_t num_steps = input_batch_sizes.size(0);
    int64_t* batch_sizes = input_batch_sizes.data<int64_t>();
    // int64_t last_batch_size = batch_sizes[0];
    const auto kmer_size = hidden.size(2);
    const auto state_size = hidden.size(1);
    
    const int threads = 512;
    const int block_size = (state_size * kmer_size + threads - 1) / threads;

    for (int64_t i = 0; i < num_steps; ++i) {
        //printf("%ld\n", batch_sizes[i]);
        //printf("%ld", num_steps);
        //return {outputs, step_output, hiddens};
        int64_t batch_size = batch_sizes[i];
        //return {outputs, step_output, hiddens};
        // auto step_input = input.narrow(0, input_offset, batch_size);
        // input_offset += batch_size;

        // int64_t dec = last_batch_size - batch_size;
        // if (dec > 0) {
        //     if compute_la
        //         hiddens.push_back(hidden_slice(step_output, last_batch_size - dec, last_batch_size));
        //     else
        //         hiddens.push_back(hidden_slice(step_output, last_batch_size - dec, last_batch_size) / i);
        //     hidden = hidden_slice(hidden, 0, last_batch_size - dec);
        //     step_output = hidden_slice(step_output, 0, last_batch_size - dec);
        // }

        // last_batch_size = batch_size;

        const dim3 blocks(block_size, batch_size);
        // hidden, output = cell_(step_input, hidden, params);
        AT_DISPATCH_FLOATING_TYPES(hidden.type(), "forget_rkn_packed_forward_cuda", ([&] {
            forget_rkn_packed_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
                outputs.data<scalar_t>(),
                hiddens.data<scalar_t>(),
                step_output.data<scalar_t>(),
                new_hidden.data<scalar_t>(),
                inputs.data<scalar_t>(),
                hidden.data<scalar_t>(),
                forget.data<scalar_t>(),
                batch_size,
                state_size,
                kmer_size,
                input_offset,
                i,
                compute_la,
                additive);
        }));
        hidden = new_hidden.clone();
        input_offset += batch_size;
        // std::cout << input_offset;
        // if compute_la
        //     step_outputs.push_back(step_output.clone() / (i + 1));
        // else
        //     step_outputs.push_back(step_output.clone());
        // step_hiddens.push_back(hidden);
    }
    // hiddens.push_back(hidden);
    // std::reverse(hiddens.begin(), hiddens.end());
    return {outputs, step_output, hiddens, hidden};
}

std::vector<at::Tensor> forget_rkn_packed_cuda_backward(
    at::Tensor d_outputs,
    at::Tensor d_output,
    at::Tensor d_hidden,
    at::Tensor inputs,
    at::Tensor input_batch_sizes,
    at::Tensor hiddens,
    at::Tensor forget,
    bool compute_la,
    bool additive) {

    auto d_inputs = at::zeros_like(inputs);
    auto d_forget = at::zeros_like(forget);
    // auto d_hidden = at::zeros_like(d_output);
    auto new_d_hidden = at::zeros_like(d_output);

    int64_t input_offset = inputs.size(0);
    int64_t num_steps = input_batch_sizes.size(0);
    int64_t* batch_sizes = input_batch_sizes.data<int64_t>();
    // int64_t last_batch_size = batch_sizes[num_steps - 1];
    // int64_t num_steps = input_batch_sizes.size(0);

    // auto hidden = hidden_slice(input_hidden, 0, batch_sizes[num_steps - 1]);
    // auto grad_input = at::zeros_like(hidden);
    const auto kmer_size = inputs.size(2);
    const auto state_size = inputs.size(1);

    const int64_t threads = 512;
    const int64_t block_size = (state_size * kmer_size + threads - 1) / threads;

    for (int64_t i = num_steps - 1; i >= 0; --i) {
        int64_t batch_size = batch_sizes[i];
        int64_t next_batch_size = (i > 0) ? batch_sizes[i - 1] : 0;
        // int64_t inc = batch_size - last_batch_size;

        // if (inc > 0) {
        //     hidden = hidden_concat(ArrayRef<Tensor>{hidden, hidden_slice(input_hidden, last_batch_size, batch_size)});
        // }

        // auto step_grad_output = grad_output.narrow(0, input_offset - batch_size, batch_size);

        // last_batch_size = batch_size;
        const dim3 blocks(block_size, batch_size);
        input_offset -= batch_size;

        AT_DISPATCH_FLOATING_TYPES(d_hidden.type(), "forget_rkn_packed_backward_cuda", ([&] {
            forget_rkn_packed_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
                d_inputs.data<scalar_t>(),
                d_forget.data<scalar_t>(),
                d_hidden.data<scalar_t>(),
                new_d_hidden.data<scalar_t>(),
                d_output.data<scalar_t>(),
                d_outputs.data<scalar_t>(),
                inputs.data<scalar_t>(),
                hiddens.data<scalar_t>(),
                forget.data<scalar_t>(),
                batch_size,
                next_batch_size,
                state_size,
                kmer_size,
                input_offset,
                i,
                compute_la,
                additive);
        }));
        // d_hidden = new_d_hidden;
        // d_hidden = at::add(d_hidden, new_d_hidden);
        d_hidden += new_d_hidden;
        // d_hidden.copy_(new_d_hidden);
        // new_d_hidden = at::zero_(new_d_hidden);

        // grad_inputs.push_back(grad_input);
    }
    return {d_inputs, d_forget, d_hidden};
}

std::vector<at::Tensor> forget_rkn_cuda_forward(
    at::Tensor inputs,
    at::Tensor forget,
    at::Tensor hidden,
    bool compute_la,
    bool additive) {
    // inputs: H x B x dim x kmer_size
    // hidden: B x dim x kmer_size

    auto outputs = at::zeros_like(inputs);
    auto hiddens = at::zeros_like(inputs);
    // auto new_hidden = at::zeros_like(hidden);
    auto output = at::zeros_like(hidden);

    // int64_t input_offset = 0;
    int64_t num_steps = inputs.size(0);

    const auto kmer_size = hidden.size(2);
    const auto state_size = hidden.size(1);
    const auto batch_size = hidden.size(0);
    
    const int threads = 512;
    const int block_size = (state_size * kmer_size + threads - 1) / threads;
    const dim3 blocks(block_size, batch_size);

    for (int64_t i = 0; i < num_steps; ++i) {
        AT_DISPATCH_FLOATING_TYPES(hidden.type(), "forget_rkn_forward_cuda", ([&] {
            forget_rkn_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
                outputs[i].data<scalar_t>(),
                hiddens[i].data<scalar_t>(),
                output.data<scalar_t>(),
                // new_hidden.data<scalar_t>(),
                inputs[i].data<scalar_t>(),
                hidden.data<scalar_t>(),
                forget.data<scalar_t>(),
                batch_size,
                state_size,
                kmer_size,
                i,
                compute_la,
                additive);
        }));
        hidden = hiddens[i];
        // input_offset += batch_size;
    }

    return {outputs, hiddens};
}

std::vector<at::Tensor> forget_rkn_cuda_backward(
    at::Tensor d_outputs,
    at::Tensor d_output,
    at::Tensor d_hidden,
    at::Tensor inputs,
    at::Tensor hiddens,
    at::Tensor forget,
    bool compute_la,
    bool additive) {
    // inputs: H x B x dim x kmer_size
    // hidden: B x dim x kmer_size

    auto d_inputs = at::zeros_like(inputs);
    auto d_forget = at::zeros_like(forget);
    // auto d_hidden = at::zeros_like(d_output);
    auto new_d_hidden = at::zeros_like(d_hidden);

    int64_t num_steps = inputs.size(0);
    // int64_t last_batch_size = batch_sizes[num_steps - 1];
    // int64_t num_steps = input_batch_sizes.size(0);

    // auto hidden = hidden_slice(input_hidden, 0, batch_sizes[num_steps - 1]);
    // auto grad_input = at::zeros_like(hidden);
    const auto kmer_size = d_hidden.size(2);
    const auto state_size = d_hidden.size(1);
    const auto batch_size = d_hidden.size(0);

    const int64_t threads = 512;
    const int64_t block_size = (state_size * kmer_size + threads - 1) / threads;
    const dim3 blocks(block_size, batch_size);

    for (int64_t i = num_steps - 1; i >= 0; --i) {
        // const auto hidden_minus1 = (i > 0) ? hiddens[i - 1] : NULL;
        if (i > 0){
            AT_DISPATCH_FLOATING_TYPES(d_hidden.type(), "forget_rkn_backward_cuda", ([&] {
            forget_rkn_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
                d_inputs[i].data<scalar_t>(),
                d_forget.data<scalar_t>(),
                d_hidden.data<scalar_t>(),
                new_d_hidden.data<scalar_t>(),
                d_output.data<scalar_t>(),
                d_outputs[i].data<scalar_t>(),
                inputs[i].data<scalar_t>(),
                // hiddens[i].data<scalar_t>(),
                hiddens[i-1].data<scalar_t>(),
                forget.data<scalar_t>(),
                batch_size,
                state_size,
                kmer_size,
                i,
                compute_la,
                additive);
            }));    
        } else {
            AT_DISPATCH_FLOATING_TYPES(d_hidden.type(), "forget_rkn_backward_cuda", ([&] {
            forget_rkn_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
                d_inputs[i].data<scalar_t>(),
                d_forget.data<scalar_t>(),
                d_hidden.data<scalar_t>(),
                new_d_hidden.data<scalar_t>(),
                d_output.data<scalar_t>(),
                d_outputs[i].data<scalar_t>(),
                inputs[i].data<scalar_t>(),
                // hiddens[i].data<scalar_t>(),
                NULL,
                forget.data<scalar_t>(),
                batch_size,
                state_size,
                kmer_size,
                i,
                compute_la,
                additive);
            }));
        }
        d_hidden += new_d_hidden;
    }
    return {d_inputs, d_forget, d_hidden};
}

std::vector<at::Tensor> forget_rkn_max_cuda_forward(
    at::Tensor inputs,
    at::Tensor forget,
    at::Tensor hidden,
    bool compute_la,
    bool additive) {
    // inputs: H x B x dim x kmer_size
    // hidden: B x dim x kmer_size

    auto outputs = at::zeros_like(inputs);
    auto hiddens = at::zeros_like(inputs);
    auto mask_outputs = at::zeros_like(inputs).to(at::kByte);//, at::kByte);
    auto mask_hiddens = at::zeros_like(hiddens).to(at::kByte);//, at::kByte);
    // auto new_hidden = at::zeros_like(hidden);
    auto output = at::zeros_like(hidden);

    // int64_t input_offset = 0;
    int64_t num_steps = inputs.size(0);

    const auto kmer_size = hidden.size(2);
    const auto state_size = hidden.size(1);
    const auto batch_size = hidden.size(0);
    
    const int threads = 512;
    const int block_size = (state_size * kmer_size + threads - 1) / threads;
    const dim3 blocks(block_size, batch_size);

    for (int64_t i = 0; i < num_steps; ++i) {
        AT_DISPATCH_FLOATING_TYPES(hidden.type(), "forget_rkn_max_forward_cuda", ([&] {
            forget_rkn_max_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
                outputs[i].data<scalar_t>(),
                hiddens[i].data<scalar_t>(),
                output.data<scalar_t>(),
                // new_hidden.data<scalar_t>(),
                inputs[i].data<scalar_t>(),
                hidden.data<scalar_t>(),
                forget.data<scalar_t>(),
                mask_hiddens[i].data<uint8_t>(),
                mask_outputs[i].data<uint8_t>(),
                batch_size,
                state_size,
                kmer_size,
                i,
                compute_la,
                additive);
        }));
        hidden = hiddens[i];
        // input_offset += batch_size;
    }

    return {outputs, hiddens, mask_outputs, mask_hiddens};
}

std::vector<at::Tensor> forget_rkn_max_cuda_backward(
    at::Tensor d_outputs,
    at::Tensor d_output,
    at::Tensor d_hidden,
    at::Tensor inputs,
    at::Tensor hiddens,
    at::Tensor forget,
    at::Tensor mask_outputs,
    at::Tensor mask_hiddens,
    bool compute_la,
    bool additive) {
    // inputs: H x B x dim x kmer_size
    // hidden: B x dim x kmer_size

    auto d_inputs = at::zeros_like(inputs);
    auto d_forget = at::zeros_like(forget);
    // auto d_hidden = at::zeros_like(d_output);
    auto new_d_hidden = at::zeros_like(d_hidden);

    int64_t num_steps = inputs.size(0);
    // int64_t last_batch_size = batch_sizes[num_steps - 1];
    // int64_t num_steps = input_batch_sizes.size(0);

    const auto kmer_size = d_hidden.size(2);
    const auto state_size = d_hidden.size(1);
    const auto batch_size = d_hidden.size(0);

    const int64_t threads = 512;
    const int64_t block_size = (state_size * kmer_size + threads - 1) / threads;
    const dim3 blocks(block_size, batch_size);

    for (int64_t i = num_steps - 1; i >= 0; --i) {
        // const auto hidden_minus1 = (i > 0) ? hiddens[i - 1] : NULL;
        if (i > 0){
            AT_DISPATCH_FLOATING_TYPES(d_hidden.type(), "forget_rkn_max_backward_cuda", ([&] {
            forget_rkn_max_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
                d_inputs[i].data<scalar_t>(),
                d_forget.data<scalar_t>(),
                d_hidden.data<scalar_t>(),
                new_d_hidden.data<scalar_t>(),
                d_output.data<scalar_t>(),
                d_outputs[i].data<scalar_t>(),
                inputs[i].data<scalar_t>(),
                // hiddens[i].data<scalar_t>(),
                hiddens[i-1].data<scalar_t>(),
                forget.data<scalar_t>(),
                mask_hiddens[i].data<uint8_t>(),
                mask_outputs[i].data<uint8_t>(),
                batch_size,
                state_size,
                kmer_size,
                i,
                compute_la,
                additive);
            }));    
        } else {
            AT_DISPATCH_FLOATING_TYPES(d_hidden.type(), "forget_rkn_max_backward_cuda", ([&] {
            forget_rkn_max_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
                d_inputs[i].data<scalar_t>(),
                d_forget.data<scalar_t>(),
                d_hidden.data<scalar_t>(),
                new_d_hidden.data<scalar_t>(),
                d_output.data<scalar_t>(),
                d_outputs[i].data<scalar_t>(),
                inputs[i].data<scalar_t>(),
                // hiddens[i].data<scalar_t>(),
                NULL,
                forget.data<scalar_t>(),
                mask_hiddens[i].data<uint8_t>(),
                mask_outputs[i].data<uint8_t>(),
                batch_size,
                state_size,
                kmer_size,
                i,
                compute_la,
                additive);
            }));
        }
        d_hidden += new_d_hidden;
    }
    return {d_inputs, d_forget, d_hidden};
}

std::vector<at::Tensor> forget_rkn_packed_max_cuda_forward(
    at::Tensor inputs,
    at::Tensor input_batch_sizes,
    at::Tensor forget,
    at::Tensor hidden,
    bool compute_la,
    bool additive) {

    auto outputs = at::zeros_like(inputs);
    auto hiddens = at::zeros_like(inputs);
    auto new_hidden = at::zeros_like(hidden);
    auto step_output = at::zeros_like(hidden);

    auto mask_outputs = at::zeros_like(inputs).to(at::kByte);
    auto mask_hiddens = at::zeros_like(hiddens).to(at::kByte);

    int64_t input_offset = 0;
    int64_t num_steps = input_batch_sizes.size(0);
    int64_t* batch_sizes = input_batch_sizes.data<int64_t>();

    const auto kmer_size = hidden.size(2);
    const auto state_size = hidden.size(1);
    
    const int threads = 512;
    const int block_size = (state_size * kmer_size + threads - 1) / threads;

    for (int64_t i = 0; i < num_steps; ++i) {
        int64_t batch_size = batch_sizes[i];

        const dim3 blocks(block_size, batch_size);

        AT_DISPATCH_FLOATING_TYPES(hidden.type(), "forget_rkn_packed_max_forward_cuda", ([&] {
            forget_rkn_packed_max_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
                outputs.data<scalar_t>(),
                hiddens.data<scalar_t>(),
                step_output.data<scalar_t>(),
                new_hidden.data<scalar_t>(),
                inputs.data<scalar_t>(),
                hidden.data<scalar_t>(),
                forget.data<scalar_t>(),
                mask_outputs.data<uint8_t>(),
                mask_hiddens.data<uint8_t>(),
                batch_size,
                state_size,
                kmer_size,
                input_offset,
                i,
                compute_la,
                additive);
        }));
        hidden = new_hidden.clone();
        input_offset += batch_size;

    }
    return {outputs, step_output, hiddens, hidden, mask_outputs, mask_hiddens};
}

std::vector<at::Tensor> forget_rkn_packed_max_cuda_backward(
    at::Tensor d_outputs,
    at::Tensor d_output,
    at::Tensor d_hidden,
    at::Tensor inputs,
    at::Tensor input_batch_sizes,
    at::Tensor hiddens,
    at::Tensor forget,
    at::Tensor mask_outputs,
    at::Tensor mask_hiddens,
    bool compute_la,
    bool additive) {

    auto d_inputs = at::zeros_like(inputs);
    auto d_forget = at::zeros_like(forget);
    auto new_d_hidden = at::zeros_like(d_output);

    int64_t input_offset = inputs.size(0);
    int64_t num_steps = input_batch_sizes.size(0);
    int64_t* batch_sizes = input_batch_sizes.data<int64_t>();

    const auto kmer_size = inputs.size(2);
    const auto state_size = inputs.size(1);

    const int64_t threads = 512;
    const int64_t block_size = (state_size * kmer_size + threads - 1) / threads;

    for (int64_t i = num_steps - 1; i >= 0; --i) {
        int64_t batch_size = batch_sizes[i];
        int64_t next_batch_size = (i > 0) ? batch_sizes[i - 1] : 0;

        const dim3 blocks(block_size, batch_size);
        input_offset -= batch_size;

        AT_DISPATCH_FLOATING_TYPES(d_hidden.type(), "forget_rkn_packed_max_backward_cuda", ([&] {
            forget_rkn_packed_max_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
                d_inputs.data<scalar_t>(),
                d_forget.data<scalar_t>(),
                d_hidden.data<scalar_t>(),
                new_d_hidden.data<scalar_t>(),
                d_output.data<scalar_t>(),
                d_outputs.data<scalar_t>(),
                inputs.data<scalar_t>(),
                hiddens.data<scalar_t>(),
                forget.data<scalar_t>(),
                mask_outputs.data<uint8_t>(),
                mask_hiddens.data<uint8_t>(),
                batch_size,
                next_batch_size,
                state_size,
                kmer_size,
                input_offset,
                i,
                compute_la,
                additive);
        }));
        d_hidden += new_d_hidden;
    }
    return {d_inputs, d_forget, d_hidden};
}

// template <typename scalar_t>
// __global__ void forget_rkn_forward_kernel(
//     scalar_t* __restrict__ outputs,
//     scalar_t* __restrict__ hiddens,
//     scalar_t* __restrict__ output,
//     scalar_t* __restrict__ hidden,
//     const scalar_t* __restrict__ inputs,
//     const int64_t* __restrict__ batch_sizes,
//     const scalar_t* __restrict__ forget,
//     size_t num_steps,
//     size_t state_size,
//     size_t kmer_size,
//     bool compute_la) {
//     const int64_t column = blockIdx.x * blockDim.x + threadIdx.x;
//     const int64_t batch = blockIdx.y * blockDim.y + threadIdx.y;
//     const int64_t block_size = state_size * kmer_size;

//     if column >= block_size
//         return;
//     // int64_t last_batch_size = batch_sizes[0];
//     int64_t input_offset = 0;

//     for (int64_t i = 0; i < num_steps; ++i) {
//         int64_t batch_size = batch_sizes[i];
//         if batch > batch_size
//             return;
//         // int64_t dec = last_batch_size - batch_size;
//         int64_t index = (input_offset + batch) * block_size + column;
//         int64_t block_index = batch * block_size + column;
//         int64_t block_index_low = block_index - batch_size * state_size;
//         const auto hidden_low = (block_index_low < 0) ? 1.0 : hidden[block_index_low];
//         input_offset += batch_size;

//         hidden[block_index] *= forget[block_index];
//         hidden[block_index] += (1 - forget[block_index]) * inputs[index] * hidden_low;
//         hiddens[index] = hidden[block_index];

//         if (compute_la) {
//             output[block_index] += inputs[index] * hidden_low;
//             outputs[index] = output[block_index] / (i + 1.);
//         }
//         else {
//             output[block_index] = hidden[block_index];
//             outputs[index] = output[block_index];
//         }

//     }
// }

// template <typename scalar_t>
// __global__ void forget_rkn_backward_kernel(
//     scalar_t* __restrict__ d_inputs,
//     scalar_t* __restrict__ d_forget,
//     scalar_t* __restrict__ d_hidden,
//     const scalar_t* __restrict__ d_outputs,
//     const scalar_t* __restrict__ d_output,
//     const scalar_t* __restrict__ outputs,
//     const scalar_t* __restrict__ hiddens,
//     const scalar_t* __restrict__ inputs,
//     const int64_t* __restrict__ batch_sizes,
//     const scalar_t* __restrict__ forget,
//     size_t num_steps,
//     size_t state_size,
//     size_t kmer_size,
//     bool compute_la) {
//     const int64_t column = blockIdx.x * blockDim.x + threadIdx.x;
//     const int64_t batch = blockIdx.y * blockDim.y + threadIdx.y;
//     const int64_t block_size = state_size * kmer_size;

//     if column >= block_size
//         return;
//     int64_t input_offset

//     for (int64_t i = num_steps - 1; i >= 0; --i) {
//         int64_t batch_size = batch_sizes[i];
//         if batch > batch_size
//             continue;
//         input_offset -= batch_size;
//         int64_t index = (input_offset + batch) * block_size + column;
//         int64_t block_index = batch * block_size + column;
//         int64_t block_index_low = block_index - batch_size * state_size;

//         d_output[block_index] += d_outputs[index];
//         if (compute_la) {

//         }
//     }
// }
