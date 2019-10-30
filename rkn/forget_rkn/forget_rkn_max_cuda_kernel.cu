#include <ATen/ATen.h>
// #include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <stdio.h>

namespace {
template <typename scalar_t>
__global__ void forget_rkn_max_cuda_forward1_kernel(
    // scalar_t* __restrict__ outputs,
    // scalar_t* __restrict__ hiddens,
    // scalar_t* __restrict__ new_output,
    scalar_t* __restrict__ new_hidden,
    scalar_t* __restrict__ step_output,
    const scalar_t* __restrict__ step_input,
    const scalar_t* __restrict__ hidden,
    const scalar_t* __restrict__ forget,
    scalar_t* __restrict__ aux,
    uint8_t* __restrict__ mask_hidden,
    // uint8_t* __restrict__ mask_output,
    size_t batch_size,
    size_t state_size,
    size_t kmer_size,
    // int64_t time_step,
    bool compute_la,
    bool additive) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int block_size = state_size * kmer_size;
    // const int64_t row = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch = blockIdx.y * blockDim.y + threadIdx.y;
    const int64_t index = batch * block_size + column;
    // const int64_t low_index = index - 1;
    const auto hidden_low = (index % kmer_size == 0) ? 1.0 : hidden[index - 1];

    if (column < block_size && batch < batch_size) {
        // new_hidden[index] = forget[index] * hidden[index];
        if (additive)
            aux[index] = step_input[index] + hidden_low;
        else
            aux[index] = step_input[index] * hidden_low;

        if (compute_la) {
            new_hidden[index] = forget[index] * hidden[index];
            mask_hidden[index] = new_hidden[index] > aux[index];
            new_hidden[index] = fmax(new_hidden[index], aux[index]);
        } else {
            step_output[index] = forget[index] * hidden[index];
            mask_hidden[index] = step_output[index] > aux[index];
            new_hidden[index] = fmax(step_output[index], aux[index]);
        }

    }
}

template <typename scalar_t>
__global__ void forget_rkn_max_cuda_forward2_kernel(
    // scalar_t* __restrict__ outputs,
    // scalar_t* __restrict__ hiddens,
    // scalar_t* __restrict__ new_output,
    scalar_t* __restrict__ hiddens,
    scalar_t* __restrict__ outputs,
    // scalar_t* __restrict__ output,
    scalar_t* __restrict__ aux,
    // uint8_t* __restrict__ mask_hidden,
    uint8_t* __restrict__ mask_outputs,
    size_t batch_size,
    size_t state_size,
    size_t kmer_size,
    int64_t time_step,
    bool compute_la,
    bool additive,
    bool has_lintrans) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int block_size = state_size * kmer_size;
    // const int64_t row = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch = blockIdx.y * blockDim.y + threadIdx.y;
    const int64_t index = batch * block_size + column;

    const int64_t global_block = block_size * batch_size;
    int64_t current_index = 0;
    auto output = 0.0;

    if (column < block_size && batch < batch_size) {
        for (int64_t i = 0; i < time_step; ++i) {
            current_index = i * global_block + index;
            if (!compute_la)
                output = outputs[current_index];
            if (compute_la || has_lintrans) {
                mask_outputs[current_index] = output > aux[current_index];
                output = fmax(output, aux[current_index]);
            } else {
                output = hiddens[current_index];
            }
            outputs[current_index] = output;
        }

    }
}

template <typename scalar_t>
__global__ void forget_rkn_max_cuda_backward1_kernel(
    scalar_t* __restrict__ d_inputs,
    // scalar_t* __restrict__ d_step_input,
    // scalar_t* __restrict__ d_forget,
    // scalar_t* __restrict__ d_hidden,
    // scalar_t* __restrict__ new_d_hidden,
    // scalar_t* __restrict__ d_hiddens,
    scalar_t* __restrict__ d_output,
    const scalar_t* __restrict__ d_outputs,
    // const scalar_t* __restrict__ outputs,
    // const scalar_t* __restrict__ inputs,
    // const scalar_t* __restrict__ hiddens,
    const scalar_t* __restrict__ inputs,
    // const scalar_t* __restrict__ hidden,
    const scalar_t* __restrict__ hiddens,
    // const scalar_t* __restrict__ forget,
    // const uint8_t* __restrict__ mask_hidden,
    const uint8_t* __restrict__ mask_outputs,
    scalar_t* __restrict__ aux,
    scalar_t* __restrict__ d_aux,
    int64_t batch_size,
    size_t state_size,
    size_t kmer_size,
    int64_t time_step,
    bool compute_la,
    bool has_lintrans) {

    const int64_t column = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t block_size = state_size * kmer_size;
    const int64_t batch = blockIdx.y * blockDim.y + threadIdx.y;
    const int64_t index = batch * block_size + column;
    // const int64_t low_index = index - 1;
    int64_t current_index = 0;

    const int64_t global_block = batch_size * block_size;
    // auto d_output = 0.0;

    // auto hidden_minus1_val = 0.0;
    auto hidden_minus1_low = 0.0;// = (index % kmer_size == 0) ? 1.0 : 0.0;
    // if (time_step >= 1) {
    //     // hidden_minus1_val = hidden_minus1[index];
    //     hidden_minus1_low = (index % kmer_size == 0) ? 1.0 : hidden_minus1[index - 1];
    // }

    if (column < block_size && batch < batch_size) {
        // d_output[index] += d_step_output[index];
        // if (compute_la || has_lintrans) {
        //     d_aux[index] = mask_output[index] ? 0.0 : d_output[index];
        //     aux[index] = step_input[index] * hidden_minus1_low;
        //     d_output[index] *= mask_output[index] ? 1.0 : 0.0;
        // } else {
        //     d_aux[index] = 0.0;
        //     d_hidden[index] += d_output[index];
        //     d_output[index] = 0.0;
        // }
        for (int64_t i = time_step - 1; i >= 0; --i) {
            current_index = i * global_block + index;
            if (i >= 1)
                hidden_minus1_low = (index % kmer_size == 0) ? 1.0 : hiddens[current_index - global_block - 1];
            else
                hidden_minus1_low = (index % kmer_size == 0) ? 1.0 : 0.0;
            d_output[index] += d_outputs[current_index];
            if (compute_la || has_lintrans) {
                d_aux[current_index] = mask_outputs[current_index] ? 0.0 : d_output[index];
                aux[current_index] = inputs[current_index] * hidden_minus1_low;
                d_output[index] *= mask_outputs[current_index] ? 1.0 : 0.0;
                d_inputs[current_index] = d_output[index];
            } else {
                d_aux[current_index] = 0.0;
                d_inputs[current_index] += d_output[index];
                d_output[index] = 0.0;
            }
            if (!compute_la) {
                d_output[index] = 0.0;
            }
            
        }


    }
}

template <typename scalar_t>
__global__ void forget_rkn_max_cuda_backward2_kernel(
    // scalar_t* __restrict__ d_inputs,
    scalar_t* __restrict__ d_step_input,
    scalar_t* __restrict__ d_forget,
    scalar_t* __restrict__ d_hidden,
    scalar_t* __restrict__ new_d_hidden,
    // scalar_t* __restrict__ d_hiddens,
    scalar_t* __restrict__ d_output,
    // const scalar_t* __restrict__ d_step_output,
    // const scalar_t* __restrict__ outputs,
    // const scalar_t* __restrict__ inputs,
    // const scalar_t* __restrict__ hiddens,
    const scalar_t* __restrict__ step_input,
    // const scalar_t* __restrict__ hidden,
    const scalar_t* __restrict__ hidden_minus1,
    const scalar_t* __restrict__ forget,
    const uint8_t* __restrict__ mask_hidden,
    // const uint8_t* __restrict__ mask_output,
    scalar_t* __restrict__ d_aux,
    int64_t batch_size,
    size_t state_size,
    size_t kmer_size,
    int64_t time_step,
    bool compute_la,
    bool additive,
    bool has_lintrans) {

    const int64_t column = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t block_size = state_size * kmer_size;
    const int64_t batch = blockIdx.y * blockDim.y + threadIdx.y;
    const int64_t index = batch * block_size + column;
    const int64_t low_index = index - 1;

    auto hidden_minus1_val = 0.0;
    auto hidden_minus1_low = (index % kmer_size == 0) ? 1.0 : 0.0;
    if (time_step >= 1) {
        hidden_minus1_val = hidden_minus1[index];
        hidden_minus1_low = (index % kmer_size == 0) ? 1.0 : hidden_minus1[index - 1];
    }

    if (column < block_size && batch < batch_size) {
        // d_aux[index] += mask_hidden[index] ? 0.0 : d_hidden[index];
        // d_hidden[index] *= mask_hidden[index] ? 1.0 : 0.0;
        // d_forget[index] += d_hidden[index] * hidden_minus1_val;
        // d_hidden[index] *= forget[index];
        if (compute_la || has_lintrans) {
            d_output[index] = d_step_input[index];
        } else {
            d_hidden[index] += d_step_input[index];
            d_output[index] = 0.0;
        }
        if (compute_la) {
            d_aux[index] += mask_hidden[index] ? 0.0 : d_hidden[index];
            d_hidden[index] *= mask_hidden[index] ? 1.0 : 0.0;
            d_forget[index] += d_hidden[index] * hidden_minus1_val;
            d_hidden[index] *= forget[index];
        } else {
            d_aux[index] += mask_hidden[index] ? 0.0 : d_hidden[index];
            d_output[index] += mask_hidden[index] ? d_hidden[index] : 0.0;
            d_forget[index] += d_output[index] * hidden_minus1_val;
            d_hidden[index] = d_output[index] * forget[index];
        }

        if (additive) {
            d_step_input[index] = d_aux[index];
            if (index % kmer_size != 0) {
                new_d_hidden[low_index] = d_aux[index];
            }
        } else {
            d_step_input[index] = d_aux[index] * hidden_minus1_low;
            if (index % kmer_size != 0) {
                new_d_hidden[low_index] = d_aux[index] * step_input[index];
            }
        }

    }
}

template <typename scalar_t>
__global__ void forget_rkn_packed_max_cuda_forward1_kernel(
    // scalar_t* __restrict__ outputs,
    scalar_t* __restrict__ hiddens,
    scalar_t* __restrict__ outputs,
    scalar_t* __restrict__ new_hidden,
    const scalar_t* __restrict__ inputs,
    const scalar_t* __restrict__ hidden,
    const scalar_t* __restrict__ forget,
    // uint8_t* __restrict__ mask_outputs,
    uint8_t* __restrict__ mask_hiddens,
    scalar_t* __restrict__ aux,
    size_t batch_size,
    size_t state_size,
    size_t kmer_size,
    int64_t input_offset,
    // int64_t time_step,
    bool compute_la,
    bool additive) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int block_size = state_size * kmer_size;
    const int batch = blockIdx.y * blockDim.y + threadIdx.y;

    const int64_t index = batch * block_size + column;
    const auto hidden_low = (index % kmer_size == 0) ? 1.0 : hidden[index - 1];
    const int64_t global_index = input_offset * block_size + index;
    // auto aux = hidden_low;

    if (column < block_size && batch < batch_size) {
        // new_hidden[index] = forget[index] * hidden[index];
        if (additive)
            aux[global_index] = inputs[global_index] + hidden_low;
        else
            aux[global_index] = inputs[global_index] * hidden_low;

        if (compute_la) {
            new_hidden[index] = forget[index] * hidden[index];
            mask_hiddens[global_index] = new_hidden[index] > aux[global_index];
            new_hidden[index] = fmax(new_hidden[index], aux[global_index]);
            hiddens[global_index] = new_hidden[index];
        } else {
            outputs[global_index] = forget[index] * hidden[index];
            mask_hiddens[global_index] = outputs[global_index] > aux[global_index];
            new_hidden[index] = fmax(outputs[global_index], aux[global_index]);
            hiddens[global_index] = new_hidden[index];
        }

    }
}

template <typename scalar_t>
__global__ void forget_rkn_packed_max_cuda_forward2_kernel(
    scalar_t* __restrict__ outputs,
    scalar_t* __restrict__ hiddens,
    scalar_t* __restrict__ step_output,
    // scalar_t* __restrict__ new_hidden,
    // const scalar_t* __restrict__ inputs,
    // const scalar_t* __restrict__ hidden,
    const scalar_t* __restrict__ forget,
    uint8_t* __restrict__ mask_outputs,
    // uint8_t* __restrict__ mask_hiddens,
    scalar_t* __restrict__ aux,
    size_t batch_size,
    size_t state_size,
    size_t kmer_size,
    int64_t input_offset,
    // int64_t time_step,
    bool compute_la,
    bool additive,
    bool has_lintrans) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int block_size = state_size * kmer_size;
    const int batch = blockIdx.y * blockDim.y + threadIdx.y;

    const int64_t index = batch * block_size + column;
    // const auto hidden_low = (index % kmer_size == 0) ? 1.0 : hidden[index - 1];
    const int64_t global_index = input_offset * block_size + index;
    // auto aux = hidden_low;

    if (column < block_size && batch < batch_size) {
        if (!compute_la)
            step_output[index] = outputs[global_index];
        if (compute_la || has_lintrans) {
            mask_outputs[global_index] = step_output[index] > aux[global_index];
            step_output[index] = fmax(step_output[index], aux[global_index]);
        }
        else {
            step_output[index] = hiddens[global_index];
        }
        outputs[global_index] = step_output[index];
    }
}

template <typename scalar_t>
__global__ void forget_rkn_packed_max_cuda_backward1_kernel(
    scalar_t* __restrict__ d_inputs,
    // scalar_t* __restrict__ d_forget,
    // scalar_t* __restrict__ d_hidden,
    // scalar_t* __restrict__ new_d_hidden,
    scalar_t* __restrict__ d_output,
    const scalar_t* __restrict__ d_outputs,
    const scalar_t* __restrict__ inputs,
    const scalar_t* __restrict__ hiddens,
    // const scalar_t* __restrict__ forget,
    const uint8_t* __restrict__ mask_outputs,
    // const uint8_t* __restrict__ mask_hiddens,
    scalar_t* __restrict__ aux,
    scalar_t* __restrict__ d_aux,
    int64_t batch_size,
    int64_t next_batch_size,
    size_t state_size,
    size_t kmer_size,
    int64_t input_offset,
    // int64_t time_step,
    bool compute_la,
    bool additive,
    bool has_lintrans) {
    const int64_t column = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t block_size = state_size * kmer_size;
    const int64_t batch = blockIdx.y * blockDim.y + threadIdx.y;
    const int64_t index = batch * block_size + column;
    // const int64_t low_index = index - 1;
    const int64_t global_index = input_offset * block_size + index;
    const int64_t global_index_minus1 = global_index - next_batch_size * block_size;
    const int64_t threshold = (input_offset - next_batch_size) * block_size;
    // const auto hidden_minus1 = (next_batch_size > 0) ? hiddens[global_index_minus1] : 0.0;
    auto hidden_minus1_low = 1.0;
    if ((global_index_minus1 - threshold) % kmer_size != 0)
        hidden_minus1_low = (next_batch_size > 0) ? hiddens[global_index_minus1 - 1] : 0.0;

    if (column < block_size && batch < batch_size) {
        d_output[index] += d_outputs[global_index];
        if (compute_la || has_lintrans) {
            d_aux[global_index] = mask_outputs[global_index] ? 0.0 : d_output[index];
            aux[global_index] = hidden_minus1_low * inputs[global_index];
            d_output[index] *= mask_outputs[global_index] ? 1.0 : 0.0;
            d_inputs[global_index] = d_output[index];
        } else {
            d_aux[global_index] = 0.0;
            d_inputs[global_index] += d_output[index];
            d_output[index] = 0.0;
        }
        if (!compute_la) {
            d_output[index] = 0.0;
        }
    }
}

template <typename scalar_t>
__global__ void forget_rkn_packed_max_cuda_backward2_kernel(
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
    scalar_t* __restrict__ d_aux,
    int64_t batch_size,
    int64_t next_batch_size,
    size_t state_size,
    size_t kmer_size,
    int64_t input_offset,
    int64_t time_step,
    bool compute_la,
    bool additive,
    bool has_lintrans) {
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
        if (compute_la || has_lintrans) {
            d_output[index] = d_inputs[global_index];
        } else {
            d_hidden[index] += d_inputs[global_index];
            d_output[index] = 0.0;
        }
        if (compute_la) {
            d_aux[global_index] += mask_hiddens[global_index] ? 0.0 : d_hidden[index];
            d_hidden[index] *= mask_hiddens[global_index] ? 1.0 : 0.0;
            d_forget[index] += d_hidden[index] * hidden_minus1;
            d_hidden[index] *= forget[index];
        } else {
            d_aux[global_index] += mask_hiddens[global_index] ? 0.0 : d_hidden[index];
            d_output[index] += mask_hiddens[global_index] ? d_hidden[index] : 0.0;
            d_forget[index] += d_output[index] * hidden_minus1;
            d_hidden[index] = d_output[index] * forget[index];
        }

        if (additive) {
            d_inputs[global_index] = d_aux[global_index];
            if (index % kmer_size != 0) {
                new_d_hidden[low_index] = d_aux[global_index];
            }
        } else {
            d_inputs[global_index] = d_aux[global_index] * hidden_minus1_low;
            if (index % kmer_size != 0) {
                new_d_hidden[low_index] = d_aux[global_index] * inputs[global_index];
            }
        }
        // printf("start step %ld; batch %ld; column %ld; index %ld; global_index %ld; d_inputs %f; new_d_hidden %f; d_hidden %f; hidden minus low %f; d_aux %f; mask hidden %d; mask outputs %d\n", time_step, batch, column, index, global_index, d_inputs[global_index], new_d_hidden[index], d_hidden[index], hidden_minus1_low, d_aux[global_index], mask_hiddens[global_index], mask_outputs[global_index]);
        // if (additive) {
        //     d_inputs[global_index] += mask_hiddens[global_index] ? 0.0 : d_hidden[index];
        //     // d_forget[index] += (hidden_minus1 - inputs[global_index] - hidden_minus1_low) * d_hidden[index];
        // } else{
        //     d_inputs[global_index] += mask_hiddens[global_index] ? 0.0 : hidden_minus1_low * d_hidden[index];
        //     // d_forget[index] += (hidden_minus1 - inputs[global_index] * hidden_minus1_low) * d_hidden[index];
        // }
        // d_forget[index] += mask_hiddens[global_index] ? hidden_minus1 * d_hidden[index] : 0.0; 

        // if (index % kmer_size != 0)
        //     if (additive)
        //         new_d_hidden[low_index] = mask_hiddens[global_index] ? 0.0 : d_hidden[index];
        //     else
        //         new_d_hidden[low_index] = mask_hiddens[global_index] ? 0.0 : inputs[global_index] * d_hidden[index];
        // if (compute_la) {
        //     if (index % kmer_size != 0)
        //         if (additive)
        //             new_d_hidden[low_index] += mask_outputs[global_index] ? 0.0 : d_output[index];
        //         else
        //             new_d_hidden[low_index] += mask_outputs[global_index] ? 0.0 : inputs[global_index] * d_output[index];
        //     d_output[index] *= mask_outputs[global_index] ? 1.0 : 0.0;
        // } else {
        //     d_output[index] = 0.;
        // }
        // d_hidden[index] = mask_hiddens[global_index] ? forget[index] * d_hidden[index] : 0.0;
    }
}

} // namespace

std::vector<at::Tensor> forget_rkn_max_cuda_forward(
    at::Tensor inputs,
    at::Tensor forget,
    at::Tensor hidden,
    bool compute_la,
    bool additive,
    at::Tensor lintrans) {
    // inputs: H x B x dim x kmer_size
    // hidden: B x dim x kmer_size
    bool has_lintrans = lintrans.numel() != 0;
    auto outputs = at::zeros_like(inputs);
    auto hiddens = at::zeros_like(inputs);
    auto mask_outputs = at::zeros_like(inputs).to(at::kByte);
    auto mask_hiddens = at::zeros_like(hiddens).to(at::kByte);
    // auto new_hidden = at::zeros_like(hidden);
    auto output = at::zeros_like(hidden);
    auto aux = at::zeros_like(inputs);

    // int64_t input_offset = 0;
    int64_t num_steps = inputs.size(0);

    const auto kmer_size = hidden.size(2);
    const auto state_size = hidden.size(1);
    const auto batch_size = hidden.size(0);
    
    const int threads = 512;
    const int block_size = (state_size * kmer_size + threads - 1) / threads;
    const dim3 blocks(block_size, batch_size);

    for (int64_t i = 0; i < num_steps; ++i) {
        AT_DISPATCH_FLOATING_TYPES(hidden.type(), "forget_rkn_max_forward1_cuda", ([&] {
            forget_rkn_max_cuda_forward1_kernel<scalar_t><<<blocks, threads>>>(
                // outputs[i].data<scalar_t>(),
                hiddens[i].data<scalar_t>(),
                outputs[i].data<scalar_t>(),
                // new_hidden.data<scalar_t>(),
                inputs[i].data<scalar_t>(),
                hidden.data<scalar_t>(),
                forget.data<scalar_t>(),
                aux[i].data<scalar_t>(),
                mask_hiddens[i].data<uint8_t>(),
                // mask_outputs[i].data<uint8_t>(),
                batch_size,
                state_size,
                kmer_size,
                // i,
                compute_la,
                additive);
        }));
        hidden = hiddens[i];
    }

    if (has_lintrans) {
        aux = at::tensordot(aux, lintrans, 2, 0).transpose(2, 3);
        aux = aux.contiguous();
    }

    AT_DISPATCH_FLOATING_TYPES(hidden.type(), "forget_rkn_max_forward2_cuda", ([&] {
        forget_rkn_max_cuda_forward2_kernel<scalar_t><<<blocks, threads>>>(
            hiddens.data<scalar_t>(),
            outputs.data<scalar_t>(),
            // output.data<scalar_t>(),
            aux.data<scalar_t>(),
            // mask_hiddens[i].data<uint8_t>(),
            mask_outputs.data<uint8_t>(),
            batch_size,
            state_size,
            kmer_size,
            num_steps,
            compute_la,
            additive,
            has_lintrans);
    }));


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
    bool additive,
    at::Tensor lintrans) {
    // inputs: H x B x dim x kmer_size
    // hidden: B x dim x kmer_size

    bool has_lintrans = lintrans.numel() != 0;
    auto d_inputs = at::zeros_like(inputs);
    auto d_forget = at::zeros_like(forget);
    // auto d_hidden = at::zeros_like(d_output);
    auto new_d_hidden = at::zeros_like(d_hidden);
    auto d_aux = at::zeros_like(d_outputs);
    auto aux = at::zeros_like(d_outputs);
    auto d_lintrans = at::zeros_like(lintrans);

    int64_t num_steps = inputs.size(0);
    // int64_t last_batch_size = batch_sizes[num_steps - 1];
    // int64_t num_steps = input_batch_sizes.size(0);

    const auto kmer_size = d_hidden.size(2);
    const auto state_size = d_hidden.size(1);
    const auto batch_size = d_hidden.size(0);

    const int64_t threads = 512;
    const int64_t block_size = (state_size * kmer_size + threads - 1) / threads;
    const dim3 blocks(block_size, batch_size);

    AT_DISPATCH_FLOATING_TYPES(d_hidden.type(), "forget_rkn_max_backward1_cuda", ([&] {
    forget_rkn_max_cuda_backward1_kernel<scalar_t><<<blocks, threads>>>(
        d_inputs.data<scalar_t>(),
        // d_forget.data<scalar_t>(),
        // d_hidden.data<scalar_t>(),
        // new_d_hidden.data<scalar_t>(),
        d_output.data<scalar_t>(),
        d_outputs.data<scalar_t>(),
        inputs.data<scalar_t>(),
        // hiddens[i].data<scalar_t>(),
        hiddens.data<scalar_t>(),
        // forget.data<scalar_t>(),
        // mask_hiddens[i].data<uint8_t>(),
        mask_outputs.data<uint8_t>(),
        aux.data<scalar_t>(),
        d_aux.data<scalar_t>(),
        batch_size,
        state_size,
        kmer_size,
        num_steps,
        compute_la,
        has_lintrans);
    }));
    // std::cout << aux << std::endl;
    if (has_lintrans) {
        // d_lintrans += at::tensordot(aux, d_aux, {0, 2}, {0, 2}); bmm
        // aux = aux.transpose(1, 2).contiguous().view({num_steps, state_size, -1});
        // std::cout << aux << std::endl;
        // d_lintrans = at::bmm(aux, d_aux.transpose(2, 3).contiguous().view({num_steps, -1, state_size}));
        // d_lintrans = d_lintrans.sum(0);
        d_lintrans = at::tensordot(aux, d_aux, {0, 1, 3}, {0, 1, 3});
        d_aux = at::tensordot(d_aux, lintrans, 2, 0);
        d_aux = d_aux.transpose(2, 3);
        d_aux = d_aux.contiguous();
    }

    for (int64_t i = num_steps - 1; i >= 0; --i) {
        // AT_DISPATCH_FLOATING_TYPES(d_hidden.type(), "forget_rkn_max_backward1_cuda", ([&] {
        // forget_rkn_max_cuda_backward1_kernel<scalar_t><<<blocks, threads>>>(
        //     // d_inputs[i].data<scalar_t>(),
        //     // d_forget.data<scalar_t>(),
        //     d_hidden.data<scalar_t>(),
        //     // new_d_hidden.data<scalar_t>(),
        //     d_output.data<scalar_t>(),
        //     d_outputs[i].data<scalar_t>(),
        //     inputs[i].data<scalar_t>(),
        //     // hiddens[i].data<scalar_t>(),
        //     hiddens[i-1].data<scalar_t>(),
        //     // forget.data<scalar_t>(),
        //     // mask_hiddens[i].data<uint8_t>(),
        //     mask_outputs[i].data<uint8_t>(),
        //     aux.data<scalar_t>(),
        //     d_aux.data<scalar_t>(),
        //     batch_size,
        //     state_size,
        //     kmer_size,
        //     i,
        //     compute_la,
        //     has_lintrans);
        // }));

        // if (has_lintrans) {
        //     d_lintrans += at::tensordot(aux, d_aux, {0, 2}, {0, 2});
        //     d_aux = at::tensordot(d_aux, lintrans, 1, 0);
        //     d_aux = d_aux.transpose(1, 2);
        //     d_aux = d_aux.contiguous();
        // }

        AT_DISPATCH_FLOATING_TYPES(d_hidden.type(), "forget_rkn_max_backward2_cuda", ([&] {
        forget_rkn_max_cuda_backward2_kernel<scalar_t><<<blocks, threads>>>(
            d_inputs[i].data<scalar_t>(),
            d_forget.data<scalar_t>(),
            d_hidden.data<scalar_t>(),
            new_d_hidden.data<scalar_t>(),
            d_output.data<scalar_t>(),
            // d_outputs[i].data<scalar_t>(),
            inputs[i].data<scalar_t>(),
            // hiddens[i].data<scalar_t>(),
            hiddens[i-1].data<scalar_t>(),
            forget.data<scalar_t>(),
            mask_hiddens[i].data<uint8_t>(),
            // mask_outputs[i].data<uint8_t>(),
            d_aux[i].data<scalar_t>(),
            batch_size,
            state_size,
            kmer_size,
            i,
            compute_la,
            additive,
            has_lintrans);
        }));
        d_hidden += new_d_hidden;
    }
    return {d_inputs, d_forget, d_hidden, d_lintrans};
}

std::vector<at::Tensor> forget_rkn_packed_max_cuda_forward(
    at::Tensor inputs,
    at::Tensor input_batch_sizes,
    at::Tensor forget,
    at::Tensor hidden,
    bool compute_la,
    bool additive,
    at::Tensor lintrans) {

    bool has_lintrans = lintrans.numel() != 0;
    auto outputs = at::zeros_like(inputs);
    auto hiddens = at::zeros_like(inputs);
    auto new_hidden = at::zeros_like(hidden);
    auto step_output = at::zeros_like(hidden);
    auto aux = at::zeros_like(inputs);

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

        AT_DISPATCH_FLOATING_TYPES(hidden.type(), "forget_rkn_packed_max_forward1_cuda", ([&] {
            forget_rkn_packed_max_cuda_forward1_kernel<scalar_t><<<blocks, threads>>>(
                // outputs.data<scalar_t>(),
                hiddens.data<scalar_t>(),
                // step_output.data<scalar_t>(),
                outputs.data<scalar_t>(),
                new_hidden.data<scalar_t>(),
                inputs.data<scalar_t>(),
                hidden.data<scalar_t>(),
                forget.data<scalar_t>(),
                // mask_outputs.data<uint8_t>(),
                mask_hiddens.data<uint8_t>(),
                aux.data<scalar_t>(),
                batch_size,
                state_size,
                kmer_size,
                input_offset,
                // i,
                compute_la,
                additive);
        }));
        hidden = new_hidden.clone();
        input_offset += batch_size;

    }

    if (has_lintrans) {
        aux = at::tensordot(aux, lintrans, 1, 0).transpose(1, 2);
        aux = aux.contiguous();
    }

    input_offset = 0;
    for (int64_t i = 0; i < num_steps; ++i) {
        int64_t batch_size = batch_sizes[i];

        const dim3 blocks(block_size, batch_size);
        AT_DISPATCH_FLOATING_TYPES(hidden.type(), "forget_rkn_packed_max_forward2_cuda", ([&] {
            forget_rkn_packed_max_cuda_forward2_kernel<scalar_t><<<blocks, threads>>>(
                outputs.data<scalar_t>(),
                hiddens.data<scalar_t>(),
                step_output.data<scalar_t>(),
                // new_hidden.data<scalar_t>(),
                // inputs.data<scalar_t>(),
                // hidden.data<scalar_t>(),
                forget.data<scalar_t>(),
                mask_outputs.data<uint8_t>(),
                // mask_hiddens.data<uint8_t>(),
                aux.data<scalar_t>(),
                batch_size,
                state_size,
                kmer_size,
                input_offset,
                // i,
                compute_la,
                additive,
                has_lintrans);
        }));
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
    bool additive,
    at::Tensor lintrans) {

    bool has_lintrans = lintrans.numel() != 0;
    auto d_inputs = at::zeros_like(inputs);
    auto d_forget = at::zeros_like(forget);
    auto new_d_hidden = at::zeros_like(d_output);
    auto d_lintrans = at::zeros_like(lintrans);
    auto aux = at::zeros_like(d_outputs);
    auto d_aux = at::zeros_like(d_outputs);

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
        AT_DISPATCH_FLOATING_TYPES(d_hidden.type(), "forget_rkn_packed_max_backward1_cuda", ([&] {
            forget_rkn_packed_max_cuda_backward1_kernel<scalar_t><<<blocks, threads>>>(
                d_inputs.data<scalar_t>(),
                // d_forget.data<scalar_t>(),
                // d_hidden.data<scalar_t>(),
                // new_d_hidden.data<scalar_t>(),
                d_output.data<scalar_t>(),
                d_outputs.data<scalar_t>(),
                inputs.data<scalar_t>(),
                hiddens.data<scalar_t>(),
                // forget.data<scalar_t>(),
                mask_outputs.data<uint8_t>(),
                // mask_hiddens.data<uint8_t>(),
                aux.data<scalar_t>(),
                d_aux.data<scalar_t>(),
                batch_size,
                next_batch_size,
                state_size,
                kmer_size,
                input_offset,
                // i,
                compute_la,
                additive,
                has_lintrans);
        }));
    }

    if (has_lintrans) {
        d_lintrans = at::tensordot(aux, d_aux, {0, 2}, {0, 2});
        d_aux = at::tensordot(d_aux, lintrans, 1, 0);
        d_aux = d_aux.transpose(1, 2);
        d_aux = d_aux.contiguous();
    }

    input_offset = inputs.size(0);
    for (int64_t i = num_steps - 1; i >= 0; --i) {
        int64_t batch_size = batch_sizes[i];
        int64_t next_batch_size = (i > 0) ? batch_sizes[i - 1] : 0;

        const dim3 blocks(block_size, batch_size);
        input_offset -= batch_size;

        // AT_DISPATCH_FLOATING_TYPES(d_hidden.type(), "forget_rkn_packed_max_backward1_cuda", ([&] {
        //     forget_rkn_packed_max_cuda_backward1_kernel<scalar_t><<<blocks, threads>>>(
        //         // d_inputs.data<scalar_t>(),
        //         // d_forget.data<scalar_t>(),
        //         d_hidden.data<scalar_t>(),
        //         // new_d_hidden.data<scalar_t>(),
        //         d_output.data<scalar_t>(),
        //         d_outputs.data<scalar_t>(),
        //         inputs.data<scalar_t>(),
        //         hiddens.data<scalar_t>(),
        //         // forget.data<scalar_t>(),
        //         mask_outputs.data<uint8_t>(),
        //         // mask_hiddens.data<uint8_t>(),
        //         aux.data<scalar_t>(),
        //         d_aux.data<scalar_t>(),
        //         batch_size,
        //         next_batch_size,
        //         state_size,
        //         kmer_size,
        //         input_offset,
        //         // i,
        //         compute_la,
        //         additive,
        //         has_lintrans);
        // }));

        // if (has_lintrans) {
        //     d_lintrans += at::tensordot(d_aux, aux, {0, 2}, {0, 2});
        //     d_aux = at::tensordot(d_aux, lintrans, 1, 0);
        //     d_aux = d_aux.transpose(1, 2);
        // }
        AT_DISPATCH_FLOATING_TYPES(d_hidden.type(), "forget_rkn_packed_max_backward2_cuda", ([&] {
            forget_rkn_packed_max_cuda_backward2_kernel<scalar_t><<<blocks, threads>>>(
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
                d_aux.data<scalar_t>(),
                batch_size,
                next_batch_size,
                state_size,
                kmer_size,
                input_offset,
                i,
                compute_la,
                additive,
                has_lintrans);
        }));
        d_hidden += new_d_hidden;
    }
    return {d_inputs, d_forget, d_hidden, d_lintrans};
}

