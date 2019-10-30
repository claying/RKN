//#include <torch/torch.h>
#include <torch/extension.h>
#include <vector>
#include <stdio.h>
// CUDA forward declarations

std::vector<at::Tensor> forget_rkn_cuda_forward(
    at::Tensor inputs,
    at::Tensor forget,
    at::Tensor hidden,
    bool computa_la,
    bool additive);

std::vector<at::Tensor> forget_rkn_cuda_backward(
    at::Tensor d_outputs,
    at::Tensor d_output,
    at::Tensor d_hidden,
    at::Tensor inputs,
    at::Tensor hiddens,
    at::Tensor forget,
    bool compute_la,
    bool additive);

std::vector<at::Tensor> forget_rkn_packed_cuda_forward(
    at::Tensor inputs,
    at::Tensor batch_sizes,
    at::Tensor forget,
    at::Tensor hidden,
    bool computa_la,
    bool additive);

std::vector<at::Tensor> forget_rkn_packed_cuda_backward(
    at::Tensor d_outputs,
    at::Tensor d_output,
    at::Tensor d_hidden,
    at::Tensor inputs,
    at::Tensor batch_sizes,
    at::Tensor hiddens,
    at::Tensor forget,
    bool compute_la,
    bool additive);

std::vector<at::Tensor> forget_rkn_max_cuda_forward(
    at::Tensor inputs,
    at::Tensor forget,
    at::Tensor hidden,
    bool computa_la,
    bool additive);

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
    bool additive);

std::vector<at::Tensor> forget_rkn_packed_max_cuda_forward(
    at::Tensor inputs,
    at::Tensor batch_sizes,
    at::Tensor forget,
    at::Tensor hidden,
    bool computa_la,
    bool additive);

std::vector<at::Tensor> forget_rkn_packed_max_cuda_backward(
    at::Tensor d_outputs,
    at::Tensor d_output,
    at::Tensor d_hidden,
    at::Tensor inputs,
    at::Tensor batch_sizes,
    at::Tensor hiddens,
    at::Tensor forget,
    at::Tensor mask_outputs,
    at::Tensor mask_hiddens,
    bool compute_la,
    bool additive);

// C++ interface

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<at::Tensor> forget_rkn_forward(
    at::Tensor inputs,
    at::Tensor forget,
    at::Tensor hidden,
    bool compute_la,
    bool additive) {
  CHECK_INPUT(inputs);
  CHECK_INPUT(forget);
  CHECK_INPUT(hidden);
  return forget_rkn_cuda_forward(inputs, forget, hidden, compute_la, additive);
}

std::vector<at::Tensor> forget_rkn_backward(
    at::Tensor d_outputs,
    at::Tensor d_output,
    at::Tensor d_hidden,
    at::Tensor inputs,
    at::Tensor hiddens,
    at::Tensor forget,
    bool compute_la,
    bool additive) {
  CHECK_INPUT(d_outputs);
  CHECK_INPUT(d_output);
  CHECK_INPUT(d_hidden);
  CHECK_INPUT(inputs);
  CHECK_INPUT(hiddens);
  CHECK_INPUT(forget);

  return forget_rkn_cuda_backward(
      d_outputs,
      d_output,
      d_hidden,
      inputs,
      hiddens,
      forget,
      compute_la,
      additive);
}

std::vector<at::Tensor> forget_rkn_packed_forward(
    at::Tensor inputs,
    at::Tensor batch_sizes,
    at::Tensor forget,
    at::Tensor hidden,
    bool compute_la,
    bool additive) {
  CHECK_INPUT(inputs);
  //CHECK_INPUT(batch_sizes);
  CHECK_INPUT(forget);
  CHECK_INPUT(hidden);
  return forget_rkn_packed_cuda_forward(inputs, batch_sizes, forget, hidden, compute_la, additive);
}

std::vector<at::Tensor> forget_rkn_packed_backward(
    at::Tensor d_outputs,
    at::Tensor d_output,
    at::Tensor d_hidden,
    at::Tensor inputs,
    at::Tensor batch_sizes,
    at::Tensor hiddens,
    at::Tensor forget,
    bool compute_la,
    bool additive) {
  CHECK_INPUT(d_outputs);
  CHECK_INPUT(d_output);
  CHECK_INPUT(d_hidden);
  CHECK_INPUT(inputs);
  // CHECK_INPUT(batch_sizes);
  CHECK_INPUT(hiddens);
  CHECK_INPUT(forget);

  return forget_rkn_packed_cuda_backward(
      d_outputs,
      d_output,
      d_hidden,
      inputs,
      batch_sizes,
      hiddens,
      forget,
      compute_la,
      additive);
}

std::vector<at::Tensor> forget_rkn_max_forward(
    at::Tensor inputs,
    at::Tensor forget,
    at::Tensor hidden,
    bool compute_la,
    bool additive) {
  CHECK_INPUT(inputs);
  CHECK_INPUT(forget);
  CHECK_INPUT(hidden);
  return forget_rkn_max_cuda_forward(inputs, forget, hidden, compute_la, additive);
}

std::vector<at::Tensor> forget_rkn_max_backward(
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
  CHECK_INPUT(d_outputs);
  CHECK_INPUT(d_output);
  CHECK_INPUT(d_hidden);
  CHECK_INPUT(inputs);
  CHECK_INPUT(hiddens);
  CHECK_INPUT(forget);
  CHECK_INPUT(mask_outputs);
  CHECK_INPUT(mask_hiddens);

  return forget_rkn_max_cuda_backward(
      d_outputs,
      d_output,
      d_hidden,
      inputs,
      hiddens,
      forget,
      mask_outputs,
      mask_hiddens,
      compute_la,
      additive);
}

std::vector<at::Tensor> forget_rkn_packed_max_forward(
    at::Tensor inputs,
    at::Tensor batch_sizes,
    at::Tensor forget,
    at::Tensor hidden,
    bool compute_la,
    bool additive) {
  CHECK_INPUT(inputs);
  //CHECK_INPUT(batch_sizes);
  CHECK_INPUT(forget);
  CHECK_INPUT(hidden);
  return forget_rkn_packed_max_cuda_forward(inputs, batch_sizes, forget, hidden, compute_la, additive);
}

std::vector<at::Tensor> forget_rkn_packed_max_backward(
    at::Tensor d_outputs,
    at::Tensor d_output,
    at::Tensor d_hidden,
    at::Tensor inputs,
    at::Tensor batch_sizes,
    at::Tensor hiddens,
    at::Tensor forget,
    at::Tensor mask_outputs,
    at::Tensor mask_hiddens,
    bool compute_la,
    bool additive) {
  CHECK_INPUT(d_outputs);
  CHECK_INPUT(d_output);
  CHECK_INPUT(d_hidden);
  CHECK_INPUT(inputs);
  // CHECK_INPUT(batch_sizes);
  CHECK_INPUT(hiddens);
  CHECK_INPUT(forget);
  CHECK_INPUT(mask_outputs);
  CHECK_INPUT(mask_hiddens);

  return forget_rkn_packed_max_cuda_backward(
      d_outputs,
      d_output,
      d_hidden,
      inputs,
      batch_sizes,
      hiddens,
      forget,
      mask_outputs,
      mask_hiddens,
      compute_la,
      additive);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forget_rkn_forward, "forget RKN forward (CUDA)");
  m.def("backward", &forget_rkn_backward, "forget RKN backward (CUDA)");
  m.def("packed_forward", &forget_rkn_packed_forward, "forget RKN packed forward (CUDA)");
  m.def("packed_backward", &forget_rkn_packed_backward, "forget RKN packed backward (CUDA)");
  m.def("max_forward", &forget_rkn_max_forward, "forget RKN forward with max pooling (CUDA)");
  m.def("max_backward", &forget_rkn_max_backward, "forget RKN backward with max pooling (CUDA)");
  m.def("packed_max_forward", &forget_rkn_packed_max_forward, "forget RKN packed forward with max pooling (CUDA)");
  m.def("packed_max_backward", &forget_rkn_packed_max_backward, "forget RKN packed backward  with max pooling(CUDA)");
}

