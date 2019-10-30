import torch
from torch.utils.cpp_extension import load
import os

curr_folder = os.path.dirname(os.path.abspath(__file__))

forget_rkn = load(
    name='forget_rkn',
    sources=["/".join([curr_folder, 'forget_rkn_cuda.cpp']), "/".join([curr_folder, 'forget_rkn_cuda_kernel.cu'])],
    # extra_cuda_cflags=["-arch sm_61"],
    verbose=False)


forget_rkn_max = load(
    name='forget_rkn_max',
    sources=["/".join([curr_folder, 'forget_rkn_max_cuda.cpp']), "/".join([curr_folder, 'forget_rkn_max_cuda_kernel.cu'])],
    # extra_cuda_cflags=["-arch sm_61"],
    verbose=False)


class ForgetRKNPacked(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, batch_sizes, forget, hidden, compute_la=False, additive=False):
        # print(forget_rkn.forward)
        outputs, output, hiddens, hidden = forget_rkn.packed_forward(
            inputs, batch_sizes, forget, hidden, compute_la, additive)
        ctx.save_for_backward(inputs, batch_sizes, hiddens, forget)
        ctx.compute_la = compute_la
        ctx.additive = additive

        return output, outputs, hidden

    @staticmethod
    def backward(ctx, grad_output, grad_outputs, grad_hidden):
        grad_inputs, grad_forget, grad_hidden = forget_rkn.packed_backward(
            grad_outputs.contiguous(), grad_output.contiguous(),
            grad_hidden.contiguous(), *(ctx.saved_variables + (ctx.compute_la, ctx.additive)))
        return grad_inputs, None, grad_forget, grad_hidden, None, None


class ForgetRKN(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, forget, hidden, compute_la=False, additive=False):
        # print(forget_rkn.forward)
        outputs, hiddens = forget_rkn.forward(
            inputs, forget, hidden, compute_la, additive)
        ctx.save_for_backward(inputs, hiddens, forget)
        ctx.compute_la = compute_la
        ctx.additive = additive

        return outputs[-1], outputs, hiddens[-1]

    @staticmethod
    def backward(ctx, grad_output, grad_outputs, grad_hidden):
        grad_inputs, grad_forget, grad_hidden = forget_rkn.backward(
            grad_outputs.contiguous(), grad_output.contiguous(),
            grad_hidden.contiguous(), *(ctx.saved_variables + (ctx.compute_la, ctx.additive)))
        return grad_inputs, grad_forget, grad_hidden, None, None


class ForgetRKNMaxPost(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, forget, hidden, compute_la=False, additive=False, lintrans=None):
        # print(forget_rkn.max_forward)
        lintrans = torch.empty(0) if lintrans is None else lintrans
        outputs, hiddens, mask_outputs, mask_hiddens = forget_rkn_max.max_forward(
            inputs, forget, hidden, compute_la, additive, lintrans)
        ctx.save_for_backward(inputs, hiddens, forget, mask_outputs, mask_hiddens)
        ctx.compute_la = compute_la
        ctx.additive = additive
        ctx.lintrans = lintrans

        return outputs[-1], outputs, hiddens[-1]

    @staticmethod
    def backward(ctx, grad_output, grad_outputs, grad_hidden):
        grad_inputs, grad_forget, grad_hidden, grad_lintrans = forget_rkn_max.max_backward(
            grad_outputs.contiguous(), grad_output.contiguous(),
            grad_hidden.contiguous(), *(ctx.saved_variables + (ctx.compute_la, ctx.additive, ctx.lintrans)))
        if ctx.lintrans.numel() == 0:
            grad_lintrans = None
        return grad_inputs, grad_forget, grad_hidden, None, None, grad_lintrans


class ForgetRKNPackedMaxPost(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, batch_sizes, forget, hidden, compute_la=False, additive=False, lintrans=None):
        # print(forget_rkn.forward)
        lintrans = torch.empty(0) if lintrans is None else lintrans
        outputs, output, hiddens, hidden, mask_outputs, mask_hiddens = forget_rkn_max.packed_max_forward(
            inputs, batch_sizes, forget, hidden, compute_la, additive, lintrans)
        ctx.save_for_backward(inputs, batch_sizes, hiddens, forget, mask_outputs, mask_hiddens)
        ctx.compute_la = compute_la
        ctx.additive = additive
        ctx.lintrans = lintrans
        return output, outputs, hidden

    @staticmethod
    def backward(ctx, grad_output, grad_outputs, grad_hidden):
        grad_inputs, grad_forget, grad_hidden, grad_lintrans = forget_rkn_max.packed_max_backward(
            grad_outputs.contiguous(), grad_output.contiguous(),
            grad_hidden.contiguous(), *(ctx.saved_variables + (ctx.compute_la, ctx.additive, ctx.lintrans)))
        if ctx.lintrans.numel() == 0:
            grad_lintrans = None
        return grad_inputs, None, grad_forget, grad_hidden, None, None, grad_lintrans


def rkn_forward(input, forget, hx, compute_la=False, additive=False):
    """
    input: H x batch_size x hidden_size x kmer_size
    legnth: batch_size
    hx: batch_size x hidden_size x kmer_size

    hidden: batch_size x hidden_size x kmer_size
    output: H x batch_size x hidden_size x kmer_size
    """
    max_length = input.size(0)
    ones = input.new_ones(hx.size(0), hx.size(1), 1,
                          requires_grad=False)
    output = []
    if compute_la:
        hh = torch.zeros_like(hx, requires_grad=False)
    for i in range(max_length):
        hx_low = torch.cat([ones, hx[:, :, :-1]], dim=-1)
        hx = forget * hx + (1. - forget) * hx_low * input[i]
        if compute_la:
            hh = hh + hx_low * input[i]
            output.append(hh / (i + 1.))
        else:
            output.append(hx)

    output = torch.stack(output, 0)
    return output[-1], output, hx


def rkn_packed(input, batch_sizes, forget, hx, compute_la=False, additive=False):
    """
    input: PackedSequence: all_length x hidden_size x kmer_size
    hx: batch_size x hidden_size x kmer_size

    hidden: batch_size x hidden_size
    output: all_length x batch_size x hidden_size
    """
    outputs = []
    input_offset = 0
    last_batch_size = batch_sizes[0]
    hiddens = []
    output = []
    ones = hx.new_ones(hx.size(0), hx.size(1), 1, requires_grad=False)
    if compute_la:
        hh = torch.zeros_like(hx, requires_grad=False)
    for i, batch_size in enumerate(batch_sizes):
        step_input = input[input_offset:input_offset+batch_size]
        input_offset += batch_size

        dec = last_batch_size - batch_size
        if dec > 0:
            hiddens.append(hx[-dec:])
            if compute_la:
                output.append(hh[-dec:] / i)
            else:
                output.append(hx[-dec:])

            hx = hx[:-dec]
            ones = ones[:-dec]
            if compute_la:
                hh = hh[:-dec]
        last_batch_size = batch_size

        hx_low = torch.cat([ones, hx[:, :, :-1]], dim=-1)
        if additive:
            hx = forget * hx + (1. - forget) * (hx_low + step_input)
        else:
            hx = forget * hx + (1. - forget) * hx_low * step_input
        # print(hx_low)

        if compute_la:
            if additive:
                hh = hh + hx_low + step_input
            else:
                hh = hh + hx_low * step_input
            outputs.append(hh / (i + 1.))
        else:
            outputs.append(hx)
    if compute_la:
        output.append(hh / (i + 1.))
    else:
        output.append(hx)
    hiddens.append(hx)
    hiddens.reverse()
    output.reverse()

    hidden = torch.cat(hiddens, 0)
    outputs = torch.cat(outputs, 0)
    output = torch.cat(output, 0)
    return output, outputs, hidden


def rkn_forward_max(input, forget, hx, compute_la=False, additive=False):
    """
    input: H x batch_size x hidden_size x kmer_size
    legnth: batch_size
    hx: batch_size x hidden_size x kmer_size

    hidden: batch_size x hidden_size x kmer_size
    output: H x batch_size x hidden_size x kmer_size
    """
    max_length = input.size(0)
    ones = input.new_ones(hx.size(0), hx.size(1), 1,
                          requires_grad=False)
    output = []
    if compute_la:
        hh = torch.zeros_like(hx, requires_grad=False)
    for i in range(max_length):
        hx_low = torch.cat([ones, hx[:, :, :-1]], dim=-1)
        hx = torch.max(forget * hx, hx_low * input[i])

        if compute_la:
            hh = torch.max(hh, hx_low * input[i])
            output.append(hh)
        else:
            output.append(hx)

    output = torch.stack(output, 0)
    return output[-1], output, hx


def rkn_packed_max(input, batch_sizes, forget, hx, compute_la=False, additive=False):
    """
    input: PackedSequence: all_length x hidden_size x kmer_size
    hx: batch_size x hidden_size x kmer_size

    hidden: batch_size x hidden_size
    output: all_length x batch_size x hidden_size
    """
    outputs = []
    input_offset = 0
    last_batch_size = batch_sizes[0]
    hiddens = []
    output = []
    ones = hx.new_ones(hx.size(0), hx.size(1), 1, requires_grad=False)
    if compute_la:
        hh = torch.zeros_like(hx, requires_grad=False)
    for i, batch_size in enumerate(batch_sizes):
        step_input = input[input_offset:input_offset+batch_size]
        input_offset += batch_size

        dec = last_batch_size - batch_size
        if dec > 0:
            hiddens.append(hx[-dec:])
            if compute_la:
                output.append(hh[-dec:])
            else:
                output.append(hx[-dec:])

            hx = hx[:-dec]
            ones = ones[:-dec]
            if compute_la:
                hh = hh[:-dec]
        last_batch_size = batch_size

        hx_low = torch.cat([ones, hx[:, :, :-1]], dim=-1)
        if additive:
            hx = torch.max(forget * hx, hx_low + step_input)
        else:
            hx = torch.max(forget * hx, hx_low * step_input)
        # print(hx_low)

        if compute_la:
            if additive:
                hh = torch.max(hh, hx_low + step_input)
            else:
                hh = torch.max(hh, hx_low * step_input)
            outputs.append(hh)
        else:
            outputs.append(hx)
    if compute_la:
        output.append(hh)
    else:
        output.append(hx)
    hiddens.append(hx)
    hiddens.reverse()
    output.reverse()

    hidden = torch.cat(hiddens, 0)
    outputs = torch.cat(outputs, 0)
    output = torch.cat(output, 0)
    return output, outputs, hidden


def rkn_forward_max_lintrans(input, forget, hx, compute_la=False, additive=False, lintrans=None):
    """
    input: H x batch_size x hidden_size x kmer_size
    legnth: batch_size
    hx: batch_size x hidden_size x kmer_size

    hidden: batch_size x hidden_size x kmer_size
    output: H x batch_size x hidden_size x kmer_size
    """
    max_length = input.size(0)
    ones = input.new_ones(hx.size(0), hx.size(1), 1,
                          requires_grad=False)
    output = []
    if compute_la:
        hh = torch.zeros_like(hx, requires_grad=False)
    for i in range(max_length):
        hx_low = torch.cat([ones, hx[:, :, :-1]], dim=-1)
        aux = hx_low * input[i]
        if not compute_la:
            hh = forget * hx
            hx = torch.max(hh, aux)
        else:
            hx = torch.max(forget * hx, aux)

        if lintrans is not None:
            aux = torch.tensordot(aux, lintrans, dims=[[1], [0]])
            aux = aux.transpose(1, 2)

        if compute_la or lintrans is not None:
            hh = torch.max(hh, aux)
            output.append(hh)
        else:
            output.append(hx)

    output = torch.stack(output, 0)
    return output[-1], output, hx


def rkn_packed_max_lintrans(input, batch_sizes, forget, hx, compute_la=False, additive=False, lintrans=None):
    """
    input: PackedSequence: all_length x hidden_size x kmer_size
    hx: batch_size x hidden_size x kmer_size

    hidden: batch_size x hidden_size
    output: all_length x batch_size x hidden_size
    """
    outputs = []
    input_offset = 0
    last_batch_size = batch_sizes[0]
    hiddens = []
    output = []
    ones = hx.new_ones(hx.size(0), hx.size(1), 1, requires_grad=False)
    if compute_la:
        hh = torch.zeros_like(hx, requires_grad=False)
    for i, batch_size in enumerate(batch_sizes):
        step_input = input[input_offset:input_offset+batch_size]
        input_offset += batch_size

        dec = last_batch_size - batch_size
        if dec > 0:
            hiddens.append(hx[-dec:])
            if compute_la or lintrans is not None:
                output.append(hh[-dec:])
            else:
                output.append(hx[-dec:])

            hx = hx[:-dec]
            ones = ones[:-dec]
            if compute_la:
                hh = hh[:-dec]
        last_batch_size = batch_size

        hx_low = torch.cat([ones, hx[:, :, :-1]], dim=-1)
        if additive:
            aux = hx_low + step_input
        else:
            aux = hx_low * step_input

        if compute_la:
            hx = torch.max(forget * hx, aux)
        else:
            hh = forget * hx
            hx = torch.max(hh, aux)
        
        if lintrans is not None:
            aux = torch.tensordot(aux, lintrans, dims=[[1], [0]])
            aux = aux.transpose(1, 2)

        if compute_la or lintrans is not None:
            hh = torch.max(hh, aux)
            outputs.append(hh)
        else:
            outputs.append(hx)

    if compute_la or lintrans is not None:
        output.append(hh)
    else:
        output.append(hx)
    hiddens.append(hx)
    hiddens.reverse()
    output.reverse()

    hidden = torch.cat(hiddens, 0)
    outputs = torch.cat(outputs, 0)
    output = torch.cat(output, 0)
    return output, outputs, hidden
