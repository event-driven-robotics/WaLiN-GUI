from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn


class SurrGradSpike(torch.autograd.Function):
    """
    Here we implement our spiking nonlinearity which also implements
    the surrogate gradient. By subclassing torch.autograd.Function,
    we will be able to use all of PyTorch's autograd functionality.
    Here we use the normalized negative part of a fast sigmoid
    as this was done in Zenke & Ganguli (2018).
    """

    scale = 100

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we compute a step function of the input Tensor
        and return it. ctx is a context object that we use to stash information which
        we need to later backpropagate our error signals. To achieve this we use the
        ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor we need to compute the
        surrogate gradient of the loss with respect to the input.
        Here we use the normalized negative part of a fast sigmoid
        as this was done in Zenke & Ganguli (2018).
        """
        (input,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input / (SurrGradSpike.scale * torch.abs(input) + 1.0) ** 2
        return grad


activation = SurrGradSpike.apply


# MN neuron
class MN_neuron(nn.Module):
    NeuronState = namedtuple("NeuronState", ["V", "i1", "i2", "Thr", "spk"])

    def __init__(
        self,
        nb_inputs,
        parameters_combination,
        dt=1 / 1000,
        a=5,
        A1=10,
        A2=-0.6,
        b=10,
        G=50,
        k1=200,
        k2=20,
        R1=0,
        R2=1,
        train=False,
    ):  # default combination: M2O of the original paper
        super(MN_neuron, self).__init__()

        # One-to-one synapse
        self.linear = nn.Parameter(torch.ones(
            1, nb_inputs), requires_grad=train)
        self.N = nb_inputs
        one2N_matrix = torch.ones(1, nb_inputs)
        # define some constants
        self.C = 1
        self.EL = -0.07
        self.Vr = -0.07
        self.Tr = -0.06
        self.Tinf = -0.05

        # define parameters
        self.a = a
        self.A1 = A1
        self.A2 = A2
        self.b = b  # units of 1/s
        self.G = G * self.C  # units of 1/s
        self.k1 = k1  # units of 1/s
        self.k2 = k2  # units of 1/s
        self.R1 = R1
        self.R2 = R2
        self.dt = dt  # get dt from sample rate!

        # update parameters from GUI
        for param_name in list(parameters_combination.keys()):
            eval_string = (
                "self.{}".format(param_name) + " = " +
                str(parameters_combination[param_name])
            )
            exec(eval_string)

        # set up missing parameters
        self.a = nn.Parameter(one2N_matrix * self.a, requires_grad=train)
        self.A1 = nn.Parameter(one2N_matrix * self.A1 *
                               self.C, requires_grad=train)
        self.A2 = nn.Parameter(one2N_matrix * self.A2 *
                               self.C, requires_grad=train)

        self.state = None

    def forward(self, x):
        if self.state is None:
            self.state = self.NeuronState(
                V=torch.ones(x.shape[0], self.N, device=x.device) * self.EL,
                i1=torch.zeros(x.shape[0], self.N, device=x.device),
                i2=torch.zeros(x.shape[0], self.N, device=x.device),
                Thr=torch.ones(x.shape[0], self.N,
                               device=x.device) * self.Tinf,
                spk=torch.zeros(x.shape[0], self.N, device=x.device),
            )

        V = self.state.V
        i1 = self.state.i1
        i2 = self.state.i2
        Thr = self.state.Thr

        i1 += -self.k1 * i1 * self.dt
        i2 += -self.k2 * i2 * self.dt
        V += self.dt * (self.linear * x + i1 + i2 -
                        self.G * (V - self.EL)) / self.C
        Thr += self.dt * (self.a * (V - self.EL) - self.b * (Thr - self.Tinf))

        spk = activation(V - Thr)

        i1 = (1 - spk) * i1 + (spk) * (self.R1 * i1 + self.A1)
        i2 = (1 - spk) * i2 + (spk) * (self.R2 * i2 + self.A2)
        Thr = (1 - spk) * Thr + (spk) * torch.max(Thr, torch.tensor(self.Tr))
        V = (1 - spk) * V + (spk) * self.Vr

        self.state = self.NeuronState(V=V, i1=i1, i2=i2, Thr=Thr, spk=spk)

        return spk

    def reset(self):
        self.state = None


class IZ_neuron(nn.Module):
    # u = membrane recovery variable
    NeuronState = namedtuple('NeuronState', ['V', 'u', 'spk'])

    def __init__(
        self,
        nb_inputs,
        parameters_combination,
        dt=1/1000,
        a=0.02,
        b=0.2,
        c=-0.065,
        d=8,
        train=False
    ):
        super(IZ_neuron, self).__init__()

        # One-to-one synapse
        self.linear = nn.Parameter(torch.ones(
            1, nb_inputs), requires_grad=train)
        self.N = nb_inputs
        # define some constants
        self.spike_value = 35  # spike threshold

        # define parameters
        self.a = a
        self.b = b
        self.c = c  # reset potential
        self.d = d
        self.dt = dt  # get dt from sample rate!

        # update parameters from GUI
        for param_name in list(parameters_combination.keys()):
            eval_string = (
                "self.{}".format(param_name) + " = " +
                str(parameters_combination[param_name])
            )
            exec(eval_string)

        self.state = None

    def forward(self, x):
        if self.state is None:
            self.state = self.NeuronState(V=torch.ones(x.shape[0], self.N, device=x.device) * -0.070,
                                          u=torch.zeros(
                                              x.shape[0], self.N, device=x.device) * -0.014,
                                          spk=torch.zeros(x.shape[0], self.N, device=x.device))

        V = self.state.V
        u = self.state.u

        dV = ((0.04 * V + 5) * V) + 140 - u + x
        V = V + (dV + u) * self.dt

        du = self.a * (self.b * V - u)
        u = u + self.dt * du

        # create spike when threshold reached
        spk = activation(V - self.spike_value)

        # (reset membrane voltage) or (only update)
        V = (spk * self.c) + ((1 - spk) * V)
        # (reset recovery) or (update currents)
        u = (spk * (u + self.d)) + ((1 - spk) * u)

        self.state = self.NeuronState(V=V, u=u, spk=spk)

        return spk

    def reset(self):
        self.state = None


class LIF_neuron(nn.Module):
    NeuronState = namedtuple("NeuronState", ["V", "spk"])

    def __init__(
            self,
            nb_inputs,
            parameters_combination,
            dt=1 / 1000,
            beta=1.0,
            thr=0.1,
            R=1.0,
            train=False,
    ):
        super(LIF_neuron, self).__init__()

        self.nb_inputs = nb_inputs
        self.beta = beta
        self.threshold = thr
        # self.V_rest = -0.04  # -40mV
        self.R = R
        self.dt = dt

        # update parameters from GUI
        for param_name in list(parameters_combination.keys()):
            eval_string = (
                "self.{}".format(param_name) + " = " +
                str(parameters_combination[param_name])
            )
            exec(eval_string)

        self.state = None

    def forward(self, x):
        if self.state is None:
            self.state = self.NeuronState(
                V=torch.zeros(x.shape[0], self.nb_inputs,
                              device=x.device),
                spk=torch.zeros(x.shape[0], self.nb_inputs, device=x.device),
            )
        V = self.state.V
        spk = self.state.spk

        V = (self.beta * V + (1.0-self.beta) * x * self.R) * (1.0 - spk)
        spk = activation(V-1.0)

        self.state = self.NeuronState(V=V, spk=spk)

        return spk

    def reset(self):
        self.state = None

class RLIF_neuron(nn.Module):
    NeuronState = namedtuple("NeuronState", ["V", "syn", "spk"])

    def __init__(
            self,
            nb_inputs,
            parameters_combination,
            dt=1 / 1000,
            alpha=1.0,
            beta=1.0,
            thr=0.1,
            R=1.0,
            train=False,
    ):
        super(RLIF_neuron, self).__init__()

        self.nb_inputs = nb_inputs
        self.alpha = alpha
        self.beta = beta
        self.threshold = thr
        # self.V_rest = -0.04  # -40mV
        self.R = R
        self.dt = dt

        # update parameters from GUI
        for param_name in list(parameters_combination.keys()):
            eval_string = (
                "self.{}".format(param_name) + " = " +
                str(parameters_combination[param_name])
            )
            exec(eval_string)

        self.state = None

    def forward(self, x):
        if self.state is None:
            self.state = self.NeuronState(
                V=torch.zeros(x.shape[0], self.nb_inputs,
                              device=x.device),
                syn=torch.zeros(x.shape[0], self.nb_inputs,
                              device=x.device),
                spk=torch.zeros(x.shape[0], self.nb_inputs, device=x.device),
            )
        V = self.state.V
        spk = self.state.spk
        syn = self.state.syn

        syn = self.alpha*syn + spk
        V = (self.beta * V + (1.0-self.beta) * x * self.R + (1.0-self.beta)*syn) * (1.0 - spk)
        spk = activation(V-1.0)

        self.state = self.NeuronState(V=V, syn=syn, spk=spk)

        return spk

    def reset(self):
        self.state = None