import torch
from torch import nn
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import numpy as np


class EMDLoss(nn.Module):
    """
    Calculates the energy mover's distance between two batches of jets differentiably as a convex optimization problem via the cvxpylayers package
    """

    def __init__(self, n_parts):
        super(EMDLoss, self).__init__()
        self.n_parts = n_parts

        x = cp.Variable(n_parts * n_parts)  # flows
        c = cp.Parameter(n_parts * n_parts)  # costs
        w = cp.Parameter(n_parts + n_parts)  # weights
        Emin = cp.Parameter(1)  # min energy out of the two jets

        g1 = np.zeros((n_parts, n_parts * n_parts))
        for i in range(n_parts):
            g1[i, i * n_parts : (i + 1) * n_parts] = 1
        g2 = np.concatenate([np.eye(n_parts) for i in range(n_parts)], axis=1)
        g = np.concatenate((g1, g2), axis=0)

        constraints = [x >= 0, g @ x <= w, cp.sum(x) == Emin]
        objective = cp.Minimize(c.T @ x)
        problem = cp.Problem(objective, constraints)

        cvxpylayer = CvxpyLayer(problem, parameters=[c, w, Emin], variables=[x])


    def forward(self, jets1, jets2, return_flow=False):
        """
        :param jets1: [nbatch * ] num_particles * 3
            - 3 particle features [eta, phi, pt]
        :param jets2: [nbatch * ] num_particles * 3
        :param return_flow: _bool_
            - return the flow as well as the EMD score
        :return:
        emd distance: nbatch * 1
        flow : (if return_flow) nbatch * num_particles * num_particles
        """
        assert jets1.shape[1] == n_parts
        assert jets2.shape[1] == n_parts

        diffs = -(jets1[:, :, :2].unsqueeze(2) - jets2[:, :, :2].unsqueeze(1)) + 1e-12
        dists = torch.norm(diffs, dim=3).view(-1, n_parts * n_parts)

        weights = torch.cat((jets1[:, :, 2], jets2[:, :, 2]), dim=1)

        E1 = torch.sum(jets1[:, :, 2], dim=1)
        E2 = torch.sum(jets2[:, :, 2], dim=1)

        Emin = torch.minimum(E1, E2).unsqueeze(1)
        EabsDiff = torch.abs(E2 - E1).unsqueeze(1)

        flows, = cvxpylayer(dists, weights, Emin)

        emds = torch.sum(dists * flows, dim=1) + EabsDiff
        return (emds, flow) if return_flow else emds
