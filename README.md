# Energy mover's distance as a differentiable loss

Testing the energy mover's distance [1] as a loss function for particle cloud autoencoders, derived from the DeepEMD [2] [implementation](https://github.com/icoz69/DeepEMD).

Instructions:

1)  `pip install qpth cvxpy` (Note: for the `cvxpy` library you may need to first install the `g++` compiler: on Ubuntu `sudo apt-get install g++`)
2)  In the training script import the `emd_loss` function: `from emd_loss import emd_loss`
3)  Use it as one would e.g. MSE: `loss = emd_loss(jets_true, jets_output)`


<br/><br/>


[1] P. T. Komiske, E. M. Metodiev, and J. Thaler, “Metric Space of Collider Events”, Phys. Rev. Lett. 123 (2019), no. 4, 041801, doi:10.1103/PhysRevLett.123.041801, arXiv:1902.02346.

[2] C. Zhang, Y. Cai, G. Lin and C. Shen, "DeepEMD: Few-Shot Image Classification With Differentiable Earth Mover’s Distance and Structured Classifiers," 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020, pp. 12200-12210, doi: 10.1109/CVPR42600.2020.01222.
