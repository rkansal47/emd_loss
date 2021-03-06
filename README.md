# Energy mover's distance as a differentiable loss

Testing the energy mover's distance [1] as a loss function for particle cloud autoencoders, derived from the DeepEMD [2] [implementation](https://github.com/icoz69/DeepEMD).

Instructions:

1)  `pip install qpth cvxpy` (Note: for the `cvxpy` library you may need to first install the `g++` compiler: on Ubuntu `sudo apt-get install g++`)
2)  In the training script import the `emd_loss` function: `from emd_loss import emd_loss`
3)  Use it as one would e.g. MSE: `loss = emd_loss(jets_true, jets_output)`

Note: because of a bug in PyTorch (https://github.com/pytorch/pytorch/issues/36921, ~~to be solved in PyTorch 1.9~~) **this will crash on cuda for jets with >32 particles.** 
 - UPDATE 5/7/21: PyTorch 1.9 has been released but this is not yet fixed :(. 
 - UPDATE 21/7/21: I have submitted a PR which should solve this https://github.com/pytorch/pytorch/pull/61815 



<br/><br/>


[1] P. T. Komiske, E. M. Metodiev, and J. Thaler, “Metric Space of Collider Events”, Phys. Rev. Lett. 123 (2019), no. 4, 041801, doi:10.1103/PhysRevLett.123.041801, arXiv:1902.02346.

[2] C. Zhang, Y. Cai, G. Lin and C. Shen, "DeepEMD: Few-Shot Image Classification With Differentiable Earth Mover’s Distance and Structured Classifiers," 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020, pp. 12200-12210, doi: 10.1109/CVPR42600.2020.01222.
