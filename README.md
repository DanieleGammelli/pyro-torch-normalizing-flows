# Density Estimation with Normalizing Flows 
 
### Implementations of [RealNVP](https://arxiv.org/pdf/1605.08803.pdf) in Pytorch/Pyro.

![alt text](assets/nf_gif.gif)

In this repository we implement Normalizing Flows for both *Unconditional Density Estimation* (i.e., ![formula](https://render.githubusercontent.com/render/math?math=p(\mathbf{x}))) and *Conditional Density Estimation* (i.e., ![formula](https://render.githubusercontent.com/render/math?math=p(\mathbf{x}|\mathbf{h}))). The repository is organized as follows:

- *models*: contains .py scripts of Unconditional and Conditional AffineCoupling Flows respectively
- *(Un)conditionalNF.ipynb*: basic usage example on 2-d density estimation
- *cnf_torch_save_run*: trained model for the Conditional Normalizing Flow notebook for faster replication
