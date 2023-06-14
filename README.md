# CaloClouds Model for Fast Calorimeter Simulation

PyTorch implementation of the CaloClouds Model introduced in *CaloClouds: Fast Geometry-Independent Highly-Granular Calorimeter Simulation* ([arXiv:2305.04847](https://arxiv.org/abs/2305.04847)).

---

The CaloClouds Model generates photon showers in point cloud format with up to 6000 points per shower for an energy range between 10 and 90 GeV. It consists out of multiple sub-generative models, including the PointWise Net trained as a denoising diffusion probabilistic model (DDPM), and two normalizing flows (the Latent Flow and the Shower Flow). The training data is generated with a Geant4 simulation of the planned electromagnetic calorimeter of the International Large Detector (ILD).

---

The CaloClouds Pointwise Net Diffusion Model can be trained simulatinously with the EPiC Encoder and the Latent Flow via `python main.py` using the default parameters set in [`config.py`](./configs.py).

The Shower Flow is trained via the notebook [`ShowerFlow_Training.ipynb`](./ShowerFlow_Training.ipynb).

The polynomial fits for the occupancy calculations are performed in [`occupancy_scale.ipynb`](./occupancy_scale.ipynb).

An outline of the sampling process is shown in [`validation.ipynb`](./validation.ipynb).

---

The training dataset is currently not public. If you're interested in receiveing access, please contact the authors. 

---

The code for training the diffusion probabilistic model is based on: https://github.com/luost26/diffusion-point-cloud
