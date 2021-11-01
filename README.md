# RealVS
Reproducing of the paper entitled "RealVS: Towards Enhancing the Precision of Top Hits in Ligand-Based Virtual Screening of Drug Leads from Large Compound Databases"

- All rights reserved by Yueming Yin, Email: 1018010514@njupt.edu.cn.
- RealVS has been deployed on our web page: www.noveldelta.com/RealVS.

The [Code Ocean](https://codeocean.com) compute capsule will allow you to reproduce the results published by the author on your local machine<sup>1</sup>. Follow the instructions below, or consult [the knowledge base](https://help.codeocean.com/user-manual/sharing-and-finding-published-capsules/exporting-capsules-and-reproducing-results-on-your-local-machine) for more information. Don't hesitate to reach out via live chat or [email](mailto:support@codeocean.com) if you have any questions.

<sup>1</sup> You may need access to additional hardware and/or software licenses.

# Prerequisites

- [Docker Community Edition (CE)](https://www.docker.com/community-edition)
- [nvidia-docker](https://github.com/NVIDIA/nvidia-docker/) for code that leverages the GPU
- Licenses where applicable

# Instructions

## The computational environment (Docker image)

This capsule is private and its environment cannot be downloaded at this time. You will need to rebuild the environment locally.

> If there's any software requiring a license that needs to be run during the build stage, you'll need to make your license available. See [the knowledge base](https://help.codeocean.com/user-manual/sharing-and-finding-published-capsules/exporting-capsules-and-reproducing-results-on-your-local-machine) for more information.

In your terminal, navigate to the folder where you've extracted the capsule and execute the following command:
```shell
cd environment && docker build . --tag RealVS; cd ..
```

> This step will recreate the environment (i.e., the Docker image) locally, fetching and installing any required dependencies in the process. If any external resources have become unavailable for any reason, the environment will fail to build.

## Running the capsule to reproduce the results

In your terminal, navigate to the folder where you've extracted the capsule and execute the following command, adjusting parameters as needed:
```shell
nvidia-docker run --it \
  --workdir /RealVS \
  --volume "$PWD/data":/Benchmark_Datasets \
  --volume "$PWD/code":/RealVS \
  RealVS
```

# Reproduction
## Reproducing the training process of RealVS
In your jupyter notebook, set the task ID to reproduce the training process of RealVS using the data in "Benchmark_Datasets": 
```
./RealVS/Train_RealVS.ipynb
```
## Reproducing the test and visualization process of RealVS
In your jupyter notebook, set the task ID to reproduce the training process of RealVS using the data in "Benchmark_Datasets" and the final models in "RealVS_Models":
```
./RealVS/Test&Viz_RealVS.ipynb
```
# Citation
## Latex
```
@article{yin2021realvs,
author = {Yin, Yueming and Hu, Haifeng and Yang, Zhen and Xu, Huajian and Wu, Jiansheng},
title = {RealVS: Toward Enhancing the Precision of Top Hits in Ligand-Based Virtual Screening of Drug Leads from Large Compound Databases},
journal = {Journal of Chemical Information and Modeling},
volume = {61},
number = {10},
pages = {4924-4939},
year = {2021},
}
```
## Word
```
Yueming Yin, Haifeng Hu, Zhen Yang, Huajian Xu, and Jiansheng Wu. Realvs: To- ward enhancing the precision of top hits in ligand-based virtual screening of drug leads from large compound databases. Journal of Chemical Information and Modeling, 61(10):4924–4939, 2021.
```
