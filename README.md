# [JAX](https://github.com/google/jax) Implementation of [NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](http://www.matthewtancik.com/nerf)

> [Original TF NeRF Implementation](https://github.com/bmild/nerf) | [Original Google Research JAX Implementation](https://github.com/google-research/google-research/tree/master/jaxnerf)

# Setup 🛠

Clone the Repository

```bash
git clone https://github.com/SauravMaheshkar/NeRF.git
```

## Docker 🐳

I have included a Docker Image with this Repository that already contains the datasets preinstalled and the codebase from this repository. This avoids downloading the original dataset from Google Drive (which might be hard if you're working on a virtual machine)

> **NOTE:** The dataset contained in the provided Docker image is taken from the [original source](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1)

The codebase contains W&B implementation for logging your metrics to a project dashboard. In order for easy logging I'd recommend creating a `.env` file in the root and adding your WANDB_API_KEY as follows:-

```
WANDB_API_KEY=<YOUR API KEY FROM wandb.ai/settings>
```

Then run the following commands to run a training job (for example on the fern scene from llff)

```bash
docker pull ghcr.io/sauravmaheshkar/nerf:latest
sudo docker run --gpus all \
  -v $PWD:/tmp -w /tmp \
  -it --env-file .env \
  nerf:latest python3.9 train.py \
  --data_dir=nerf_llff_data/fern \
  --train_dir=logs/fern \
  --config=configs/llff
```

## Local

The following codebase can be used with Python>=3.8 and is tested on 3.8 and 3.9

```bash
python -m pip install --upgrade pip setuptools wheel
# Make Sure you have Cuda >= 11.1 and cudnn >= 8.2
python -m pip install jax[cuda11_cudnn82] -f https://storage.googleapis.com/jax-releases/jax_releases.html
# Other requirements
python -m pip install --no-cache-dir -r requirements.txt
```

# Citations

```
@software{jaxnerf2020github,
  author = {Boyang Deng and Jonathan T. Barron and Pratul P. Srinivasan},
  title = {{JaxNeRF}: an efficient {JAX} implementation of {NeRF}},
  url = {https://github.com/google-research/google-research/tree/master/jaxnerf},
  version = {0.0},
  year = {2020},
}
```
```
@inproceedings{mildenhall2020nerf,
  title={NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis},
  author={Ben Mildenhall and Pratul P. Srinivasan and Matthew Tancik and Jonathan T. Barron and Ravi Ramamoorthi and Ren Ng},
  year={2020},
  booktitle={ECCV},
}
```
