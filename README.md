# Diffusion Deepdive
### Exploration-based project aimed at understanding how diffusion models work by implementing various diffusion models.


## Approach:
  1. Single-Image Generation (to verify that diffusion does work)
  2. Single-Class Generation ([Landscape data](https://www.kaggle.com/datasets/arnaud58/landscape-pictures) from Kaggle)
  3. Multi-Class Conditional Generation (CIFAR 10)
  4. Latent Diffusion (CIFAR 10)

## Single-Image Generation
All Training done on 1 x RTX 2080

### Oski (32 x 32)
<p align="center">
  <img src="https://github.com/henryhmko/project_cerulean/blob/main/single_class/result_imgs/oski_result.png" width="940"/>
</p>

### VLSB Santa Dino (64 x 64)
<p align="center">
  <img src="https://github.com/henryhmko/project_cerulean/blob/main/single_class/result_imgs/vlsb_santa_dino_result.png" width="940"/>
</p>

## Single-Class Generation
Introduction: Extend the Single-Image Generation into Single-Class Generation. Test with the landscape data first.

### Landscape Data
#### **1. Training Run #1**
```python
epochs = 500
lr = 1e-3
bs = 16 #gpu is 2080...
diffusion_timesteps = 1000
```
> Training Time: ~9 hours

**Summary Result**
<p align="center">
  <img src="https://github.com/henryhmko/project_cerulean/blob/main/single_class/result_imgs/Landscape_Run1_Summary.png" width="940"/>
</p>

**Detailed Sampling**
<p align="center">
  <img src="https://github.com/henryhmko/project_cerulean/blob/main/single_class/result_imgs/Landscape_Run1_Detailed_1.png" width="940"/>
</p>
<p align="center">
  <img src="https://github.com/henryhmko/project_cerulean/blob/main/single_class/result_imgs/Landscape_Run1_Detailed_2.png" width="940"/>
</p>
<p align="center">
  <img src="https://github.com/henryhmko/project_cerulean/blob/main/single_class/result_imgs/Landscape_Run1_Detailed_3.png" width="940"/>
</p>
<p align="center">
  <img src="https://github.com/henryhmko/project_cerulean/blob/main/single_class/result_imgs/Landscape_Run1_Detailed_4.png" width="940"/>
</p>
<p align="center">
  <img src="https://github.com/henryhmko/project_cerulean/blob/main/single_class/result_imgs/Landscape_Run1_Detailed_5.png" width="940"/>
</p>

### Question:
One question came up when I was looking at the outputs:
- *Why does the model seem to generate complicated parts of a scene(e.g. mountains, grass) first?*
  * For example, the model seemed to start generating the mountains rather than the sky or the grass before the sky. 
  * Is something with texture easier for a model to start generating from **(i.e. easiest way to optimize for lower loss is constructing prominent features first)**?

#### **2. Training Run #2**
```python
epochs = 500
lr = 1e-4
bs = 16
diffusion_timesteps = 1000
```
> Training Time: ~9 hours

**Summary Result**
<p align="center">
  <img src="https://github.com/henryhmko/project_cerulean/blob/main/single_class/result_imgs/Landscape_Run2_Summary.png" width="940"/>
</p>

**Detailed Sampling**
<p align="center">
  <img src="https://github.com/henryhmko/project_cerulean/blob/main/single_class/result_imgs/Landscape_Run2_Detailed_1.png" width="940"/>
</p>
<p align="center">
  <img src="https://github.com/henryhmko/project_cerulean/blob/main/single_class/result_imgs/Landscape_Run2_Detailed_2.png" width="940"/>
</p>
<p align="center">
  <img src="https://github.com/henryhmko/project_cerulean/blob/main/single_class/result_imgs/Landscape_Run2_Detailed_3.png" width="940"/>
</p>
<p align="center">
  <img src="https://github.com/henryhmko/project_cerulean/blob/main/single_class/result_imgs/Landscape_Run2_Detailed_4.png" width="940"/>
</p>
<p align="center">
  <img src="https://github.com/henryhmko/project_cerulean/blob/main/single_class/result_imgs/Landscape_Run2_Detailed_5.png" width="940"/>
</p>

#### **3. Training Run #3**
```python
epochs = 500 #should have increased epochs for this lr
lr = 1e-5 
bs = 16
diffusion_timesteps = 1000
```
> Training Time: ~9 hours

**Summary Result**
<p align="center">
  <img src="https://github.com/henryhmko/project_cerulean/blob/main/single_class/result_imgs/Landscape_Run3_Summary.png" width="940"/>
</p>

**Detailed Sampling**
<p align="center">
  <img src="https://github.com/henryhmko/project_cerulean/blob/main/single_class/result_imgs/Landscape_Run3_Detailed_1.png" width="940"/>
</p>
<p align="center">
  <img src="https://github.com/henryhmko/project_cerulean/blob/main/single_class/result_imgs/Landscape_Run3_Detailed_2.png" width="940"/>
</p>
<p align="center">
  <img src="https://github.com/henryhmko/project_cerulean/blob/main/single_class/result_imgs/Landscape_Run3_Detailed_3.png" width="940"/>
</p>
<p align="center">
  <img src="https://github.com/henryhmko/project_cerulean/blob/main/single_class/result_imgs/Landscape_Run3_Detailed_4.png" width="940"/>
</p>
<p align="center">
  <img src="https://github.com/henryhmko/project_cerulean/blob/main/single_class/result_imgs/Landscape_Run3_Detailed_5.png" width="940"/>
</p>

### Italian Greyhound Data
just an extension since  i like them
#### **1. Training Run #1**
```python
epochs = 3000
lr = 1e-4
bs = 12
diffusion_timesteps = 1000
``````
> Training Time: ~ (err look at tf log) hours

**Summary Result**
<p align="center">
  <img src="https://github.com/henryhmko/project_cerulean/blob/main/single_class/result_imgs/IG_Run1_Summary.png" width="940"/>
</p>

**Detailed Sampling**
<p align="center">
  <img src="https://github.com/henryhmko/project_cerulean/blob/main/single_class/result_imgs/IG_Run1_Detailed_1.png" width="940"/>
</p>

## Interesting Papers
- [Extracting Training Data from Diffusion Models](https://browse.arxiv.org/abs/2301.13188)
  * Shows how diffusion models spit out training data if prompted well. 
  * Security risks of diffusion models might be interesting to look at
- [Cold Diffusion: Inverting Arbitrary Image Transforms Without Noise](https://browse.arxiv.org/abs/2208.09392)
  * Questions the necessity of random Gaussian noise for noising.
  * Shows that other methods like burring or masking can be used in place of Gaussian noise to train diffusion models.
- [Understanding Diffusion Models: A Unified Perspective](https://browse.arxiv.org/abs/2208.11970)
  * Overview of various diffusion models along with background on...
    * ELBO, VAE
    * Variational Diffusion Models
    * Score-based Generative Models
    * Guidance(CG and CFG)
