+++
title = 'Math Derivation of Diffusion Models'
date = 2023-11-18T10:40:31+08:00
draft = false
math = true
author = 'Author: Chaoyang Zhu'
+++


Diffusion models are inspired by non-equilibrium thermodynamics. 
They define a Markov chain of diffusion steps to slowly add random noise to data, 
then learn to reverse the diffusion process to construct desired data samples from the noise.

# Forward Diffusion

Suppose a multi-variate random variable $\mathbf{X}=[X\_1,\dots,X\_L] \sim q(\bm{x})$, 
where $L=H\*W\*3$ and $\bm{x}=[x\_1,\dots,x\_L]$, **i.e.**, the observed image, $\bm{x}\_0 \in \mathbb{R}^{L}$, 
is sampled from the underlying true distribution $q(\bm{x})$.

Starting from the observation $\bm{x}\_0$, 
forward diffusion progressively adds gaussian noise sampled from $\mathcal{N}(\bm{\mu}\_t,\bm{\Sigma}\_t)$ in $T$ steps, 
producing a sequence of noisy samples $\bm{x}\_1,\dots,\bm{x}\_T$.

$$
\begin{equation}
q(\bm{x}\_{t}|\bm{x}\_{t-1})=\mathcal{N}(\bm{x}\_t;\bm{\mu}\_t=\sqrt{1-\beta\_t}\bm{x}\_{t-1},\bm{\Sigma}\_t=\beta\_t\mathbf{I}),
\end{equation}
$$

where $\mathbf{I} \in \mathbb{R}^{L\times L}$ is an identity matrix, $\beta\_t$ (a floating point number) is the variance of the gaussian noise at step $t$.

## Reparameterization Trick

Sampling $\bm{x}\_{t}$ is not differentiable, and $\bm{x}\_{t-1},\dots,\bm{x}\_{1}$ have to be sequentially sampled first in order to sample $\bm{x}\_{t}$. Reparameterization tricks introduces an auxiliary independent variable $\bm{\epsilon}\_t \sim \mathcal{N}(\mathbf{0},\mathbf{I}) \in \mathbb{R}^L$ such that

$$
\begin{equation}\bm{x}\_{t}=\sqrt{1-\beta\_t}\bm{x}\_{t-1}+\sqrt{\beta\_t}*\bm{\epsilon}\_t,\end{equation}
$$

therefore it is still differentiable. We can verify that

$$
\begin{align}
\mathbf{E}[\bm{x}\_{t}] 
&=\mathbf{E}[\sqrt{1-\beta\_t}\bm{x}\_{t-1}+\sqrt{\beta\_t}\*\bm{\epsilon}\_t] \\\\
&=\mathbf{E}[\sqrt{1-\beta\_t}\bm{x}\_{t-1}]+\mathbf{E}[\sqrt{\beta\_t}\*\bm{\epsilon}\_t] \\\\
&=\mathbf{E}[\sqrt{1-\beta\_t}\bm{x}\_{t-1}]+\mathbf{0} \\\\
&=\mathbf{E}[\sqrt{1-\beta\_t}\bm{x}\_{t-1}] \\\\
&=\sqrt{1-\beta\_t}\mathbf{E}[\bm{x}\_{t-1}] \\\\
&=\sqrt{1-\beta\_t}\bm{x}\_{t-1},
\end{align}
$$

and 

$$
\begin{align}
\bm{\Sigma}[\bm{x}\_{t}]
&=\bm{\Sigma}[\sqrt{1-\beta\_t}\bm{x}\_{t-1}+\sqrt{\beta\_t}\*\bm{\epsilon}\_t]\\\\
&=\bm{\Sigma}[\sqrt{1-\beta\_t}\bm{x}\_{t-1}]+\bm{\Sigma}[\sqrt{\beta\_t}\*\bm{\epsilon}\_t] \notag\\\\
&+2\bm{Cov}(\sqrt{1-\beta\_t}\bm{x}\_{t-1}, \sqrt{\beta\_t}\*\bm{\epsilon}\_t)\\\\
&=(1-\beta\_t)\bm{\Sigma}[\bm{x}\_{t-1}]+\beta\_t*\bm{\Sigma}[\bm{\epsilon}\_t]+\mathbf{0}\\\\
&=\mathbf{0}+\beta\_t*\bm{\Sigma}[\bm{\epsilon}\_t]+\mathbf{0}\\\\
&=\beta\_t\mathbf{I},
\end{align}
$$

where $\alpha\_t=1-\beta\_t, \bar{\alpha\_{t}}=\prod\_{k=1}^t\alpha_k$. 

Since two independent gaussians is still gaussians, i.e., 
$\mathcal{N}(\mathbf{0}, \sigma\_1^2\mathbf{I}) + \mathcal{N}(\mathbf{0},\sigma\_2^2\mathbf{I}) \sim \mathcal{N}(\mathbf{0}, \sigma_1^2\mathbf{I}+\sigma_2^2\mathbf{I})$, therefore $\sqrt{(1-\beta\_t)\beta\_{t-1}}\bm{\epsilon}\_{t-1}+\sqrt{\beta\_{t}}\bm{\epsilon}\_t \sim \mathcal{N}(\mathbf{0}, ((1-\beta\_t)\beta\_{t-1}+\beta\_t)\mathbf{I})=\mathcal{N}(\mathbf{0}, (1-\alpha\_t\alpha_{t-1})\mathbf{I})$, using reparameterization trick again we get $\sqrt{1-\alpha\_t\alpha\_{t-1}}\bm{\epsilon}$.

$$
\begin{equation}q(\bm{x}\_t|\bm{x}\_0)\sim\mathcal{N}(\bm{x}\_t;\bm{\mu}\_t=\sqrt{\bar{\alpha}\_t}\bm{x}\_{0},\bm{\Sigma}\_t=(1-\bar{\alpha}\_t)\mathbf{I})\end{equation}
$$

## Variance Schedule

The variance of Gaussian noise in each step is increasing, $0 < \beta\_{1} < \beta\_{2} < \dots < \beta\_{T} < 1$, 
just like learning rate schedule but in ascending order, variance schedule defines each value of variance across $T$ timesteps. 
As $T$ goes up, $\beta\_t$ increases, $\alpha\_t$ decreases, $\bar{\alpha\_t}$ decreases faster, 
eventually $q(\bm{x}\_t|\bm{x}\_0)\sim \mathcal{N}(\mathbf{0}, \mathbf{I})$, nearly isotropic Gaussian. 
DDPM adopts a linear schedule, i.e., $T=300,\beta\_1=0.0001,\beta\_T=0.02$.

# Reverse Diffusion

![DDPM](/images/DDPM.png "Image source: [Lilien Weng's Blog](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/#forward-diffusion-process).")

If we know the distribution $q(\bm{x}{t-1}|\bm{x}{t})$, then we can start from $\bm{x}T\sim\mathcal{N}(\bm{0},\mathbf{I})$, 
then run the reverse process and acquire a sample from the underlying distribution $q(\bm{x})$, synthesizing a novel image from the observed data distribution. 
But in practice, we do not know this $q(\bm{x}{t-1}|\bm{x}t)$, instead we approximate $q(\bm{x}{t-1}|\bm{x}t)$ with a parameterized neural network $p\theta(\bm{x}{t-1}|\bm{x}t)$, and if $\beta\_t$ is small enough, $q(\bm{x}{t-1}|\bm{x}t)$ will still be gaussian. Then we can $p(\bm{x}{0:T})=p(\bm{x}T)\prod{t=1}^Tp\theta(\bm{x}{t-1}|\bm{x}t)$ and choose $p\theta(\bm{x}{t-1}|\bm{x}t)$ to just model the mean and variance of $q(\bm{x}{t-1}|\bm{x}{t})$. In short, $p\theta(\bm{x}\_{t-1}|\bm{x}t)$ needs to learn $q(\bm{x}{t-1}|\bm{x}\_t)$, since they are two distributions, their distance can be measured by KL-Divergence.

## Evidence Lower Bound

Given an observed dataset $\mathcal{D}=\{\bm{x}^1, \dots, \bm{x}^N\}$ (images) following i.i.d., we want to maximize its joint distribution to generate realistic images:

$$
\begin{equation}
\textbf{maximize}~~~p(\bm{x}^1,\dots,\bm{x}^N)=\log p(\bm{x}^1,\dots,\bm{x}^N)=\sum_{k=1}^N\log p(\bm{x}^k),
\end{equation}
$$

recall that $\bm{x}\_0$ is an observed data sample (same as $\bm{x}^i$), in fact we assume that observations are generated by an unseen latent multi-variate $\bm{z}$, the likelihood/chance of observing $\bm{x}\_0$ is given by

$$
\begin{equation}
p(\bm{x}\_0)=\int p(\bm{x}\_0,\bm{z})d\bm{z}=\frac{p(\bm{x}\_0,\bm{z})}{p(\bm{z}|\bm{x}\_0)},
\end{equation}
$$

diffusion models, instead of associating the image variable with one latent, it associates each data sample with multiple latents $\bm{z}\_1,\dots,\bm{z}\_T$, 
forming a markov chain. $\bm{z}\_i$ is harder to observe and more abstract (high-level) than $\bm{z}\_j, \forall i > j$. 
And each latent is only dependent on its predecessor. With a slight abuse of notation, $\bm{x}\_1,\dots,\bm{x}\_T$ are those latents instead of representing them with $\bm{z}$, now the likelihood of observing data point $\bm{x}\_0$ is given by

$$
\begin{equation}
p(\bm{x}\_0)=\int p(\bm{x}\_0,\bm{x}\_{1:T})d\bm{x}\_{1:T}=\frac{p(\bm{x}\_0,\bm{x}\_{1:T})}{p(\bm{x}\_{1:T}|\bm{x}\_0)},
\end{equation}
$$

Let's decompose $\log p(\bm{x}\_0)$ to derive the ELBO that we aim to optimize

$$
\begin{align}
\log p(\bm{x}\_0) 
&= \log p(\bm{x}\_0)\int q(\bm{x}\_{1:T}|\bm{x}\_0)d\bm{x}\_{1:T} & \\\\
&= \int q(\bm{x}\_{1:T}|\bm{x}\_0)\log p(\bm{x}\_0)d\bm{x}\_{1:T} & \\\\
&= \mathbf{E}\_{q(\bm{x}\_{1:T}|\bm{x}\_0)}[\log p(\bm{x}\_0)] & \\\\
&= \mathbf{E}\_{q(\bm{x}\_{1:T}|\bm{x}\_0)}\left[\log \frac{p(\bm{x}\_0,\bm{x}\_{1:T})}{p(\bm{x}\_{1:T}|\bm{x}\_0)}\right] & \\\\
&= \mathbf{E}\_{q(\bm{x}\_{1:T}|\bm{x}\_0)}\left[\log \frac{p(\bm{x}\_0,\bm{x}\_{1:T})q(\bm{x}\_{1:T}|\bm{x}\_0)}{p(\bm{x}\_{1:T}|\bm{x}\_0)q(\bm{x}\_{1:T}|\bm{x}\_0)}\right] & \\\\
&= \mathbf{E}\_{q(\bm{x}\_{1:T}|\bm{x}\_0)}\left[\log \frac{p(\bm{x}\_0,\bm{x}\_{1:T})}{q(\bm{x}\_{1:T}|\bm{x}\_0)}\right]+ \mathbf{E}\_{q(\bm{x}\_{1:T}|\bm{x}\_0)}\left[\log \frac{q(\bm{x}\_{1:T}|\bm{x}\_0)}{p(\bm{x}\_{1:T}|\bm{x}\_0)}\right]& \\\\
&= \mathbf{E}\_{q(\bm{x}\_{1:T}|\bm{x}\_0)}\left[\log \frac{p(\bm{x}\_0,\bm{x}\_{1:T})}{q(\bm{x}\_{1:T}|\bm{x}\_0)}\right]+ \mathcal{D}\_{KL}(q(\bm{x}\_{1:T}|\bm{x}\_0)~||~p(\bm{x}\_{1:T}|\bm{x}\_0)) & \\\\
&\geq \mathbf{E}\_{q(\bm{x}\_{1:T}|\bm{x}\_0)}\left[\log \frac{p(\bm{x}\_0,\bm{x}\_{1:T})}{q(\bm{x}\_{1:T}|\bm{x}\_0)}\right] &
\end{align}
$$

From Eq.(24) to Eq.(25), the integral over all possible values of $\bm{x}\_{1:T}$ equals to 1. And from Eq.(30) to Eq.(31), KL-Divergence is always greater than 0.

$$
\begin{align}
\log p(\bm{x}\_0) & \geq\mathbf{E}\_{q(\bm{x}\_{1: T} | \bm{x}\_0)}\left[\log \frac{p(\bm{x}\_T) \prod\_{t=1}^T p\_{\bm{\theta}}(\bm{x}\_{t-1} | \bm{x}\_t)}{\prod\_{t=1}^T q(\bm{x}\_t | \bm{x}\_{t-1})}\right] & \\\\
&=\mathbf{E}\_{q(\bm{x}\_{1: T} | \bm{x}\_0)}\left[\log \frac{p(\bm{x}\_T) p\_{\bm{\theta}}(\bm{x}\_0 | \bm{x}\_1) \prod\_{t=2}^T p\_{\bm{\theta}}(\bm{x}\_{t-1} | \bm{x}\_t)}{q(\bm{x}\_1 | \bm{x}\_0) \prod\_{t=2}^T q(\bm{x}\_t | \bm{x}\_{t-1})}\right] &\\\\
&=\mathbf{E}\_{q(\bm{x}\_{1: T} | \bm{x}\_0)}\left[\log \frac{p(\bm{x}\_T) p\_{\bm{\theta}}(\bm{x}\_0 | \bm{x}\_1) \prod\_{t=2}^T p\_{\bm{\theta}}(\bm{x}\_{t-1} | \bm{x}\_t)}{q(\bm{x}\_1 | \bm{x}\_0) \prod\_{t=2}^T q(\bm{x}\_t | \bm{x}\_{t-1}, \bm{x}\_0)}\right] &\\\\
&=\mathbf{E}\_{q(\bm{x}\_{1: T} | \bm{x}\_0)}\left[\log \frac{p(\bm{x}\_T) p\_{\bm{\theta}}(\bm{x}\_0 | \bm{x}\_1)}{q(\bm{x}\_1 | \bm{x}\_0)}+\log \prod\_{t=2}^T \frac{p\_{\theta}(\bm{x}\_{t-1} | \bm{x}\_t)}{q(\bm{x}\_t | \bm{x}\_{t-1}, \bm{x}\_0)}\right] &\\\\
&=\mathbf{E}\_{q(\bm{x}\_{1: T} | \bm{x}\_0)}\left[\log \frac{p(\bm{x}\_T) p\_{\bm{\theta}}(\bm{x}\_0 | \bm{x}\_1)}{q(\bm{x}\_1 | \bm{x}\_0)}+\log \prod\_{t=2}^T \frac{p\_\theta(\bm{x}\_{t-1} | \bm{x}\_t)}{\frac{q(\bm{x}\_{t-1}|\bm{x}\_t,\bm{x}\_0)q(\bm{x}\_t|\bm{x}\_0)}{q(\bm{x}\_{t-1}|\bm{x}\_0)}}\right] &\\\\
&=\mathbf{E}\_{q(\bm{x}\_{1: T} | \bm{x}\_0)}\left[\log \frac{p(\bm{x}\_T) p\_{\bm{\theta}}(\bm{x}\_0 | \bm{x}\_1)}{q(\bm{x}\_1 | \bm{x}\_0)}+\log \prod\_{t=2}^T \frac{p\_{\bm{\theta}}(\bm{x}\_{t-1} | \bm{x}\_t)}{\frac{q(\bm{x}\_{t-1} | \bm{x}\_t, \bm{x}\_0) \cancel{q(\bm{x}\_t|\bm{x}\_0)}}{\cancel{q(\bm{x}\_{t-1} | \bm{x}\_0)}}}\right] &\\\\
&=\mathbf{E}\_{q(\bm{x}\_{1: T} | \bm{x}\_0)}\left[\log \frac{p(\bm{x}\_T) p\_{\bm{\theta}}(\bm{x}\_0 | \bm{x}\_1)}{q(\bm{x}\_1 | \bm{x}\_0)}+\log \frac{q(\bm{x}\_1|\bm{x}\_0)}{q(\bm{x}\_T | \bm{x}\_0)}+\log \prod\_{t=2}^T \frac{p\_{\bm{\theta}}(\bm{x}\_{t-1} | \bm{x}\_t)}{q(\bm{x}\_{t-1} | \bm{x}\_t, \bm{x}\_0)}\right] &\\\\
&=\mathbf{E}\_{q(\bm{x}\_{1: T} | \bm{x}\_0)}\left[\log \frac{p(\bm{x}\_T) p\_{\bm{\theta}}(\bm{x}\_0 | \bm{x}\_1)}{\cancel{q(\bm{x}\_1 | \bm{x}\_0)}}+\log \frac{\cancel{q(\bm{x}\_1|\bm{x}\_0)}}{q(\bm{x}\_T | \bm{x}\_0)}+\log \prod\_{t=2}^T \frac{p\_{\bm{\theta}}(\bm{x}\_{t-1} | \bm{x}\_t)}{q(\bm{x}\_{t-1} | \bm{x}\_t, \bm{x}\_0)}\right] &\\\\
&=\mathbf{E}\_{q(\bm{x}\_{1: T} | \bm{x}\_0)}\left[\log \frac{p(\bm{x}\_T) p\_{\bm{\theta}}(\bm{x}\_0 | \bm{x}\_1)}{q(\bm{x}\_T | \bm{x}\_0)}+\sum\_{t=2}^T \log \frac{p\_{\bm{\theta}}(\bm{x}\_{t-1} | \bm{x}\_t)}{q(\bm{x}\_{t-1} | \bm{x}\_t, \bm{x}\_0)}\right] &\\\\
&=\mathbf{E}\_{q(\bm{x}\_{1} | \bm{x}\_0)}\left[\log p\_{\bm{\theta}}(\bm{x}\_0 | \bm{x}\_1)\right] + \mathbf{E}\_{q(\bm{x}\_{1: T} | \bm{x}\_0)}\left[\log \frac{p(\bm{x}\_T)}{q(\bm{x}\_T | \bm{x}\_0)}\right] &\notag\\\\
& +\sum\_{t=2}^T \mathbf{E}\_{q(\bm{x}\_{1: T} | \bm{x}\_0)}\left[\log \frac{p\_{\bm{\theta}}(\bm{x}\_{t-1} | \bm{x}\_t)}{q(\bm{x}\_{t-1} | \bm{x}\_t, \bm{x}\_0)}\right] &\\\\
&=\underbrace{\mathbf{E}\_{q(\bm{x}\_1 | \bm{x}\_0)}\left[\log p\_{\bm{\theta}}(\bm{x}\_0 | \bm{x}\_1)\right]}\_{\text {Reconstruction}} -\underbrace{\mathcal{D}\_{KL}(q(\bm{x}\_T | \bm{x}\_0) \| p(\bm{x}\_T))}\_{\text {Prior Matching}} \notag\\\\
& -\sum\_{t=2}^T \underbrace{\mathbf{E}\_{q(\bm{x}\_t | \bm{x}\_0)}\left[\mathcal{D}\_{KL}(q(\bm{x}\_{t-1} | \bm{x}\_t, \bm{x}\_0) \| p\_{\bm{\theta}}(\bm{x}\_{t-1} | \bm{x}\_t))\right]}\_{\text {Denoising Matching}} &\\\\
\end{align}
$$

- Reconstruction:
- Prior Matching: since we assume that $p(\bm{x}\_T)$ is sampled from $\mathcal{N}(\bm{0},\mathbf{I})$, and as $T$ is large enough → $\infty$, $q(\bm{x}\_T|\bm{x}\_0)$ will be an isotrophic gaussian, so the two distributions are identical, this tem does not need to be optimized.
- Denoising Matching: We learn desired denoising transition step $p\_\theta(\bm{x}\_{t-1}|\bm{x}\_t)$ as an approximation to tractable, ground-truth denoising transition step $q(\bm{x}\_{t-1} | \bm{x}\_t, \bm{x}\_0)$ (later we’ll show why it is tractable), as $q(\bm{x}\_{t-1} | \bm{x}\_t)$ is not tractable. The $q(\bm{x}\_{t-1} | \bm{x}\_t, \bm{x}\_0)$ transition step can act as a ground-truth signal, since it defines how to denoise a noisy image $\bm{x}\_t$ with access to what the final, completely denoised image $\bm{x}\_0$ should be

So, by minimizing the denoising matching term, we maximize the bulk of lower bound Eq.(31) of the evidence $p(\bm{x}\_0)$, since the summation over time steps dominates the loss. So we can ensure that the optimization will maximize the model’s likelihood of generating real data samples.

Why $q(\bm{x}\_{t-1} | \bm{x}\_t, \bm{x}\_0)$ is tractable?

$$
\begin{align}
q(\bm{x}\_{t-1} | \bm{x}\_t, \bm{x}\_0) 
&=\frac{q(\bm{x}\_t, \bm{x}\_{t-1}, \bm{x}\_0)}{q(\bm{x}\_t, \bm{x}\_0)} \\\\
&= \frac{q(\bm{x}\_t, \bm{x}\_{t-1}, \bm{x}\_0)/q(\bm{x}\_0)}{q(\bm{x}\_t, \bm{x}\_0)/q(\bm{x}\_0)} \\\\
&=\frac{\left[q(\bm{x}\_t, \bm{x}\_{t-1}, \bm{x}\_0)/q(\bm{x}\_{t-1}, \bm{x}\_0)\right]\left[q(\bm{x}\_{t-1}, \bm{x}\_0)/q(\bm{x}\_0)\right]}{q(\bm{x}\_t | \bm{x}\_0)} \\\\
&=\frac{q(\bm{x}\_t | \bm{x}\_{t-1}, \bm{x}\_0) q(\bm{x}\_{t-1} | \bm{x}\_0)}{q(\bm{x}\_t | \bm{x}\_0)},
\end{align}
$$

As Eq.(1) and Eq.(20) shows, we already know exactly the form of $q(\bm{x}\_t|\bm{x}\_0)$ and $q(\bm{x}\_t|\bm{x}\_{t-1})$, thus we can deduce the form of the above three distributions

In Eq.(47), thanks to the markov property, each latent $\bm{x}\_t$ is only dependent on its predecessor $\bm{x}\_{t-1}$, so $q(\bm{x}\_t|\bm{x}\_{t-1},\bm{x}\_0)=q(\bm{x}\_t|\bm{x}\_{t-1})$. 
Recall the probability density function of normal distribution, and substitute Eq.(47-49) into Eq.(46), we get

$$
\begin{align}
& q(\bm{x}\_{t-1}|\bm{x}\_t,\bm{x}\_0) \\\\
& \propto \exp -\frac{1}{2}\left[\frac{(\bm{x}\_t-\sqrt{\alpha\_t} \bm{x}\_{t-1})^2}{1-\alpha\_t}+\frac{(\bm{x}\_{t-1}-\sqrt{\bar{\alpha}\_{t-1}} \bm{x}\_0)^2}{1-\bar{\alpha}\_{t-1}}-\frac{(\bm{x}\_t-\sqrt{\bar{\alpha}\_t} \bm{x}\_0)^2}{1-\bar{\alpha}\_t}\right] \\\\
& =\exp -\frac{1}{2}\left[\frac{-2 \sqrt{\alpha\_t} \bm{x}\_t \bm{x}\_{t-1}+\alpha\_t \bm{x}\_{t-1}^2}{1-\alpha\_t}+\frac{\bm{x}\_{t-1}^2-2 \sqrt{\bar{\alpha}\_{t-1}} \bm{x}\_{t-1} \bm{x}\_0}{1-\bar{\alpha}\_{t-1}}+C(\bm{x}\_t, \bm{x}\_0)\right] \\\\
& \propto \exp -\frac{1}{2}\left[-\frac{2 \sqrt{\alpha\_t} \bm{x}\_t \bm{x}\_{t-1}}{1-\alpha\_t}+\frac{\alpha\_t \bm{x}\_{t-1}^2}{1-\alpha\_t}+\frac{\bm{x}\_{t-1}^2}{1-\bar{\alpha}\_{t-1}}-\frac{2 \sqrt{\bar{\alpha}\_{t-1}} \bm{x}\_{t-1} \bm{x}\_0}{1-\bar{\alpha}\_{t-1}}\right] \\\\
& =\exp -\frac{1}{2}\left[(\frac{\alpha\_t}{1-\alpha\_t}+\frac{1}{1-\bar{\alpha}\_{t-1}}) \bm{x}\_{t-1}^2-2(\frac{\sqrt{\alpha\_t} \bm{x}\_t}{1-\alpha\_t}+\frac{\sqrt{\bar{\alpha}\_{t-1}} \bm{x}\_0}{1-\bar{\alpha}\_{t-1}}) \bm{x}\_{t-1}\right]
\end{align}
$$

From Eq.(50) to Eq.(51), we move terms involving $\bm{x}\_t,\bm{x}\_0$ into $C(\bm{x}\_t,\bm{x}\_0)$ as they are not variables (they are already known and given as input), we only care about $\bm{x}\_{t-1}$ in the conditional distribution $q(\bm{x}\_{t-1}|\bm{x}\_t,\bm{x}\_0)$. Next, we rearrange Eq.(53) following the gaussian distribution form, first, the variance coefficient for $q(\bm{x}\_{t-1}|\bm{x}\_t,\bm{x}\_0)$ is

$$
\begin{align}
\tilde{\beta\_t}
&=1/\left(\frac{\alpha\_t}{1-\alpha\_t}+\frac{1}{1-\bar{\alpha}\_{t-1}}\right)\\\\
&=1/\left(\frac{\alpha\_t(1-\bar{\alpha}\_{t-1})+1-\alpha\_t}{(1-\alpha\_t)(1-\bar{\alpha}\_{t-1})}\right)\\\\
&=1/\left(\frac{\alpha\_t-\bar{\alpha}\_t+1-\alpha\_t}{(1-\alpha\_t)(1-\bar{\alpha}\_{t-1})}\right)\\\\
&=\frac{\beta\_t(1-\bar{\alpha}\_{t-1})}{1-\bar{\alpha}\_t}
\end{align}
$$

Then, the mean for $q(\bm{x}\_{t-1}|\bm{x}\_t,\bm{x}\_0)$ is

$$
\begin{align}
\tilde{\bm{\mu}}\_t
&=\left(\frac{\sqrt{\alpha\_t} \bm{x}\_t}{1-\alpha\_t}+\frac{\sqrt{\bar{\alpha}\_{t-1}} \bm{x}\_0}{1-\bar{\alpha}\_{t-1}}\right)/\left(\frac{\alpha\_t}{1-\alpha\_t}+\frac{1}{1-\bar{\alpha}\_{t-1}}\right)\\\\
&=\left(\frac{\sqrt{\alpha\_t} \bm{x}\_t}{\beta\_t}+\frac{\sqrt{\bar{\alpha}\_{t-1}} \bm{x}\_0}{1-\bar{\alpha}\_{t-1}}\right)\tilde{\beta}\_t\\\\
&=\left(\frac{\sqrt{\alpha\_t} \bm{x}\_t}{\beta\_t}+\frac{\sqrt{\bar{\alpha}\_{t-1}} \bm{x}\_0}{1-\bar{\alpha}\_{t-1}}\right)\frac{\beta\_t(1-\bar{\alpha}\_{t-1})}{1-\bar{\alpha}\_t}\\\\
&=\frac{\sqrt{\alpha\_t}(1-\bar{\alpha}\_{t-1})}{1-\bar{\alpha}\_t}\bm{x}\_t+\frac{\beta\_t\sqrt{\bar{\alpha}\_{t-1}}}{1-\bar{\alpha}\_t}\bm{x}\_0
\end{align}
$$

Recall Eq.(19) using reparameterization trick, we can represent $\bm{x}\_0=\frac{1}{\sqrt{\bar{\alpha}}\_t}(\bm{x}\_t-\sqrt{1-\bar{\alpha}\_t}\bm{\epsilon}\_t)$, substitute it with Eq.(61), the mean becomes

$$
\begin{align}
\tilde{\bm{\mu}}\_t
&=\frac{\sqrt{\alpha\_t}(1-\bar{\alpha}\_{t-1})}{1-\bar{\alpha}\_t}\bm{x}\_t+\frac{\beta\_t\sqrt{\bar{\alpha}\_{t-1}}}{1-\bar{\alpha}\_t}\frac{1}{\sqrt{\bar{\alpha}}\_t}(\bm{x}\_t-\sqrt{1-\bar{\alpha}\_t}\bm{\epsilon}\_t)\\\\
&=\frac{\alpha\_t(1-\bar{\alpha}\_{t-1})+\beta\_t}{(1-\bar{\alpha}\_t)\sqrt{\alpha\_t}}\bm{x}\_t-\frac{\beta\_t}{\sqrt{1-\bar{\alpha}\_t}\sqrt{\alpha\_t}}\bm{\epsilon}\_t\\\\
&=\frac{1-\bar{\alpha}\_t}{(1-\bar{\alpha}\_t)\sqrt{\alpha\_t}}\bm{x}\_t-\frac{\beta\_t}{\sqrt{1-\bar{\alpha}\_t}\sqrt{\alpha\_t}}\bm{\epsilon}\_t\\\\
&=\frac{1}{\sqrt{\alpha}\_t}(\bm{x}\_t-\frac{\beta\_t}{\sqrt{1-\bar{\alpha}\_t}}\bm{\epsilon\_t})
\end{align}
$$

Finally we derive the analytical form of the conditional distribution $q(\bm{x}\_{t-1}|\bm{x}\_t,\bm{x}\_0)$ from Eq.(53), resembling the gaussian distribution

$$
\begin{align}
q(\bm{x}\_{t-1}|\bm{x}\_t,\bm{x}\_0)
& \propto \exp -\frac{1}{2}\left[(\frac{\alpha\_t}{1-\alpha\_t}+\frac{1}{1-\bar{\alpha}\_{t-1}}) \bm{x}\_{t-1}^2-2(\frac{\sqrt{\alpha\_t} \bm{x}\_t}{1-\alpha\_t}+\frac{\sqrt{\bar{\alpha}\_{t-1}} \bm{x}\_0}{1-\bar{\alpha}\_{t-1}}) \bm{x}\_{t-1}\right]\\\\
&=\exp -\frac{1}{2\tilde{\beta}\_t}(\bm{x}\_{t-1}-\tilde{\bm{\mu}}\_t)^2
\end{align}
$$

So, we have 

$$
\begin{equation}
q(\bm{x}\_{t-1}|\bm{x}\_t,\bm{x}\_0)=\mathcal{N}(\bm{x}\_{t-1};\tilde{\bm{\mu}}\_t=\frac{1}{\sqrt{\alpha}\_t}(\bm{x}\_t-\frac{\beta\_t}{\sqrt{1-\bar{\alpha}\_t}}\bm{\epsilon\_t}),\tilde{\bm{\Sigma}}\_t=\tilde{\beta}\_t\mathbf{I})
\end{equation}
$$

The KL-Divergence between two gaussians are given by

$$
\begin{equation}
\mathcal{D}\_{KL}\left(\mathcal{N}\left(\boldsymbol{x} ; \boldsymbol{\mu}\_x, \boldsymbol{\Sigma}\_x\right) \| \mathcal{N}\left(\boldsymbol{y} ; \boldsymbol{\mu}\_y, \boldsymbol{\Sigma}\_y\right)\right)=\frac{1}{2}\left[\log \frac{\left|\boldsymbol{\Sigma}\_y\right|}{\left|\boldsymbol{\Sigma}\_x\right|}-d+\operatorname{tr}\left(\boldsymbol{\Sigma}\_y^{-1} \boldsymbol{\Sigma}\_x\right)+\left(\boldsymbol{\mu}\_y-\boldsymbol{\mu}\_x\right)^T \boldsymbol{\Sigma}\_y^{-1}\left(\boldsymbol{\mu}\_y-\boldsymbol{\mu}\_x\right)\right]
\end{equation}
$$

Now let’s derive the analytical form of the denoising term, since the variance coefficient $\tilde{\beta}\_t$ of $q(\bm{x}\_{t-1}|\bm{x}\_t,\bm{x}\_0)$ is a constant, so we can simply let the variance of $p\_\theta(\bm{x}\_t|\bm{x}\_{t-1})$ equal to $\tilde{\beta}\_t$ without letting the neural network learn to predict the variance.

$$
\begin{align}
& \underset{\theta}{\arg \min } \mathcal{D}\_{KL}(q(\bm{x}\_{t-1} | \bm{x}\_t, \bm{x}\_0) \| p\_{\theta}(\bm{x}\_{t-1} | \bm{x}\_t)) \\\\
= & \underset{\theta}{\arg \min } \mathcal{D}\_{KL}(\mathcal{N}(\bm{x}\_{t-1} ; \tilde{\bm{\mu}}\_t, \tilde{\bm{\Sigma}}\_t) \| \mathcal{N}(\bm{x}\_{t-1} ; \bm{\mu}\_{\theta}, \tilde{\bm{\Sigma}}\_t) \\\\
= & \underset{\theta}{\arg \min } \frac{1}{2}\left[\log \frac{\left|\tilde{\bm{\Sigma}}\_t\right|}{\left|\tilde{\bm{\Sigma}}\_t\right|}-d+\operatorname{tr}(\tilde{\bm{\Sigma}}\_t^{-1} \tilde{\bm{\Sigma}}\_t)+(\bm{\mu}\_{\theta}-\tilde{\bm{\mu}}\_t)^T \tilde{\bm{\Sigma}}\_t^{-1}(\bm{\mu}\_{\theta}-\tilde{\bm{\mu}}\_t)\right] \\\\
= & \underset{\theta}{\arg \min } \frac{1}{2}\left[\log 1-d+d+(\bm{\mu}\_{\theta}-\tilde{\bm{\mu}}\_t)^T \tilde{\bm{\Sigma}}\_t^{-1}(\bm{\mu}\_{\theta}-\tilde{\bm{\mu}}\_t)\right] \\\\
= & \underset{\theta}{\arg \min } \frac{1}{2}\left[(\bm{\mu}\_{\theta}-\tilde{\bm{\mu}}\_t)^T \tilde{\bm{\Sigma}}\_t^{-1}(\bm{\mu}\_{\theta}-\tilde{\bm{\mu}}\_t)\right] \\\\
= & \underset{\theta}{\arg \min } \frac{1}{2}\left[(\bm{\mu}\_{\theta}-\tilde{\bm{\mu}}\_t)^T(\tilde{\beta}\_t^2 \mathbf{I})^{-1}(\bm{\mu}\_{\theta}-\tilde{\bm{\mu}}\_t)\right] \\\\
= & \underset{\theta}{\arg \min } \frac{1}{2 \tilde{\beta}\_t^2}\left[\left\|\bm{\mu}\_{\theta}-\tilde{\bm{\mu}}\_t\right\|\_2^2\right]
\end{align}
$$

In code implementation, instead of minimizing the difference between $\bm{\mu}\_\theta$ and $\tilde{\bm{\mu}}\_t$, we let the neural network to predict the noise $\bm{\epsilon}\_t$ in $\tilde{\bm{\mu}}\_t$ and minimize the difference between $\bm{\epsilon}\_t$ and $\bm{\epsilon}\_\theta$

$$
\begin{align}
& \underset{\theta}{\arg \min } \mathcal{D}\_{KL}(q(\bm{x}\_{t-1} | \bm{x}\_t, \bm{x}\_0) \| p\_{\theta}(\bm{x}\_{t-1} | \bm{x}\_t)) \\\\
= & \underset{\theta}{\arg \min } \mathcal{D}\_{KL}(\mathcal{N}(\bm{x}\_{t-1} ; \tilde{\bm{\mu}}\_t, \tilde{\bm{\Sigma}}\_t) \| \mathcal{N}(\bm{x}\_{t-1} ; \bm{\mu}\_{\theta}, \tilde{\bm{\Sigma}}\_t)) \\\\
= & \underset{\theta}{\arg \min } \frac{1}{2 \tilde{\beta}\_t^2}\left[\left\|\bm{\mu}\_{\theta}-\tilde{\bm{\mu}}\_t\right\|\_2^2\right] \\\\
= & \underset{\theta}{\arg \min }\frac{1}{2 \tilde{\beta}\_t^2}\left[\left\|\frac{1}{\sqrt{\alpha}\_t}(\bm{x}\_t-\frac{\beta\_t}{\sqrt{1-\bar{\alpha}\_t}}\bm{\epsilon}\_\theta)-\frac{1}{\sqrt{\alpha}\_t}(\bm{x}\_t-\frac{\beta\_t}{\sqrt{1-\bar{\alpha}\_t}}\bm{\epsilon}\_t)\right\|\_2^2\right] \\\\
= & \underset{\theta}{\arg \min }\frac{1}{2 \tilde{\beta}\_t^2}\left[\left\|\frac{\beta\_t}{\sqrt{\alpha\_t}\sqrt{1-\bar{\alpha}\_t}}(\bm{\epsilon}\_t-\bm{\epsilon}\_\theta)\right\|\_2^2\right]\\\\
= & \underset{\theta}{\arg \min }\frac{1}{2 \tilde{\beta}\_t^2}\frac{\beta\_t^2}{\alpha\_t(1-\bar{\alpha}\_t)}\left[\left\|(\bm{\epsilon}\_t-\bm{\epsilon}\_\theta)\right\|\_2^2\right]
\end{align}
$$

For each reverse diffusion, the input is $\bm{x}\_t$, and the output is the predicted noise vector $\bm{\epsilon}\_\theta$ that mimicks $\bm{\epsilon}\_t$ applied on $\bm{x}\_0$ using $\bm{x}\_0=\frac{1}{\sqrt{\bar{\alpha}}\_t}(\bm{x}\_t-\sqrt{1-\bar{\alpha}\_t}\bm{\epsilon}\_t)$

# Inference

Since $p\_\theta(\bm{x}\_{t-1}|\bm{x}\_t)=\mathcal{N}(\bm{x}\_{t-1} ; \bm{\mu}\_{\theta}, \tilde{\bm{\Sigma}}\_t)$, according to the reparameterization trick, we have

$$
\begin{align}
\bm{x}\_{t-1}&=\bm{\mu}\_\theta+\tilde{\beta}\_t*\bm{\epsilon}\\&=\frac{1}{\sqrt{\alpha}\_t}(\bm{x}\_t-\frac{\beta\_t}{\sqrt{1-\bar{\alpha}\_t}}\bm{\epsilon}\_\theta)+\tilde{\beta}\_t*\bm{\epsilon}
\end{align}
$$

# References

[1] **[From Autoencoder to Beta-VAE](https://lilianweng.github.io/posts/2018-08-12-vae)**

[2] **[What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models)**

[3] **[Diffusion Models: Toward State-of-the-Art Image Generation](https://theaisummer.com/diffusion-models)**

[4] **[Understanding Diffusion Models: A Unified Perspective](https://arxiv.org/pdf/2208.11970.pdf)**

[5] **[Diffusion Models, Paper Explanation](https://www.youtube.com/watch?v=HoKDTa5jHvg)**

[6] **[A Diffusion Model from Scratch in Pytorch](https://colab.research.google.com/drive/1sjy9odlSSy0RBVgMTgP7s99NXsqglsUL?usp=sharing#scrollTo=k13hj2mciCHA)**

[7] **[Diffusion Models | Pytorch Implementation](https://github.com/dome272/Diffusion-Models-pytorch)**