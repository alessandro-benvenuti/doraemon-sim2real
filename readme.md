# Starting code for final course project extension of Robot Learning - 01HFNOV

Official assignment at [Google Doc](https://docs.google.com/document/d/1XWE2NB-keFvF-EDT_5muoY8gdtZYwJIeo48IMsnr9l0/edit?usp=sharing).


# DORAEMON+: Enhancing Adaptive Domain Randomization via Performance-Gated Warmup

**Authors:** Alessandro Benvenuti, Irene Bartolini

**Institution:** Politecnico di Torino / University of Bologna

## 🎥 Demo & Visualizations

We validate our approach on high-dimensional continuous control tasks. Below are visual demonstrations of the policies learned using DORAEMON+ compared to standard baselines.

### 🦘 Hopper: Robustness to Unmodeled Dynamics

| **Nominal Dynamics** | **Extreme Edge Case** |
| --- | --- |
| `<video src="img/hopper_original.mov" controls width="100%"></video>` | `<video src="img/hopper_low_mass_low_friction.mov" controls width="100%"></video>` |
| *Standard performance on nominal physics.* | *Robust recovery under low mass/friction.* |

### 🐆 Half-Cheetah: Peak Performance vs. Conservatism

`<video src="img/half_cheetah.mp4" controls width="100%"></video>` 

---

## 📖 Abstract

Bridging the reality gap remains a critical challenge in deploying Deep Reinforcement Learning (DRL) policies onto physical systems. This project introduces **DORAEMON+**, an enhanced sim-to-real transfer framework based on *Domain Randomization via Entropy Maximization*.

We introduce a novel **Performance-Gated Warmup** strategy to stabilize policy initialization. This phase mitigates the "cold start" problem where random policies fail to provide useful gradients for curriculum updates. Experimental results on MuJoCo (Hopper & Half-Cheetah) demonstrate that this architecture significantly reduces transfer variance and achieves higher asymptotic performance compared to Uniform Domain Randomization (UDR).

---

## ⚙️ Methodology

### 1. The Problem: Gradient Failure at Initialization

Standard active domain randomization methods often fail at the start of training. When a policy is randomly initialized, its success rate is near zero (). Consequently, the gradient for the distribution parameters becomes uninformative:



This leads to stagnation or random drift in the environment parameters.

### 2. The Solution: Performance-Gated Warmup

We implement a "latching" mechanism that freezes the distribution parameters  until the agent achieves a minimum competence threshold .

* **Static Phase:** Train on nominal dynamics until .
* **Adaptive Phase:** Unlock  and maximize entropy  subject to the success constraint.

---

## 📊 Experiments & Results

### Experiment 1: Hopper (Emergent Robustness)

We evaluated the agent's ability to generalize to unseen dynamics (e.g., Torso Mass) even when those parameters were fixed during training.

* **Training Dynamics:** The agent successfully expands the distribution entropy over time.
* *See:* `img/hopper_training_performance.png`


* **Zero-Shot Generalization:** The agent maintains >90% performance within a [-1.5kg, +1.0kg] torso mass shift.
*

### Experiment 2: Half-Cheetah (7-Dimensional Randomization)

We compared DORAEMON+ against Uniform Domain Randomization (UDR) across a complex 7D parameter space (masses of all links + friction).

#### Heatmap Analysis: DORAEMON vs. UDR

These heatmaps visualize the "Feasibility Manifold" (Reward) across varying Friction and Mass shifts.

| **DORAEMON (Ours)** | **Uniform DR (UDR)** |
| --- | --- |
|  |  |
| *High-reward plateau () concentrated around feasible physics.* | *Lower, diffuse performance () due to overly conservative training.* |

#### Detailed Robustness Profiles

While UDR maintains marginal stability at extreme outliers, DORAEMON dominates in the "likely" physics range.

* **Friction Shift Analysis:**
* *DORAEMON:* `img/hc_doraemon_shift_friction.png`
* *UDR:* `img/hc_udr_shift_friction.png`


* **Mass Shift Analysis:**
* *DORAEMON:* `img/hc_doraemon_shift_mass.png`
* *UDR:* `img/hc_udr_shift_mass.png`



---

## 🚀 Usage

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/doraemon-plus.git
cd doraemon-plus

# Install dependencies
pip install -r requirements.txt
# Requires MuJoCo and Gymnasium

```

### Training

To train the DORAEMON agent on Hopper with the warmup enabled:

```bash
python train.py --env Hopper-v4 --algorithm doraemon --warmup True --seed 42

```

### Evaluation

To generate the heatmaps and videos found in the `img/` folder:

```bash
python eval.py --model_path checkpoints/best_model.pth --plot_type heatmap

```

---

## 📝 Citation

If you use this code or method, please cite our work:

```bibtex
@inproceedings{benvenuti2025doraemon,
  title={DORAEMON+: Enhancing Adaptive Domain Randomization via Performance-Gated Warmup},
  author={Benvenuti, Alessandro and Bartolini, Irene},
  booktitle={Politecnico di Torino M.Sc. Thesis},
  year={2025}
}

```