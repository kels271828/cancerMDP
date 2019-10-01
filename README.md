# A Markov Decision Process approach to optimizing cancer therapy using multiple treatment modalities

There are several different modalities, e.g., surgery, chemotherapy, and radiotherapy, that are currently used to treat cancer. It is common practice to use a combination of these modalities to maximize clinical outcomes, which are often measured by a balance between maximizing tumor damage and minimizing normal tissue side effects due to treatment. However, multi-modality treatment policies are mostly empirical in current practice and are therefore subject to individual clinicians' experiences and intuition. We present a novel formulation of optimal multi-modality cancer management using a finite-horizon Markov decision process approach. Specifically, at each decision epoch, the clinician chooses an optimal treatment modality based on the patient's observed state, which we define as a combination of tumor progression and normal tissue side effect. Treatment modalities are categorized as (1) type 1, which has a high risk and high reward, but is restricted in the frequency of administration during a treatment course; (2) type 2, which has a lower risk and lower reward than type 1, but may be repeated without restriction; and (3) type 3, no treatment (surveillance), which has the possibility of reducing normal tissue side effect at the risk of worsening tumor progression. Numerical simulations using various intuitive, concave reward functions show the structural insights of optimal policies and demonstrate the potential applications of using a rigorous approach to optimizing multi-modality cancer management.

## Documents
* [Paper](https://doi.org/10.1093/imammb/dqz004), March 2019
* [Preprint](https://arxiv.org/abs/1706.09481), June 2017
* [Poster](https://github.com/kels271828/cancerMDP/blob/master/mdpPoster.pdf), January 2017

## Code
* [Main](https://github.com/kels271828/cancerMDP/blob/master/mdpMain.m): Compute the optimal multi-modality cancer treatment policy using backward induction for stationary transition probabilities
* [GUI](https://github.com/kels271828/cancerMDP/blob/master/mdpGUI.m): Interactive GUI that allows user to experiment with different transition probabilities and reward functions for three treatment modalities and three treatment periods
* [Examples](https://github.com/kels271828/cancerMDP/blob/master/examples.m): Code to reproduce the examples from our paper
