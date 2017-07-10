# A Markov Decision Process approach to optimizing cancer therapy using multiple treatment modalities

There are several different modalities, e.g., surgery, chemotherapy, and radiotherapy, that are currently used to treat cancer. It is common practice to use a combination of these modalities to maximize clinical outcomes, which are often measured by a balance between maximizing tumor damage and minimizing normal tissue side effects due to treatment. However, multi-modality treatment policies are mostly empirical in current practice, and are therefore subject to individual clinicians' experiences and intuition. We present a novel formulation of optimal multi-modality cancer management using a finite-horizon Markov decision process approach. Specifically, at each decision epoch, the clinician chooses an optimal treatment modality based on the patient's observed state, which we define as a combination of tumor progression and normal tissue side effect. Treatment modalities are categorized as (1) Type 1, which has a high risk and high reward, but is restricted in the frequency of administration during a treatment course, (2) Type 2, which has a lower risk and lower reward than Type 1, but may be repeated without restriction, and (3) Type 3, no treatment (surveillance), which has the possibility of reducing normal tissue side effect at the risk of worsening tumor progression. Numerical simulations using various intuitive, concave reward functions show the structural insights of optimal policies and demonstrate the potential applications of using a rigorous approach to optimizing multi-modality cancer management.

## Contents
* [Paper](https://arxiv.org/abs/1706.09481)
  
  Follow the link to access our paper on arXiv.org

* [Poster](https://github.com/kels271828/cancerMDP/blob/master/mdpPoster.pdf)

  Project poster for SIAM UW's annual poster competition (won 2nd place)
* [MDP code](https://github.com/kels271828/cancerMDP/blob/master/mdpMain.m)

  Compute the optimal multi-modality cancer treatment policy using backward induction for stationary transition probabilities.
* [GUI code](https://github.com/kels271828/cancerMDP/blob/master/mdpGUI.m)

  Interactive GUI that allows user to experiment with different transition probabilities and reward functions for three treatment modalities and three treatment periods. Note: I specified GUI dimensions so that it looks okay on my MacBook Pro, but things might look weird on different computers or monitors. 

