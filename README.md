# sPINN: Deep Learning solutions to singular problems
This repository contains example code complementary to the pre-print https://arxiv.org/abs/2409.20150.

*Singular regular points often arise in differential equations describing physical phenomena such as fluid dynamics, electromagnetism, and gravitation. Traditional numerical techniques often fail or become unstable near these points, requiring the use of semi-analytical tools, such as series expansions and perturbative methods, in combination with numerical algorithms; or to invoke more sophisticated methods. In this work, we take an alternative route and leverage the power of machine learning to exploit Physics Informed Neural Networks (PINNs) as a modern approach to solving differential equations with singular points. PINNs utilize deep learning architectures to approximate solutions by embedding the differential equations into the loss function of the neural network. We discuss the advantages of PINNs in handling singularities, particularly their ability to bypass traditional grid-based methods and provide smooth approximations across irregular regions. Techniques for enhancing the accuracy of PINNs near singular points, such as adaptive loss weighting, are used in order to achieve high efficiency in the training of the network. We exemplify our results by studying four differential equations of interest in mathematics and gravitation -- the Legendre equation, the hypergeometric equation, the solution for black hole space-times in theories of Lorentz violating gravity, and the spherical accretion of a perfect fluid in a Schwarzschild geometry.*

## Skeletal overview

```bash
├── finite_diff_data/
│     ├── accretion.dat
├── sPINN/
│     ├── __init__.py
├── LICENSE
├── README.md
├── examples.ipynb
```
## Usage
Load the files with `import sPINN`. Working examples for the Legendre equation, the hypergeometric equation and accretion of a fluid in a curved space-time can be found in the notebook `examples.ipynb`.

## Citation
Please, if you use any of our work, consider citing the paper and this repository as follows

```text
@article{Cayuso:2024jau,
    author = "Cayuso, R. and Herrero-Valea, M. and Barausse, E.",
    title = "{Deep Learning solutions to singular problems: from special functions to spherical accretion}",
    eprint = "2409.20150",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    month = "9",
    year = "2024"
}
```
```text
@misc{github_sPINN,
  author = {Herrero-Valea, M.},
  title = {sPINN: Deep Learning solutions to singular problems},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/mehrrero/sPINN}}
}
```

