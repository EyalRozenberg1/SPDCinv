This repo contains the official implementation of the paper:

# Inverse Design of Spontaneous Parametric Down Conversion for Generation of High-Dimensional Qudits
![illustration](illustration.png)

## about
(**If you use the code, please _cite our papers_.**)
We have introduced an algorithm for solving the inverse design problem of generating structured and entangled photon pairs in quantum optics, using tailored nonlinear interactions in the SPDC process. The *SPDCinv* algorithm extracts the optimal physical parameters which yield a desired quantum state or correlations between structured photon-pairs, that can then be used in future experiments. To ensure convergence to  realizable results and to improve the predictive accuracy, our algorithm obeyed physical constraints through the integration of the time-unfolded propagation dynamics governing the interaction of the SPDC Hamiltonian.

### in this version
We have shown how we can apply our algorithm to obtain the optimal nonlinear $\chi^{(2)}$ volume holograms (2D/3D) as well as different pump structures for generating the desired maximally-entangled states. Using this version, one can further obtain all-optical coherent control over the generated quantum states by actively changing the profile of the pump beam.

## extensions
This work can readily be extended to the spectral-temporal domain, by allowing non-periodic volume holograms along the propagation axis -- making it possible to shape the joint spectral amplitude of the photon pairs. Furthermore, one can easily adopt our approach for other optical systems, such as: nonlinear waveguides and resonators, $\chi^{(3)}$ effects (e.g. spontaneous four wave mixing), spatial solitons, fiber optics communication systems, and even higher-order coincidence probabilities. Moreover, the algorithm can be upgraded to include passive optical elements such as beam-splitters, holograms, and mode sorters, thereby providing greater flexibility for generating and manipulating quantum optical states. The *SPDCinv* algorithm can incorporate decoherence mechanisms arising from non-perturbative high-order photon pair generation in the high gain regime. Other decoherence effects due to losses such as absorption and scattering can be incorporated into the model in the future. Finally, the current scheme can be adapted to other quantum systems sharing a similar Hamiltonian structure, such as superfluids and superconductors.

## Data Availability
Data underlying the results presented in this paper are available at `SPDCinv/Data availability/`.

## running the code
To understand and determine the variables of interaction and learning hyperparameters, see `src/spdc_inv/experiments/experiment.py` and read the documentation there.

Before you run the code, please run the following line from the bash: 
`export PYTHONPATH="${PYTHONPATH}:/home/jupyter/src"`
Later, run experiment.py by: 
`python src/spdc_inv/experiments/experiment.py`

## Giving Credit
If you use this code in your work, please cite the associated papers.

```
@article{Rozenberg:22,
  author = {Eyal Rozenberg and Aviv Karnieli and Ofir Yesharim and Joshua Foley-Comer and Sivan Trajtenberg-Mills and Daniel Freedman and Alex M. Bronstein and Ady Arie},
  journal = {Optica},
  keywords = {Computation methods; Four wave mixing; Nonlinear photonic crystals; Quantum information processing; Quantum key distribution; Quantum optics},
  number = {6},
  pages = {602--615},
  publisher = {Optica Publishing Group},
  title = {Inverse design of spontaneous parametric downconversion for generation of high-dimensional qudits},
  volume = {9},
  month = {Jun},
  year = {2022},
  url = {http://opg.optica.org/optica/abstract.cfm?URI=optica-9-6-602},
  doi = {10.1364/OPTICA.451115},
  abstract = {Spontaneous parametric downconversion (SPDC) in quantum optics is an invaluable resource for the realization of high-dimensional qudits with spatial modes of light. One of the main open challenges is how to directly generate a desirable qudit state in the SPDC process. This problem can be addressed through advanced computational learning methods; however, due to difficulties in modeling the SPDC process by a fully differentiable algorithm, progress has been limited. Here, we overcome these limitations and introduce a physically constrained and differentiable model, validated against experimental results for shaped pump beams and structured crystals, capable of learning the relevant interaction parameters in the process. We avoid any restrictions induced by the stochastic nature of our physical model and integrate the dynamic equations governing the evolution under the SPDC Hamiltonian. We solve the inverse problem of designing a nonlinear quantum optical system that achieves the desired quantum state of downconverted photon pairs. The desired states are defined using either the second-order correlations between different spatial modes or by specifying the required density matrix. By learning nonlinear photonic crystal structures as well as different pump shapes, we successfully show how to generate maximally entangled states. Furthermore, we simulate all-optical coherent control over the generated quantum state by actively changing the profile of the pump beam. Our work can be useful for applications such as novel designs of high-dimensional quantum key distribution and quantum information processing protocols. In addition, our method can be readily applied for controlling other degrees of freedom of light in the SPDC process, such as spectral and temporal properties, and may even be used in condensed-matter systems having a similar interaction Hamiltonian.},
}
```

```
@inproceedings{Rozenberg:21,
  author = {Eyal Rozenberg and Aviv Karnieli and Ofir Yesharim and Sivan Trajtenberg-Mills and Daniel Freedman and Alex M. Bronstein and Ady Arie},
  booktitle = {Conference on Lasers and Electro-Optics},
  journal = {Conference on Lasers and Electro-Optics},
  keywords = {Light matter interactions; Nonlinear optical crystals; Nonlinear photonic crystals; Photonic crystals; Quantum communications; Quantum optics},
  pages = {FM1N.7},
  publisher = {Optica Publishing Group},
  title = {Inverse Design of Quantum Holograms in Three-Dimensional Nonlinear Photonic Crystals},
  year = {2021},
  url = {http://opg.optica.org/abstract.cfm?URI=CLEO_QELS-2021-FM1N.7},
  doi = {10.1364/CLEO_QELS.2021.FM1N.7},
  abstract = {We introduce a systematic approach for designing 3D nonlinear photonic crystals and pump beams for generating desired quantum correlations between structured photon-pairs. Our model is fully differentiable, allowing accurate and efficient learning and discovery of novel designs.},
}
```
