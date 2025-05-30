# pyhough

[![DOI](https://zenodo.org/badge/753611572.svg)](https://doi.org/10.5281/zenodo.15512454)
[![PyPI version](https://badge.fury.io/py/pyhough.svg)](https://pypi.org/project/pyhough/)
![License](https://img.shields.io/github/license/andrew-l-miller/pyhough)

`pyhough` is a Python package implementing the (Generalized) Frequency-Hough transform for searching **(transient) continuous gravitational waves** from:

- Asymmetrically rotating neutron stars
- Planetary-mass primordial black hole (PBH) binaries
- Newborn neutron stars

This method maps time-frequency tracks from spectrograms (e.g., created by [PyFstat](https://github.com/PyFstat/PyFstat)) into the frequency–spindown parameter space, allowing efficient searches for weak, long-duration gravitational-wave signals.

The frequency-Hough Transform can be applied to either the spectrogram directly after thresholding (and selecting local maxima) to create the peakmap

The Generalized frequency-Hough transform is implemented, but no Python codes exist yet to inject and recover PBH inspirals or signals from newborn neutron stars. Help is welcome on these fronts.

---

## Features

- Construct time–frequency peakmaps from preprocessed data
- Doppler correction for sources in the sky
- Standard Frequency-Hough transform for persistent CW signals
- Generalized Frequency-Hough transform for transient or chirping signals

---

## Installation

```bash
pip install pyhough
```

---
## Contributions

Contributions are welcome, especially in the following areas:

- Signal injection and recovery tools for PBH binaries and newborn neutron stars

- Unit tests and test coverage

- Improving documentation and usage examples

- Feel free to open issues or submit pull requests!


If you use this code, please cite the public, version-independent Zenodo entry: 

[![DOI](https://zenodo.org/badge/753611572.svg)](https://doi.org/10.5281/zenodo.15512454)

and also cite the papers that are the basis behind the codes:

The frequency-Hough has been developed by the Rome Virgo group for all-sky searches for continuous waves from non-axisymmetric, rotating neutron stars and can be cited as:
```
@article{Astone:2014esa,
    author = "Astone, Pia and Colla, Alberto and D'Antonio, Sabrina and Frasca, Sergio and Palomba, Cristiano",
    title = "{Method for all-sky searches of continuous gravitational wave signals using the frequency-Hough transform}",
    eprint = "1407.8333",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.IM",
    doi = "10.1103/PhysRevD.90.042002",
    journal = "Phys. Rev. D",
    volume = "90",
    number = "4",
    pages = "042002",
    year = "2014"
}
```

The Generalized Frequency-Hough transform has been developed by the Rome Virgo group for transient continuous-wave searches for newborn neutron stars and can be cited as:

```
@article{Miller:2018rbg,
    author = "Miller, Andrew and others",
    title = "{Method to search for long duration gravitational wave transients from isolated neutron stars using the generalized frequency-Hough transform}",
    eprint = "1810.09784",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.IM",
    doi = "10.1103/PhysRevD.98.102004",
    journal = "Phys. Rev. D",
    volume = "98",
    number = "10",
    pages = "102004",
    year = "2018"
}
```

It has been further generalized to search for gravitational waves from inspiraling planetary-mass primordial black hole binaries:

```
@article{Miller:2020kmv,
    author = "Miller, Andrew L. and Clesse, S\'ebastien and De Lillo, Federico and Bruno, Giacomo and Depasse, Antoine and Tanasijczuk, Andres",
    title = "{Probing planetary-mass primordial black holes with continuous gravitational waves}",
    eprint = "2012.12983",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.HE",
    doi = "10.1016/j.dark.2021.100836",
    journal = "Phys. Dark Univ.",
    volume = "32",
    pages = "100836",
    year = "2021"
}

@article{Miller:2024jpo,
    author = "Miller, Andrew L. and Aggarwal, Nancy and Clesse, Sebastien and De Lillo, Federico and Sachdev, Surabhi and Astone, Pia and Palomba, Cristiano and Piccinni, Ornella J. and Pierini, Lorenzo",
    title = "{Method to search for inspiraling planetary-mass ultracompact binaries using the generalized frequency-Hough transform in LIGO O3a data}",
    eprint = "2407.17052",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.IM",
    doi = "10.1103/PhysRevD.110.082004",
    journal = "Phys. Rev. D",
    volume = "110",
    number = "8",
    pages = "082004",
    year = "2024"
}
```

