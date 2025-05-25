# pyhough Package

This package provides codes to perform the (Generalized) frequency-Hough transform to search for (transient) continuous gravitational waves from asymmetrically rotating neutron stars, from primordial black hole binaries, and from newborn neutron stars.

The frequency-Hough transform is a pattern-recognition technique that maps points in the time-frequency plane to lines in the frequency-spindown plane of the source. It is essentially a clever way to search over different time-frequency tracks, where each track corresponds to a particular gravitational-wave signal.

In the package, which builds off of spectrograms created by Pyfstat, one can create a time/frequency peakmap, doppler correct it, and perform the Hough Transform.

The frequency-Hough Transform can be applied to either the spectrogram directly after thresholding (and selecting local maxima) to create the peakmap

The Generalized frequency-Hough transform is implemented, but no Python codes exist yet to inject and recover PBH inspirals or signals from newborn neutron stars. Help is welcome on these fronts.

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

