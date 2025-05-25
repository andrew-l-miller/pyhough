This package contains some basic functions to create time/frequency peakmaps and run the frequency-Hough Transform on them. Users are highly encouraged to go through the [tutorial](https://colab.research.google.com/drive/1Ym90BZyehZbj7cfUZU82jGcJ2J1iBdcU#scrollTo=mYJ6RwMZxxmK) available on [my website](https://andrew-l-miller.github.io/post/tutorial/).

These codes rely on [PyFstat](https://github.com/PyFstat/PyFstat), [LAL](https://doi.org/10.7935/GT1W-FZ16), and codes therein and are conceptually based on the [frequency-Hough transform](https://doi.org/10.1103/PhysRevD.90.042002) and codes in the [Snag toolbox](https://www.mathworks.com/matlabcentral/fileexchange/9703-snag).

The package [pyhough](https://pypi.org/project/pyhough/) is installable in the following way:

```
pip install pyhough

```
Please contact me if you have any questions!

Note: The frequency-Hough transform algorithm and its implementation for CW searches has been designed and implemented for real searches by the Virgo Rome group of the LVK collaboration. It can be cited as:
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
```

```
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
