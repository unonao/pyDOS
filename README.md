Network Density of States in Python3 (pyDOS)
==============================
paper:  [Network Density of States](https://arxiv.org/abs/1905.09758) by Kun Dong, Austin Benson, David Bindel. 

original code(matlab & Python2):  https://github.com/kd383/NetworkDOS.

Data & library: [SNAP](http://snap.stanford.edu)

Data: [Repository of Difficult Graph Experiments and Results (RODGER)](https://www.cs.purdue.edu/homes/dgleich/rodger/)

Data: [The KONECT Project](http://konect.cc)

Data: [The SuiteSparse Matrix Collection (formerly the University of Florida Sparse Matrix Collection)](https://sparse.tamu.edu)

## Usage (demo)
If you want to use `demo_dos.py`,`demo_pdos.py`,`demo_dos_pdos.py`, or `demo_motif.py`, your current directory need to be `demo`.

```
cd demo
python demo_dos.py [<filepath> <method> Nz moment_num bin_num is_filter]
```

## Project Organization
------------
    ├── data 
    │   ├── fetch.sh    <- Download network eigenvalue tarball from David Gleich
    │   ├── gen_snap.ipynb    <- generate graph using SNAP (Colab)
    │   ├── erdos02-cc.mat   
    │   ├── facebook_combined.txt
    │   └── HepTh.mat
    │
    ├── demo
    │   ├── demo_dos.py    
    │   ├── demo_pdos.py   
    │   ├── demo_dos_pdos.py 
    │   ├── demo_motif.py   <- calculate DOS using Motif Filtering
    │   └── demo_heatmap.py <- calculate cosine distance of DOS and show heatmap
    │
    ├── pyDOS
    │   ├── moments     <- only 'cheb' (not 'lan', 'nd', 'exact')
    │   │   └── dos_by_cheb.py     <- moments for DOS & PDOS by cheb
    │   ├── cal_for_plot.py
    │   ├── convert_matrix.py
    │   ├── load_graph.py
    │   ├── moment_filter.py    <- only Jackson filter
    │   ├── motif_filter.py     <- only zero filter
    │   └── normalize_matrix.py
    │
    ├── requirements.txt
    │
    ├── heatmap.png <- cosine distances of DOS
    │
    └── README.md          <- The top-level README for developers using this project.

--------
