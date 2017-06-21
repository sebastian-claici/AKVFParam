# As-Killing-As-Possible Mesh Parameterization
This repository contains source code for the implementation used in the SGP 2017 paper:

"Isometry-Aware Preconditioning for Mesh Parameterization"
Sebastian Claici, Mikhail Bessmeltsev, Scott Schaefer, and Justin Solomon

This implementation is based on the SLIM implementation found at [https://github.com/MichaelRabinovich/Scalable-Locally-Injective-Mappings].

We have tested this software as-is on Ubuntu 16.04.

The implementation requires PARDISO to solve a sparse linear system. Due to licensing
requirements, we cannot include PARDISO in the archive, but note that academic licenses
are available from [http://www.pardiso-project.org/].

To run, invoke the binary as:

`$ OMP_NUM_THREADS=4 ./AKVFParam horse_b.obj`

Acknowledgments:  
This work was supported in part by NSF
CAREER award IIS 1148976. J. Solomon acknowledges funding from an MIT Skoltech Seed Fund grant (“Boundary Element
Methods for Shape Analysis”) and from the MIT Research Support Committee (“Structured Optimization for Geometric Problems”), as well as Army Research Office grant W911NF-12-R-0011 (“Smooth Modeling of Flows on Graphs”).
