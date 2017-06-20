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

`$ ./AKVFParam horse_b.obj`
