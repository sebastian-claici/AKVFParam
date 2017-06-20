#ifndef AKVF_PARAMETRIZATION_H
#define AKVF_PARAMETRIZATION_H

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <string>

#include "FastLsBuildUtils.h"
#include "LocalWeightedArapParametrizer.h"
#include "ParametrizationAlgorithm.h"

#include <igl/jet.h>
#include <igl/readOBJ.h>
#include <igl/facet_components.h>
#include <igl/slice.h>
#ifdef HAS_GUI
#include <igl/viewer/Viewer.h>
#endif

extern "C" {
  /* PARDISO prototype. */
  void pardisoinit (void   *, int    *,   int *, int *, double *, int *);
  void pardiso     (void   *, int    *,   int *, int *,    int *, int *,
                    double *, int    *,    int *, int *,   int *, int *,
                    int *, double *, double *, int *, double *);
  void pardiso_chkmatrix  (int *, int *, double *, int *, int *, int *);
  void pardiso_chkvec     (int *, int *, double *, int *);
  void pardiso_printstats (int *, int *, double *, int *, int *, int *,
                           double *, int *);

  void pardiso_residual(int *, int *, double *, int *, int *, double *, double *, double *, double *, double *);

}

class AKVFParametrization : public ParametrizationAlgorithm {

 public:

  AKVFParametrization(StateManager& state_manager, Param_State* m_state);

  /**
   * @brief Update elements in KVF matrix.
   *
   * Avoids allocating a new chunk of memory and avoids the call to
   * setFromTriplets().
   *
   * K must be preallocated and of size 2 * V.rows() * 2 * V.rows()
   */
  void update_pattern(const Eigen::MatrixXd &V,
                      const Eigen::MatrixXi &F,
                      Eigen::SparseMatrix<double> &K);

  /**
   * @brief Construct KVF operator matrix.
   */
  void kvf_operator(const Eigen::MatrixXd &V,
                    const Eigen::MatrixXi &F,
                    Eigen::SparseMatrix<double> &K);

  /**
   * @brief Perform initial Tutte parameterization, and symbolic factorization of KVF matrix.
   */
  void init_parametrization(std::ofstream &fout, bool flaps = true);

  /**
   * @brief Solve d^{k+1} = K^{-1} x^k
   */
  void linear_solve(Eigen::MatrixXd &grad,
                    Eigen::MatrixXd &dest_res);

  bool single_iteration(std::ofstream &fout,
                        double eps=1e-5);

 private:

  SymmetricDirichlet* symmd_p;

  // PARDISO variables
  int      mtype = -2;        /* Real symmetric matrix */
  int      nrhs  = 1;

  /* Internal solver memory pointer pt,                  */
  /* 32-bit: int pt[64]; 64-bit: long int pt[64]         */
  /* or void *pt[64] should be OK on both architectures  */
  void    *pt[64];

  /* Pardiso control parameters. */
  int      iparm[64];
  double   dparm[64];
  int      maxfct, mnum, phase, error, msglvl, solver;

  /* Number of processors. */
  int      num_procs;

  double   ddum;              /* Double dummy */
  int      idum;              /* Integer dummy. */

  // PARDISO variables
  int *ia;
  int *ja;
  double *res;

  // saved pattern information
  std::vector<int> nonzero_col;
  int nnz, n;

  // time keeping
  double total_time = 0;
};

#endif // GLOBAL_LOCAL_PARAMETRIZATION_H
