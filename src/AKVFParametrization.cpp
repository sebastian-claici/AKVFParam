#include "AKVFParametrization.h"

#include "Param_State.h"
#include "eigen_stl_utils.h"
#include "parametrization_utils.h"
#include "LinesearchParametrizer.h"

#include <igl/boundary_loop.h>
#include <igl/cotmatrix.h>
#include <igl/edge_lengths.h>
#include <igl/local_basis.h>
#include <igl/min_quad_with_fixed.h>
#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <igl/repdiag.h>
#include <igl/vector_area_matrix.h>

#include <chrono>
#include <iostream>

#undef NDEBUG
#include <assert.h>
#define NDEBUG

const int DEBUG = 1;

using namespace std;

AKVFParametrization::AKVFParametrization(StateManager& state_manager, Param_State* m_state) :
ParametrizationAlgorithm(state_manager, m_state), symmd_p(NULL) {
  assert (m_state->F.cols() == 3);

  symmd_p = new SymmetricDirichlet(m_state);
}

void AKVFParametrization::update_pattern(const Eigen::MatrixXd& V,
                                         const Eigen::MatrixXi& F,
                                         Eigen::SparseMatrix<double>& K) {
  int nv = V.rows();
  int nt = F.rows();

  K = Eigen::SparseMatrix<double>(2*nv, 2*nv);
  K.reserve(nonzero_col);
  for (int t = 0; t < nt; ++t) {
    int u = F(t,0);
    int v = F(t,1);
    int w = F(t,2);

    double x1, y1, x2, y2, x3, y3;
    x1 = V(u, 0), x2 = V(v, 0), x3 = V(w, 0);
    y1 = V(u, 1), y2 = V(v, 1), y3 = V(w, 1);

    double x12 = x1 - x2;
    double y12 = y1 - y2;
    double x23 = x2 - x3;
    double y23 = y2 - y3;
    double x31 = x3 - x1;
    double y31 = y3 - y1;

    double area = std::abs ( y12 * x31 -x12 * y31 ); // really the sqrt of area
    area = area * area / m_state->M(t);
    area = sqrt(0.5 * area);
    x12 /= area;
    y12 /= area;
    x23 /= area;
    y23 /= area;
    x31 /= area;
    y31 /= area;

    // diagonals
    //u u
    K.coeffRef(2*u, 2*u) += x23*x23 + 2*y23*y23;
    K.coeffRef(2*u+1,2*u) += -x23*y23;
    K.coeffRef(2*u+1,2*u+1) += 2*x23*x23 + y23*y23;

    //v v
    K.coeffRef(2*v,2*v) += x31*x31 + 2*y31*y31;
    K.coeffRef(2*v+1,2*v) += -x31*y31;
    K.coeffRef(2*v+1,2*v+1) += 2*x31*x31 + y31*y31;

    //w w
    K.coeffRef(2*w,2*w) += x12*x12 + 2*y12*y12;
    K.coeffRef(2*w+1,2*w) += -x12*y12;
    K.coeffRef(2*w+1,2*w+1) += 2*x12*x12 + y12*y12;

    // 3 conditionals per triangle
    if ( u > v )
    {
      K.coeffRef(2*u,2*v) += x31*x23 + 2*y23*y31;
      K.coeffRef(2*u+1,2*v) += -x31*y23;
      K.coeffRef(2*u+1,2*v+1) += 2*x31*x23 + y23*y31;
      K.coeffRef(2*u,2*v+1) += -x23*y31;
    }
    else
    {
      K.coeffRef(2*v,2*u) += x31*x23 + 2*y23*y31;
      K.coeffRef(2*v+1,2*u) += -x23*y31;
      K.coeffRef(2*v+1,2*u+1) += 2*x31*x23 + y23*y31;
      K.coeffRef(2*v,2*u+1) += -x31*y23;
    }

    if ( u > w )
    {
      K.coeffRef(2*u,2*w) += x12*x23 + 2*y12*y23;
      K.coeffRef(2*u+1,2*w) += -x12 * y23;
      K.coeffRef(2*u+1,2*w+1) += 2 * x12 * x23 + y12 * y23;
      K.coeffRef(2*u,2*w+1) += -x23 * y12;
    }
    else
    {
      K.coeffRef(2*w,2*u) += x12*x23+2*y12*y23;
      K.coeffRef(2*w+1,2*u) += -x23 * y12;
      K.coeffRef(2*w+1,2*u+1) += 2 * x12 * x23 + y12 * y23;
      K.coeffRef(2*w,2*u+1) += -x12 * y23;
    }

    if( v > w )
    {
      K.coeffRef(2*v,2*w) += x12 * x31 + 2 * y12 * y31;
      K.coeffRef(2*v+1,2*w) += -x12 * y31;
      K.coeffRef(2*v,2*w+1) += -x31 * y12;
      K.coeffRef(2*v+1,2*w+1) += 2 * x12 * x31 + y12 * y31;
    }
    else
    {
      K.coeffRef(2*w,2*v) += x12 * x31 + 2 * y12 * y31;
      K.coeffRef(2*w+1,2*v) += -x31 * y12;
      K.coeffRef(2*w,2*v+1) += -x12 * y31;
      K.coeffRef(2*w+1,2*v+1) += 2 * x12 * x31 + y12 * y31;
    }
  }
  for (int i = 0; i < 2 * nv; ++i)
    K.coeffRef(i, i) += 0.0001;

  K.makeCompressed();
}

void AKVFParametrization::kvf_operator(const Eigen::MatrixXd& V,
                                       const Eigen::MatrixXi& F,
                                       Eigen::SparseMatrix<double>& K) {
  int nv = V.rows();
  int nt = F.rows();

  K = Eigen::SparseMatrix<double>(2*nv, 2*nv);
  std::vector<Eigen::Triplet<double>> IJV;
  IJV.reserve(21 * nt);

  for (int t = 0; t < nt; ++t) {
    int u = F(t,0);
    int v = F(t,1);
    int w = F(t,2);

    double x1, y1, x2, y2, x3, y3;
    x1 = V(u, 0), x2 = V(v, 0), x3 = V(w, 0);
    y1 = V(u, 1), y2 = V(v, 1), y3 = V(w, 1);

    double x12 = x1 - x2;
    double y12 = y1 - y2;
    double x23 = x2 - x3;
    double y23 = y2 - y3;
    double x31 = x3 - x1;
    double y31 = y3 - y1;

    double area = std::abs ( y12 * x31 -x12 * y31 ); // really the sqrt of area
    area = area * area / m_state->M(t);
    area = sqrt(0.5 * area);
    x12 /= area;
    y12 /= area;
    x23 /= area;
    y23 /= area;
    x31 /= area;
    y31 /= area;

    // diagonals
    //u u
    IJV.push_back(Eigen::Triplet<double>(2*u, 2*u, x23*x23 + 2*y23*y23));
    IJV.push_back(Eigen::Triplet<double>(2*u+1, 2*u+1, 2*x23*x23 + y23*y23));
    IJV.push_back(Eigen::Triplet<double>(2*u+1, 2*u, -x23*y23));

    //v v
    IJV.push_back(Eigen::Triplet<double>(2*v, 2*v, x31*x31 + 2*y31*y31));
    IJV.push_back(Eigen::Triplet<double>(2*v+1, 2*v+1, 2*x31*x31 + y31*y31));
    IJV.push_back(Eigen::Triplet<double>(2*v+1, 2*v, -x31*y31));

    //w w
    IJV.push_back(Eigen::Triplet<double>(2*w, 2*w, x12*x12 + 2*y12*y12));
    IJV.push_back(Eigen::Triplet<double>(2*w+1, 2*w+1, 2*x12*x12 + y12*y12));
    IJV.push_back(Eigen::Triplet<double>(2*w+1, 2*w, -x12*y12));

    // 3 conditionals per triangle
    if ( u > v )
    {
      IJV.push_back(Eigen::Triplet<double>(2*u, 2*v, x31*x23 + 2*y23*y31));
      IJV.push_back(Eigen::Triplet<double>(2*u+1, 2*v+1, 2*x31*x23 + y23*y31));
      IJV.push_back(Eigen::Triplet<double>(2*u+1, 2*v, -x31*y23));
      IJV.push_back(Eigen::Triplet<double>(2*u, 2*v+1, -x23*y31));
    }
    else
    {
      IJV.push_back(Eigen::Triplet<double>(2*v, 2*u, x31*x23 + 2*y23*y31));
      IJV.push_back(Eigen::Triplet<double>(2*v+1, 2*u+1, 2*x31*x23 + y23*y31));
      IJV.push_back(Eigen::Triplet<double>(2*v+1, 2*u, -x23*y31));
      IJV.push_back(Eigen::Triplet<double>(2*v, 2*u+1, -x31*y23));
    }

    if ( u > w )
    {
      IJV.push_back(Eigen::Triplet<double>(2*u, 2*w, x12*x23 + 2*y12*y23));
      IJV.push_back(Eigen::Triplet<double>(2*u+1, 2*w+1, 2 * x12 * x23 + y12 * y23));
      IJV.push_back(Eigen::Triplet<double>(2*u+1, 2*w, -x12 * y23));
      IJV.push_back(Eigen::Triplet<double>(2*u, 2*w+1, -x23 * y12));
    }
    else
    {
      IJV.push_back(Eigen::Triplet<double>(2*w, 2*u, x12*x23+2*y12*y23));
      IJV.push_back(Eigen::Triplet<double>(2*w+1, 2*u+1, 2 * x12 * x23 + y12 * y23));
      IJV.push_back(Eigen::Triplet<double>(2*w+1, 2*u, -x23 * y12));
      IJV.push_back(Eigen::Triplet<double>(2*w, 2*u+1, -x12 * y23));
    }

    if( v > w )
    {
      IJV.push_back(Eigen::Triplet<double>(2*v, 2*w, x12 * x31 + 2 * y12 * y31));
      IJV.push_back(Eigen::Triplet<double>(2*v+1, 2*w+1, 2 * x12 * x31 + y12 * y31));
      IJV.push_back(Eigen::Triplet<double>(2*v+1, 2*w, -x12 * y31));
      IJV.push_back(Eigen::Triplet<double>(2*v, 2*w+1, -x31 * y12));
    }
    else
    {
      IJV.push_back(Eigen::Triplet<double>(2*w, 2*v, x12 * x31 + 2 * y12 * y31));
      IJV.push_back(Eigen::Triplet<double>(2*w+1, 2*v+1, 2 * x12 * x31 + y12 * y31));
      IJV.push_back(Eigen::Triplet<double>(2*w, 2*v+1, -x12 * y31));
      IJV.push_back(Eigen::Triplet<double>(2*w+1, 2*v, -x31 * y12));
    }
  }
  for (int i = 0; i < 2 * nv; ++i)
    IJV.push_back(Eigen::Triplet<double>(i, i, 0.0001));

  K.setFromTriplets(IJV.begin(), IJV.end());
}

void AKVFParametrization::init_parametrization(std::ofstream &fout,
                                               bool flaps) {
  dirichlet_on_circle(m_state->V,m_state->F,m_state->uv);
  if (count_flips(m_state->V,m_state->F,m_state->uv) > 0)
    tutte_on_circle(m_state->V,m_state->F,m_state->uv);
  m_state->energy = symmd_p->compute_energy(m_state->V, m_state->F, m_state->uv)/m_state->mesh_area;

  total_time = 0;
  fout << "iters,time,energy" << std::endl;
  fout << 0 << "," << 0 << "," << m_state->energy << std::endl;

  // Figure out AKVF sparsity pattern
  Eigen::SparseMatrix<double> K;
  kvf_operator(m_state->uv, m_state->F, K);

  nonzero_col = std::vector<int>(K.cols());
  for (int i = 0; i < K.cols(); ++i)
    nonzero_col[i] = K.innerVector(i).nonZeros();

  int nv = m_state->uv.rows();
  n = K.rows();
  nnz = K.nonZeros();
  K.makeCompressed();

  ia = new int[n + 1];
  ja = new int[nnz];
  res = new double[n];
  int *iia = K.outerIndexPtr();
  int *jja = K.innerIndexPtr();
  double *a = K.valuePtr();
  for (int i = 0; i < n + 1; ++i)
    ia[i] = iia[i] + 1;
  for (int i = 0; i < nnz; ++i)
    ja[i] = jja[i] + 1;

  pardiso_chkmatrix  (&mtype, &n, a, ia, ja, &error);
  if (error != 0) {
    printf("\nERROR in consistency of matrix: %d", error);
    exit(1);
  }
  /* Auxiliary variables. */
  char    *var;
  int      i;

  error = 0;
  solver = 0; /* use sparse direct solver */
  pardisoinit (pt,  &mtype, &solver, iparm, dparm, &error);

  if (error != 0)
  {
    if (error == -10 )
      printf("No license file found \n");
    if (error == -11 )
      printf("License is expired \n");
    if (error == -12 )
      printf("Wrong username or hostname \n");
    exit(1);
  }
  else
    printf("[PARDISO]: License check was successful ... \n");

  var = getenv("OMP_NUM_THREADS");
  if(var != NULL)
    sscanf( var, "%d", &num_procs );
  else {
    printf("Set environment OMP_NUM_THREADS to 1");
    exit(1);
  }
  iparm[2]  = num_procs;

  maxfct = 1;   /* Maximum number of numerical factorizations.  */
  mnum   = 1;         /* Which factorization to use. */
  msglvl = 0;         /* Print statistical information  */
  error  = 0;         /* Initialize error flag */

  // Symbolic solve
  phase = 11;
  pardiso (pt, &maxfct, &mnum, &mtype, &phase,
           &n, a, ia, ja, &idum, &nrhs,
           iparm, &msglvl, &ddum, &ddum, &error, dparm);

  if (error != 0) {
    printf("\nERROR during symbolic factorization: %d", error);
    exit(1);
  }

  // find boundary
  if (flaps) {
    m_state->energy = symmd_p->compute_energy(m_state->V,m_state->F,m_state->uv);
    std::cout << "Energy before flaps: " << m_state->energy << std::endl;

    find_boundary(m_state->V, m_state->F);
    fix_flaps(m_state->uv, m_state->F);

    m_state->energy = symmd_p->compute_energy(m_state->V,m_state->F,m_state->uv);
    std::cout << "Energy after flaps: " << m_state->energy << std::endl;
  }

  // find area in mesh
  igl::doublearea(m_state->V, m_state->F, m_state->M);
  m_state->M /= 2;
}

void AKVFParametrization::linear_solve(Eigen::MatrixXd &grad,
                                       Eigen::MatrixXd &dest_res) {
  int nv = grad.rows();
  // Construct AKVF matrix
  Eigen::SparseMatrix<double> K;
  update_pattern(m_state->uv, m_state->F, K);

  double *a = K.valuePtr();
  phase = 22;  // numerical factorization
  iparm[32] = 1; /* compute determinant */
  pardiso (pt, &maxfct, &mnum, &mtype, &phase,
           &n, a, ia, ja, &idum, &nrhs,
           iparm, &msglvl, &ddum, &ddum, &error, dparm);
  if (error != 0) {
    printf("\nERROR during numerical factorization: %d", error);
    exit(2);
  }

  Eigen::VectorXd aux_grad(2 * nv);
  for (int i = 0; i < n; i += 2) {
    aux_grad(i) = grad(i / 2, 0);
    aux_grad(i + 1) = grad(i / 2, 1);
  }

  // Apply AKVF inverse
  double *b = aux_grad.data();
  phase = 33;
  iparm[7] = 1; /* Max numbers of iterative refinement steps. */
  pardiso (pt, &maxfct, &mnum, &mtype, &phase,
           &n, a, ia, ja, &idum, &nrhs,
           iparm, &msglvl, b, res, &error,  dparm);

  // copy solution into gradient
  for (int i = 0; i < n; i += 2) {
    grad(i / 2, 0) = res[i];
    grad(i / 2, 1) = res[i + 1];
  }

  dest_res = -grad;

  phase = 0;                 /* Release internal memory. */
  pardiso(pt, &maxfct, &mnum, &mtype, &phase,
          &n, &ddum, ia, ja, &idum, &nrhs,
          iparm, &msglvl, &ddum, &ddum, &error,  dparm);
}

bool AKVFParametrization::single_iteration(std::ofstream &fout,
                                           double eps) {
  // Line search
  LinesearchParametrizer linesearchParam(m_state);
  linesearchParam.type = 0;
  linesearchParam.wolfe_c1 = 1e-5;
  linesearchParam.wolfe_c2 = 0.99;

  m_state->timer.start();

  // fix flaps
  fix_flaps(m_state->uv, m_state->F);

  // Compute gradient
  double energy = symmd_p->compute_energy(m_state->V, m_state->F, m_state->uv);
  Eigen::MatrixXd grad(m_state->uv.rows(), m_state->uv.cols());
  symmd_p->compute_grad(m_state->V, m_state->F, m_state->uv, grad);
  double old_energy = energy;

  if (DEBUG) {
    std::cout << "Energy (AKVF): " << energy << std::endl;
    std::cout << "Gradient norm (AKVF): " << grad.norm() << std::endl;
  }

  Eigen::MatrixXd dest_res = -grad;
  linear_solve(grad, dest_res);

  linesearchParam.parametrize(m_state->V, m_state->F, m_state->uv, dest_res, symmd_p, m_state->energy*m_state->mesh_area)/m_state->mesh_area;

  m_state->energy = symmd_p->compute_energy(m_state->V, m_state->F, m_state->uv);

  if (DEBUG) {
    cout << "Initial akvf energy = " << old_energy << endl;
    cout << "Finished akvf iter, time: " << m_state->timer.getElapsedTime() << " (energy = " << m_state->energy << ")" << endl;
    cout << endl;
  }

  total_time += m_state->timer.getElapsedTime();
  fout << m_state->global_local_iters + 1 << ","
       << total_time << ","
       << m_state->energy << std::endl;

  m_state->global_local_iters++;

  return true;
}
