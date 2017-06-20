#include "GlobalLocalParametrization.h"

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

const int DEBUG = 1;

#undef NDEBUG
#include <assert.h>
#define NDEBUG

using namespace std;

GlobalLocalParametrization::GlobalLocalParametrization(StateManager& state_manager, Param_State* m_state) :
      ParametrizationAlgorithm(state_manager, m_state),
      WArap_p(NULL)
  {
  assert (m_state->F.cols() == 3);

  WArap_p = new LocalWeightedArapParametrizer(m_state);
}

void GlobalLocalParametrization::init_parametrization(std::ofstream &fout) {
  WArap_p->pre_calc();
  dirichlet_on_circle(m_state->V,m_state->F,m_state->uv);
  if (count_flips(m_state->V,m_state->F,m_state->uv) > 0) {
    tutte_on_circle(m_state->V,m_state->F,m_state->uv);
  }
  m_state->energy = WArap_p->compute_energy(m_state->V, m_state->F, m_state->uv)/m_state->mesh_area;

  total_time = 0;
  fout << "iters,time,energy" << std::endl;
  fout << 0 << "," << 0 << "," << m_state->energy << std::endl;

  // find_boundary(m_state->V, m_state->F);
}

bool GlobalLocalParametrization::single_iteration(std::ofstream &fout) {
  bool flag = single_line_search_arap(fout);
  m_state->global_local_iters++;
  return flag;
}

void GlobalLocalParametrization::get_linesearch_params(Eigen::MatrixXd& dest_res,
                                                        Energy** param_energy) {
  dest_res = m_state->uv;
  WArap_p->parametrize(m_state->V, m_state->F, m_state->b, m_state->bc, dest_res);
  *param_energy = WArap_p;
}

bool GlobalLocalParametrization::single_line_search_arap(std::ofstream &fout) {
  // weighted arap for riemannian metric
  LinesearchParametrizer linesearchParam(m_state);
  Eigen::MatrixXd dest_res;
  Energy* param_energy = NULL;

  m_state->timer.start();

  get_linesearch_params(dest_res, &param_energy);
  dest_res = dest_res - m_state->uv;

  Eigen::MatrixXd old_uv = m_state->uv;
  double old_energy = m_state->energy;

  m_state->energy = linesearchParam.parametrize(m_state->V,m_state->F, m_state->uv, dest_res, param_energy, m_state->energy*m_state->mesh_area)/m_state->mesh_area;

  if (DEBUG) {
    cout << "Initial SLIM energy: " << old_energy << std::endl;
    cout << "Finished SLIM iter, time: " << m_state->timer.getElapsedTime() << "(energy = " << m_state->energy << ")" << endl << endl;
  }

  total_time += m_state->timer.getElapsedTime();
  fout << m_state->global_local_iters + 1 << ","
       << total_time << ","
       << m_state->energy << std::endl;

  return true;
}
