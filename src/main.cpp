#include <iostream>

#include "Param_State.h"
#include "GlobalLocalParametrization.h"
#include "AKVFParametrization.h"
#include "eigen_stl_utils.h"
#include "parametrization_utils.h"
#include "StateManager.h"

#include "igl/components.h"
#include "igl/writeOBJ.h"
#include "igl/Timer.h"

#include <igl/serialize.h>
#include <igl/read_triangle_mesh.h>

#include <stdlib.h>

#include <string>
#include <vector>

using namespace std;

void read_mesh(const std::string& mesh_file, Param_State& state);
void check_mesh_for_issues(Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::VectorXd& areas);

const int ITER_NUM = 100;
const int DEBUG = 0;

int main(int argc, char *argv[]) {
  if (argc < 2) {
      cerr << "Syntax: " << argv[0] << " <input mesh>" << std::endl;
      return -1;
  }
  const string input_mesh = argv[1];
  size_t lastindex = input_mesh.find_last_of(".");
  const string rawname = input_mesh.substr(0, lastindex);

  const string output_mesh2 = "results/" + rawname + "-AKVF.obj";
  ofstream output2("results/AKVF-results.txt");

  cout << "Parameterizing mesh " << input_mesh << endl;
  Param_State state2;
  read_mesh(input_mesh, state2);

  std::cout << "Vertices: " << state2.V.rows() << std::endl
            << "Faces: " << state2.F.rows() << std::endl;

  #ifndef USE_PARDISO
  cout << "Warning! Unoptimized version without Pardiso Solver. The algorithm will be significantly slower!" << endl;
  #endif
  StateManager state_manager;
  AKVFParametrization param_AKVF(state_manager, &state2);

  bool flaps = true;
  param_AKVF.init_parametrization(output2, false);
  cout << "initialized parametrization" << endl;

  for (int i = 0; i < ITER_NUM; ++i) {
    std::cout << "iteration " << i + 1 << endl;
    param_AKVF.single_iteration(output2);
  }

  igl::writeOBJ(output_mesh2, state2.V, state2.F, Eigen::MatrixXd(), Eigen::MatrixXi(), state2.uv, state2.F);

  return 0;
}

void read_mesh(const std::string& mesh_file, Param_State& state) {
  memset( &state, 0, sizeof( Param_State ) );
  state.method = Param_State::GLOBAL_ARAP_IRLS;
  state.flips_linesearch = true;
  state.update_all_energies = false;
  state.proximal_p = 0.0001;


  cout << "\tReading mesh object" << endl;
  igl::read_triangle_mesh(mesh_file, state.V, state.F);
  state.v_num = state.V.rows();
  state.f_num = state.F.rows();

  // set uv coords scale
  igl::doublearea(state.V,state.F, state.M); state.M /= 2.;

  state.global_local_energy = Param_State::SYMMETRIC_DIRICHLET;
  state.cnt_flips = false;
  check_mesh_for_issues(state.V,state.F, state.M);
  cout << "\tMesh is valid!" << endl;

  state.mesh_area = state.M.sum();
  state.V /= sqrt(state.mesh_area);
  state.mesh_area = 1;
}

void check_mesh_for_issues(Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::VectorXd& areas) {

  Eigen::SparseMatrix<double> A;
  igl::adjacency_matrix(F,A);

  Eigen::MatrixXi C, Ci;
  igl::components(A, C, Ci);
  //cout << "#Connected_Components = " << Ci.rows() << endl;
  //cout << "is edge manifold = " << igl::is_edge_manifold(V,F) << endl;
  int connected_components = Ci.rows();
  if (connected_components!=1) {
    cout << "Error! Input has multiple connected components" << endl; exit(1);
  }
  int euler_char = get_euler_char(V, F);
  if (!euler_char) {
    cout << "Error! Input does not have a disk topology, it's euler char is " << euler_char << endl; exit(1);
  }
  bool is_edge_manifold = igl::is_edge_manifold(V, F);
  if (!is_edge_manifold) {
    cout << "Error! Input is not an edge manifold" << endl; exit(1);
  }
  const double eps = 1e-14;
  for (int i = 0; i < areas.rows(); i++) {
    if (areas(i) < eps) {
      cout << "Error! Input has zero area faces" << endl; exit(1);
    }
  }
}
