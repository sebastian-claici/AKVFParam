#ifndef GLOBAL_LOCAL_PARAMETRIZATION_H
#define GLOBAL_LOCAL_PARAMETRIZATION_H

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <string>


#include "LocalWeightedArapParametrizer.h"
#include "ParametrizationAlgorithm.h"

#include <igl/jet.h>
#include <igl/readOBJ.h>
#include <igl/facet_components.h>
#include <igl/slice.h>
#ifdef HAS_GUI
#include <igl/viewer/Viewer.h>
#endif

class GlobalLocalParametrization : public ParametrizationAlgorithm {

public:

  GlobalLocalParametrization(StateManager& state_manager, Param_State* m_state);

  void init_parametrization(std::ofstream &fout);
  bool single_iteration(std::ofstream &fout);

private:
  bool single_line_search_arap(std::ofstream &fout);
  void get_linesearch_params(Eigen::MatrixXd& dest_res, Energy** param_energy);

  LocalWeightedArapParametrizer* WArap_p;

  // time keeping
  double total_time = 0;
};

#endif // GLOBAL_LOCAL_PARAMETRIZATION_H
