#ifndef PARAMETRIZATION_ALGORITHM_H
#define PARAMETRIZATION_ALGORITHM_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <map>
#include <set>
#include <vector>

#ifdef HAS_GUI
#include <igl/viewer/Viewer.h>
#endif

#include "StateManager.h"

class ParametrizationAlgorithm {

 public:
  ParametrizationAlgorithm(StateManager& state_manager, Param_State* m_state) : m_stateManager(state_manager),
                                                                                m_state(m_state) {}

  StateManager& m_stateManager;
  Param_State* m_state;

  void find_boundary(const Eigen::MatrixXd &V,
                     const Eigen::MatrixXi &F) {
    int nt = F.rows();
    int nv = V.rows();

    std::vector<std::pair<int, int>> edges;
    std::set<int> vertices;
    for (int t = 0; t < nt; ++t) {
      std::vector<int> edge {F(t, 0), F(t, 1), F(t, 2)};
      std::sort(edge.begin(), edge.end());

      edges.push_back(std::make_pair(edge[0], edge[1]));
      edges.push_back(std::make_pair(edge[0], edge[2]));
      edges.push_back(std::make_pair(edge[1], edge[2]));

      vertices.insert(F(t, 0));
      vertices.insert(F(t, 1));
      vertices.insert(F(t, 2));
    }
    std::sort(edges.begin(), edges.end());

    std::map<std::pair<int, int>, int> count_edges;
    std::vector<std::pair<int, int>> boundary_edges;
    std::vector<int> boundary_vertices;
    for (std::size_t i = 0; i < edges.size(); ++i) {
      count_edges[edges[i]]++;
    }
    for (auto &kv : count_edges) {
      if (kv.second == 1) {
        boundary_edges.push_back(kv.first);
        boundary_vertices.push_back(kv.first.first);
        boundary_vertices.push_back(kv.first.second);
      }
    }
    std::sort(boundary_vertices.begin(), boundary_vertices.end());
    auto vit = std::unique(boundary_vertices.begin(), boundary_vertices.end());
    boundary_vertices.resize(std::distance(boundary_vertices.begin(), vit));

    auto eit = std::unique(edges.begin(), edges.end());
    edges.resize(std::distance(edges.begin(), eit));

    std::vector<std::vector<int>> neighbors(vertices.size());
    for (auto &e : edges) {
      neighbors[e.first].push_back(e.second);
      neighbors[e.second].push_back(e.first);
    }
    for (auto &v : boundary_vertices) {
      if (neighbors[v].size() == 2) {
        flap_vertices.push_back(v);
        flap_neighbors.push_back(std::make_pair(neighbors[v][0], neighbors[v][1]));
      }
    }

    flap_paral = Eigen::VectorXd(flap_neighbors.size());
    flap_ortho = Eigen::VectorXd(flap_neighbors.size());
    flap_oppos_edges = Eigen::MatrixXd(flap_neighbors.size(), 3);
    flap_oppos_leng = Eigen::VectorXd(flap_neighbors.size());
    flap_ortho_edges = Eigen::MatrixXd(flap_neighbors.size(), 3);
    for (std::size_t i = 0; i < flap_neighbors.size(); ++i) {
      int x, y, z;
      x = flap_neighbors[i].first;
      y = flap_neighbors[i].second;
      z = flap_vertices[i];

      flap_oppos_edges.row(i) = V.row(y) - V.row(x);
      flap_oppos_leng(i) = flap_oppos_edges.row(i).norm();
      flap_oppos_edges.row(i) /= flap_oppos_edges.row(i).norm();
      flap_paral(i) = (V.row(z) - V.row(x)).dot(flap_oppos_edges.row(i));

      flap_ortho_edges.row(i) = V.row(z) - V.row(x);
      double dprod = flap_ortho_edges.row(i).dot(flap_oppos_edges.row(i));
      flap_ortho_edges.row(i) -= dprod * flap_oppos_edges.row(i);
      flap_ortho_edges.row(i) /= flap_ortho_edges.row(i).norm();
      flap_ortho(i) = (V.row(z) - V.row(x)).dot(flap_ortho_edges.row(i));
    }
  }

  void fix_flaps(Eigen::MatrixXd &V,
                 const Eigen::MatrixXi &F) {
    Eigen::MatrixXd new_V(flap_neighbors.size(), 2);
    for (std::size_t i = 0; i < flap_neighbors.size(); ++i) {
      int x, y, z;
      x = flap_neighbors[i].first;
      y = flap_neighbors[i].second;
      z = flap_vertices[i];

      Eigen::VectorXd cur_oppos_edge = V.row(y) - V.row(x);
      double cur_oppos_leng = cur_oppos_edge.norm();
      cur_oppos_edge /= cur_oppos_leng;
      Eigen::VectorXd cur_paral =
        (cur_oppos_leng / flap_oppos_leng(i)) * flap_paral(i) * cur_oppos_edge;

      Eigen::VectorXd cur_ortho_edge = V.row(z) - V.row(x);
      double dprod = cur_ortho_edge.dot(cur_oppos_edge);
      cur_ortho_edge -= dprod * cur_oppos_edge;
      cur_ortho_edge /= cur_ortho_edge.norm();
      Eigen::VectorXd cur_ortho = cur_ortho_edge * flap_ortho(i);

      new_V.row(i) = cur_ortho + cur_paral;
      new_V.row(i) += V.row(x);
    }

    for (int i = 0; i < flap_neighbors.size(); ++i) {
      int z = flap_vertices[i];
      V.row(z) = new_V.row(i);
    }
  }

  // Flap variables
  std::vector<int> flap_vertices;
  std::vector<std::pair<int, int>> flap_neighbors;

  Eigen::MatrixXd flap_oppos_edges;
  Eigen::VectorXd flap_oppos_leng;
  Eigen::VectorXd flap_paral;
  Eigen::MatrixXd flap_ortho_edges;
  Eigen::VectorXd flap_ortho;
};

#endif // PARAMETRIZATION_ALGORITHM_H
