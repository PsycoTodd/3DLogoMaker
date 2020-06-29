#include <igl/opengl/glfw/Viewer.h>
#include <igl/triangle/triangulate.h>
#include "nlohmann/json.hpp"
#include <fstream>
#include <iostream>
#include <queue>

void loadContourData(const std::string& filePath, std::vector<Eigen::MatrixXd>& contours, std::vector<std::vector<int>>& layout)
{
  using json = nlohmann::json;
  std::ifstream file(filePath);
  json js;
  if(file.is_open()) {
    file >> js;
  }
  json hi(js["hierarchy"]);
  for(int i=0; i<hi.size(); ++i) {
    json h = hi[i];
    std::vector<int> ele;
    for(int j=0; j<h.size(); ++j) {
      ele.push_back(h[j]);
    }
    layout.emplace_back(std::move(ele));
  }

  json cots(js["contours"]);
  for(int i=0; i<cots.size(); ++i) {
    json cot = cots[i];
    Eigen::MatrixXd mat(cot.size(), 2);
    for(int j=0; j<cot.size(); ++j) {
      json pt = cot[j];
      mat.row(j) = Eigen::RowVector2d(pt[0], pt[1]);
    }
    contours.emplace_back(std::move(mat));
  }

  return;
}

int setEdge(Eigen::MatrixXi& edgeMat, int& beginIndex) {
  for(int i = 0; i <edgeMat.rows(); ++i) {
    edgeMat.row(i) << beginIndex + i, beginIndex + i + 1;
    if(i == edgeMat.rows() - 1) {
      edgeMat.row(i) << beginIndex + i, beginIndex;
    }
  }
  beginIndex = edgeMat.rows();
}

int main(int argc, char *argv[])
{
  std::vector<Eigen::MatrixXd> contours;
  std::vector<std::vector<int>> layout;
  
  loadContourData("../batchRunner/data.json", contours, layout);
  int firstId = 0, edgeIdOffset = 0;
  int parentId = -1;
  // Input polygon
  Eigen::MatrixXd V;
  Eigen::MatrixXi E;
  Eigen::MatrixXd H;

  // Triangulated interior
  Eigen::MatrixXd V2;
  Eigen::MatrixXi F2;

  while(firstId < layout.size()) {
    auto out = layout[firstId];
    V = contours[firstId];
    E.resize(V.rows(), 2);
    setEdge(E, edgeIdOffset);
    if(out[3] >= 0) {
        std::queue<int> inQ;
        inQ.push(out[3]);
        while(!inQ.empty()) {
          int child = inQ.front(); inQ.pop();
          auto in = layout[child];
          if(in[0] >= 0) {
            inQ.push(in[0]);
          }
          Eigen::MatrixXd inV = contours[child];
          Eigen::MatrixXi inE;
          setEdge(inE, edgeIdOffset);
          H << H, inV.colwise().mean();
          V << V, inV;
          E << E, inE;
        }
    }
    firstId = out[0];
  }
  igl::triangle::triangulate(V,E,H,"a0.5q",V2,F2);

  igl::opengl::glfw::Viewer viewer;
  viewer.data().set_mesh(V2, F2);
  viewer.launch();
}
  
