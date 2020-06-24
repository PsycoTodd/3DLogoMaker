#include <igl/opengl/glfw/Viewer.h>
#include "nlohmann/json.hpp"
#include <fstream>
#include <iostream>

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

int main(int argc, char *argv[])
{
  std::vector<Eigen::MatrixXd> contour;
  std::vector<std::vector<int>> layout;
  
  loadContourData("../batchRunner/data.json", contour, layout);
  igl::opengl::glfw::Viewer viewer;
  viewer.launch();
}
  
