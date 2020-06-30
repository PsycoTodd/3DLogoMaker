#include <igl/boundary_loop.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/triangle/triangulate.h>
#include <igl/writeSTL.h>
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
  beginIndex = beginIndex + edgeMat.rows();
}

int main(int argc, char **argv)
{
  if(argc < 5) {
    std::cout<<"Please call by providing <(I)nterative/(B)atch> <input_Contour_Json> <Output_Stl_Path> <triangulationParameter>" <<std::endl;
    return false;
  }
  bool interactiveMode = argv[1] == "I" ? true : false;
  std::string inputJson = argv[2];
  std::string outputPath = argv[3];
  std::string triangulationSetting = argv[4];

  std::vector<Eigen::MatrixXd> contours;
  std::vector<std::vector<int>> layout;
  
  loadContourData(inputJson, contours, layout);
  int firstId = 0, edgeIdOffset = 0;
  int parentId = -1;
  // Input polygon
  Eigen::MatrixXd V(0, 2);
  Eigen::MatrixXi E(0, 2);
  Eigen::MatrixXd H(0, 2);

  // Triangulated interior
  Eigen::MatrixXd V2;
  Eigen::MatrixXi F2;

  while(firstId < layout.size()) {
    auto out = layout[firstId];
    Eigen::MatrixXd outV = contours[firstId];
    Eigen::MatrixXi outE;
    outE.resize(outV.rows(), 2);
    setEdge(outE, edgeIdOffset);
    V.conservativeResize(V.rows() + outV.rows(), Eigen::NoChange);
    E.conservativeResize(E.rows() + outE.rows(), Eigen::NoChange);
    V.bottomRows(outV.rows()) = outV;
    E.bottomRows(outE.rows()) = outE;
    if(out[2] >= 0) {
        std::queue<int> inQ;
        inQ.push(out[2]);
        while(!inQ.empty()) {
          int child = inQ.front(); inQ.pop();
          auto in = layout[child];
          if(in[0] >= 0) {
            inQ.push(in[0]);
          }
          Eigen::MatrixXd inV = contours[child];
          Eigen::MatrixXi inE;
          inE.resize(inV.rows(), 2);
          setEdge(inE, edgeIdOffset);
          H.conservativeResize(H.rows() + 1, Eigen::NoChange);
          V.conservativeResize(V.rows() + inV.rows(), Eigen::NoChange);
          E.conservativeResize(E.rows() + inE.rows(), Eigen::NoChange);
          H.bottomRows(1) = inV.colwise().mean();
          V.bottomRows(inV.rows()) = inV;
          E.bottomRows(inE.rows()) = inE;
        }
    }
    firstId = out[0];
  }
  igl::triangle::triangulate(V,E,H,triangulationSetting,V2,F2); // u can use a5q for the setting.

  V2.conservativeResize(V2.rows(), 3);
  V2.rightCols(1) = 4 * Eigen::MatrixXd::Ones(V2.rows(), 1);

  Eigen::MatrixXd Vext = V2 - Eigen::RowVector3d(0.0, 0.0, 8.0).replicate(V2.rows(), 1);
  Eigen::MatrixXi Fext = F2;
  int boffset = V2.rows();
  Fext.array() += boffset;
  Fext.col(1).swap(Fext.col(2));

  std::vector<std::vector<int>> loops;
  std::vector<Eigen::RowVector3i> sideTris;
  igl::boundary_loop(F2, loops);

  for(int i=0; i<loops.size(); ++i) {
    std::vector<int>& curLoop = loops[i];
    const int curLoopSize = curLoop.size();
    for(int j=0; j<curLoopSize; ++j) {
      Eigen::RowVector3i t1, t2;
      t1 << curLoop[j]+boffset, curLoop[(j+1)%curLoopSize], curLoop[j];
      t2 << curLoop[j]+boffset, curLoop[(j+1)%curLoopSize]+boffset, curLoop[(j+1)%curLoopSize];
      sideTris.push_back(t1);
      sideTris.push_back(t2);
    }
  }

  Eigen::MatrixXi sideF(sideTris.size(), 3);
  for(int i=0; i<sideF.rows(); ++i) {
    sideF.row(i) = sideTris[i];
  }

  const int F2Rows = F2.rows();

  V2.conservativeResize(V2.rows() + Vext.rows(), Eigen::NoChange);
  F2.conservativeResize(F2.rows() + Fext.rows() + sideF.rows(), Eigen::NoChange);

  V2.bottomRows(Vext.rows()) = Vext;
  F2.middleRows(F2Rows, Fext.rows()) = Fext;
  F2.bottomRows(sideF.rows()) = sideF;


  if(interactiveMode)
  {
    igl::opengl::glfw::Viewer viewer;
    viewer.callback_key_down = 
    [&V2, &F2, &outputPath]
    (igl::opengl::glfw::Viewer& viewer, unsigned char key, int modifier)->bool
    {
      if (key == 'S')
      {
        igl::writeSTL(outputPath, V2, F2, false);
        std::cout << "Saved to " << outputPath <<std::endl;
        return true;
      }
      return true;
    };
    viewer.data().set_mesh(V2, F2);
    viewer.launch();
  }
  else
  {
    igl::writeSTL(outputPath, V2, F2, false);
    std::cout << "Saved to " << outputPath <<std::endl;
  }
  return true;
}
  
