#include <igl/boundary_loop.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/triangle/triangulate.h>
#include <igl/png/readPNG.h>
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

Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic>
loadImageRedChannel(const std::string& imgPath) 
{
  Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> r, g, b, a;
  igl::png::readPNG(imgPath, r, g, b, a);
  return r;
}

Eigen::MatrixXd getOnePointOutside2(const Eigen::MatrixXd& contour, 
                                    Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic>& img, 
                                    size_t neighborOffset = 10,
                                    size_t testBlockNum = 20,
                                    size_t trails = 1000)
{
  size_t rcount = contour.rows();
  double x, y;
  Eigen::MatrixXd ret(1, 2);
  ret.row(0) << -1, -1;
  std::srand((unsigned int)time(0)); // set random for sampling the valid hole point.
  for(size_t i = 0; i < trails; ++i)
  {
    // sampleing a random index as the center of two neighbor vectors.
    int id = rand() % rcount;
    Eigen::Vector2d p0 = contour.row(id);
    Eigen::Vector2d p1 = contour.row((id+rcount-neighborOffset%rcount) % rcount);
    Eigen::Vector2d p2 = contour.row((id+neighborOffset%rcount ) % rcount);
    Eigen::Vector2d v1 = p1 - p0;
    Eigen::Vector2d v2 = p2 - p0;
    double w1 = ((double) rand() / (RAND_MAX)) * 0.5;
    double w2 = ((double) rand() / (RAND_MAX)) * 0.5;
    Eigen::Vector2d vs = w1 * v1 + w2 * v2;
    // Make sure betweeen p0 and ps there is no black spot.
    double norm = vs.norm();
    Eigen::Vector2d nor_vs = vs.normalized();
    bool trueOut = true;
    for(int i=1; i<=testBlockNum; ++i) {
      Eigen::Vector2d psi = p0 + (norm / 20 * i) * nor_vs;
      size_t cols = img.cols();
      size_t rows = img.rows();
      std::cout<<"ahahaaha " << psi.transpose()<<std::endl;
      std::cout<<"xxxxx" << cols << " " <<rows <<std::endl;
      std::cout<<"YYYYYYYYYYY" << int(psi.y()) << " " << rows - int(psi.x()) - 1<<std::endl;
      std::cout<<"norm" << norm << " " << nor_vs<<std::endl;
      if(psi.x() >= rows || psi.y() >= cols) { // out of range than the sample would also be out of range.
        trueOut = false;
        break;
      }
      unsigned char var = img(int(psi.x()), cols - int(psi.y()) - 1);
      if(var != 0) {
        trueOut = false;
        break;
      }
    }
    if(trueOut) {
      Eigen::Vector2d ps = p0 + vs;
      ret.row(0) << ps.x(), ps.y();
      return ret;
    }
  }
  return ret;
}

Eigen::MatrixXd getOnePointOutside(const Eigen::MatrixXd& contour, 
                                   Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic>& img, 
                                   size_t trails = 1000)
{
  size_t rcount = contour.rows();
  double x, y;
  Eigen::MatrixXd ret(1, 2);
  ret.row(0) << -1, -1;
  std::srand((unsigned int)time(0)); // set random for sampling the valid hole point.
  for(size_t i = 0; i < trails; ++i)
  {
    Eigen::MatrixXd weight = Eigen::MatrixXd::Random(rcount, 1) + Eigen::MatrixXd::Constant(rcount, 1, 1);
    int sum = weight.sum();
    weight /= sum;
    x = y = 0.0;
    for(int j = 0; j < rcount;  ++j) 
    {
      x += contour.row(j)[0] * weight.row(j)[0];
      y += contour.row(j)[1] * weight.row(j)[0];
    }
    if(abs(x - 501) < 20) {
      y = 800;
    }
    size_t cols = img.cols();
    size_t rows = img.rows();
    unsigned char var = img(int(x), rows - int(y) - 1);
    if(var == 0.f)
    {
      std::cout<<"Todd " << x << ' ' <<y << ' ' << var <<std::endl;
      ret.row(0) << x, y;
      return ret;
    }
    //std::cout<<"Todd fail " << x <<' ' << y <<' ' << var<< std::endl;
    // just get the average.
    //ret = contour.colwise().mean();
    //return ret;
  }
  
  return ret;
}

int main(int argc, char **argv)
{
  if(argc < 6) {
    std::cout<<"Please call by providing <(I)nterative/(B)atch> <input_Contour_Json> <Output_Stl_Path> <triangulationParameter>" <<std::endl;
    return false;
  }
  std::string modeStr = argv[1];
  bool interactiveMode = (modeStr == "I" ? true : false);
  std::string inputJson = argv[2];
  std::string outputPath = argv[3];
  std::string triangulationSetting = argv[4];
  std::string imagePath = argv[5];

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

  Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> red = 
    loadImageRedChannel(imagePath);

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
          H.bottomRows(1) = getOnePointOutside2(inV, red);
          if(H.row(0)[0] < 0) {
            H.bottomRows(1) = inV.colwise().mean();
            std::cout << "We cannot find the outside range in 1000 iteration, use fallback location." << std::endl;
          }
          //H.bottomRows(1) = 0.35 * inV.row(0) + 0.75 * inV.colwise().mean();
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
  
