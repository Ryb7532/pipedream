#include <iostream>
#include <fstream>
#include <iomanip>

using namespace std;

int main() {
  string label, lbra, rank, rbra, units;
  double time, total = 0.000;

  ifstream ifs("./tmp.txt");
  while (ifs >> label >> lbra >> rank >> rbra >> time >> units) {
    total += time;
  }

  cout << label << ": " << fixed << setprecision(3) << total << " " << units << endl;
  return 0;
}
