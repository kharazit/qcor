#include "qcor.hpp"

__qpu__ void ansatz(qreg q, double theta) {
  X(q[0]);
  Ry(q[1], theta);
  CX(q[1], q[0]);
}

int main(int argc, char **argv) {
  // Allocate 2 qubits
  auto q = qalloc(2);

  // Create the Deuteron Hamiltonian (Observable)
  auto H = 5.907 - 2.1433 * qcor::X(0) * qcor::X(1) -
           2.1433 * qcor::Y(0) * qcor::Y(1) + .21829 * qcor::Z(0) -
           6.125 * qcor::Z(1);

  // Create the ObjectiveFunction, here we want to run VQE
  // need to provide ansatz and the Observable
  auto objective = qcor::createObjectiveFunction("vqe", ansatz, H);

  // Create the Optimizer
  auto optimizer = qcor::createOptimizer("nlopt");

  // Call taskInitiate, kick off optimization of the give
  // functor dependent on the ObjectiveFunction, async call
  auto handle = qcor::taskInitiate(
      objective, optimizer,
      [&](const std::vector<double> x, std::vector<double> &dx) {
        return (*objective)(q, x[0]);
      },
      1);

  // Go do other work...

  // Query results when ready.
  auto results = qcor::sync(handle);

  // Print the optimal value.
  printf("<H> = %f\n", results.opt_val);
}
