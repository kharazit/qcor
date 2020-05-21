#ifndef RUNTIME_QCOR_QRT_HPP_
#define RUNTIME_QCOR_QRT_HPP_

#include "qalloc.hpp"
#include <CompositeInstruction.hpp>
#include <memory>

using namespace xacc::internal_compiler;

namespace xacc {
class AcceleratorBuffer;
class CompositeInstruction;
class IRProvider;
class Observable;
} // namespace xacc

namespace quantum {

extern std::shared_ptr<xacc::CompositeInstruction> program;
extern std::shared_ptr<xacc::IRProvider> provider;

void initialize(const std::string qpu_name, const std::string kernel_name);
void set_shots(int shots);
void one_qubit_inst(const std::string &name, const qubit &qidx,
                    std::vector<double> parameters = {});

void h(const qubit &qidx);
void x(const qubit &qidx);

void rx(const qubit &qidx, const double theta);
void ry(const qubit &qidx, const double theta);
void rz(const qubit &qidx, const double theta);

void mz(const qubit &qidx);

void cnot(const qubit &src_idx, const qubit &tgt_idx);

void exp(qreg q, const double theta, xacc::Observable * H);
void exp(qreg q, const double theta, std::shared_ptr<xacc::Observable> H);

void submit(xacc::AcceleratorBuffer *buffer);
void submit(xacc::AcceleratorBuffer **buffers, const int nBuffers);

std::shared_ptr<xacc::CompositeInstruction> getProgram();

} // namespace quantum

#endif