#ifndef RUNTIME_QCOR_HPP_
#define RUNTIME_QCOR_HPP_

#include <IRTransformation.hpp>
#include <functional>
#include <future>
#include <memory>
#include <tuple>

#include "CompositeInstruction.hpp"
#include "Observable.hpp"
#include "Optimizer.hpp"

#include "qalloc"
#include "xacc_internal_compiler.hpp"
#include "PauliOperator.hpp"
#include "FermionOperator.hpp"
#include "ObservableTransform.hpp"
#include "qrt.hpp"


namespace qcor {

using OptFunction = xacc::OptFunction;
using HeterogeneousMap = xacc::HeterogeneousMap;
using Observable = xacc::Observable;
using Optimizer = xacc::Optimizer;
using CompositeInstruction = xacc::CompositeInstruction;
using PauliOperator = xacc::quantum::PauliOperator;
using FermionOperator = xacc::quantum::FermionOperator;

PauliOperator X(int idx){
  return PauliOperator({{idx, "X"}});
}
PauliOperator Y(int idx){
  return PauliOperator({{idx, "Y"}});
}
PauliOperator Z(int idx){
  return PauliOperator({{idx, "Z"}});
}
PauliOperator SP(int idx){
  std::complex<double> imag (0.0, 1.0);
  return X(idx) + imag * Y(idx);
}
PauliOperator SM(int idx){
  std::complex<double> imag (0.0, 1.0);
  return X(idx) - imag * Y(idx);
}
FermionOperator a(int idx){
  std::string s("(1.0, 0) "+std::to_string(idx));
  return FermionOperator(s);
}
FermionOperator adag(int idx){
  std::string s("(1.0, 0) "+std::to_string(idx) +"^");
  return FermionOperator(s);
}

PauliOperator allZs(const int nQubits) {
    auto ret = Z(0);
    for (int i = 1; i < nQubits; i++) {
        ret *= Z(i);
    }
    return ret;
}

//transform FermionOperator to PauliOperator
PauliOperator transform(FermionOperator& obs, std::string transf = "JW");


template<typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
PauliOperator operator+(T coeff, PauliOperator &op){
  return PauliOperator(coeff) + op;
}
template<typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
PauliOperator operator+(PauliOperator &op, T coeff){
  return PauliOperator(coeff) + op;
}
template<typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
PauliOperator operator-(T coeff, PauliOperator &op){
  return -1.0*coeff + op;
}
template<typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
PauliOperator operator-(PauliOperator &op, T coeff){
  return -1.0*coeff + op;
}
template<typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
FermionOperator operator+(T coeff, FermionOperator &op){
  return FermionOperator(coeff) + op;
}
template<typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
FermionOperator operator+(FermionOperator &op, T coeff){
  return FermionOperator(coeff) + op;
}
template<typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
FermionOperator operator-(T coeff, FermionOperator &op){
  return -1.0*coeff + op;
}
template<typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
FermionOperator operator-(FermionOperator &op, T coeff){
  return -1.0*coeff + op;
}

PauliOperator operator+(FermionOperator &&fop, PauliOperator &&pop){
  auto pfop = transform(fop);
  return pfop + pop;
}

PauliOperator operator+( PauliOperator &&pop, FermionOperator &&fop){
  auto pfop = transform(fop);
  return pop + pfop;
}

PauliOperator operator*( PauliOperator &&pop, FermionOperator &&fop){
  auto pfop = transform(fop);
  return pfop*pop;
}

PauliOperator operator*(FermionOperator &&fop,  PauliOperator &&pop){
  auto pfop = transform(fop);
  return pop*pfop;
}


PauliOperator operator+(FermionOperator &fop, PauliOperator &pop){
  auto pfop = transform(fop);
  return pfop + pop;
}

PauliOperator operator+( PauliOperator &pop, FermionOperator &fop){
  auto pfop = transform(fop);
  return pop + pfop;
}

PauliOperator operator*( PauliOperator &pop, FermionOperator &fop){
  auto pfop = transform(fop);
  return pfop*pop;
}

PauliOperator operator*(FermionOperator &fop,  PauliOperator &pop){
  auto pfop = transform(fop);
  return pop*pfop;
}




class ResultsBuffer {
public:
  xacc::internal_compiler::qreg q_buffer;
  double opt_val;
  std::vector<double> opt_params;
};

using Handle = std::future<ResultsBuffer>;
ResultsBuffer sync(Handle &handle) { return handle.get(); }

void set_verbose(bool verbose);
void set_shots(const int shots);

class ObjectiveFunction;

template <typename... Args> class ArgTranslator {
public:
  std::function<std::tuple<Args...>(std::vector<double>)> t;
  ArgTranslator(std::function<std::tuple<Args...>(std::vector<double> x)> &&ts)
      : t(ts) {}

  std::tuple<Args...> operator()(std::vector<double> x) { return t(x); }
};

template <typename... Args>
using TranslationFunctor =
    std::function<std::tuple<Args...>(const std::vector<double>)>;

namespace __internal__ {

template <typename Function, typename Tuple, size_t... I>
auto call(Function f, Tuple t, std::index_sequence<I...>) {
  return f->operator()(std::get<I>(t)...);
}

template <typename Function, typename Tuple> auto call(Function f, Tuple t) {
  static constexpr auto size = std::tuple_size<Tuple>::value;
  return call(f, t, std::make_index_sequence<size>{});
}

// Given a quantum kernel functor / function pointer, create the xacc
// CompositeInstruction representation of it
template <typename QuantumKernel, typename... Args>
std::shared_ptr<CompositeInstruction>
kernel_as_composite_instruction(QuantumKernel &k, Args... args) {
  // #ifdef QCOR_USE_QRT
  quantum::clearProgram();
  // #endif
  // turn off execution
  const auto cached_exec = xacc::internal_compiler::__execute;
  xacc::internal_compiler::__execute = false;
  // Execute to compile, this will store and we can get it
  k(args...);
  // turn execution on
  xacc::internal_compiler::__execute = cached_exec;
  // #ifdef QCOR_USE_QRT
  return quantum::getProgram();
  // #else
  //   return xacc::internal_compiler::getLastCompiled();
  // #endif
}



// Observe the given kernel, and return the expected value
double observe(std::shared_ptr<CompositeInstruction> program,
               std::shared_ptr<Observable> obs,
               xacc::internal_compiler::qreg &q);

// Observe the kernel and return the measured kernels
std::vector<std::shared_ptr<CompositeInstruction>>
observe(std::shared_ptr<Observable> obs,
        std::shared_ptr<CompositeInstruction> program);

// Get the objective function from the service registry
std::shared_ptr<ObjectiveFunction> get_objective(const std::string &type);
std::shared_ptr<xacc::IRTransformation>
get_transformation(const std::string &transform_type);

} // namespace __internal__

template <typename QuantumKernel, typename... Args>
void print_kernel(std::ostream &os, QuantumKernel &kernel, Args... args) {
  os << __internal__::kernel_as_composite_instruction(kernel, args...)
            ->toString();
  // #ifdef QCOR_USE_QRT
  quantum::clearProgram();
  // #endif
}

// The ObjectiveFunction represents a functor-like data structure that
// models a general parameterized scalar function. It is initialized with a
// problem-specific Observable and Quantum Kernel, and exposes a method for
// evaluation, given a list or array of scalar parameters.
// Implementations of this concept are problem-specific, and leverage the
// observe() functionality of the provided Observable to produce one or many
// measured Kernels that are then queued for execution on the available quantum
// co-processor, given the current value of the input parameters. The results of
// these quantum executions are to be used by the ObjectiveFunction to return a
// list of scalar values, representing the evaluation of the ObjectiveFunction
// at the given set of input parameters. Furthermore, the ObjectiveFunction has
// access to a global ResultBuffer that it uses to publish execution results at
// the current input parameters
class ObjectiveFunction : public xacc::Identifiable {
private:
  // This points to provided functor representation
  // of the quantum kernel, used to reconstruct
  // CompositeInstruction in variadic operator()
  void *pointer_to_functor = nullptr;

protected:
  // Pointer to the problem-specific Observable
  Observable* observable;

  // Pointer to the quantum kernel
  std::shared_ptr<CompositeInstruction> kernel;

  // The buffer containing all execution results
  xacc::internal_compiler::qreg qreg;
  bool kernel_is_xacc_composite = false;

  HeterogeneousMap options;

  // To be implemented by subclasses. Subclasses
  // can assume that the kernel has been evaluated
  // at current iterates (or evaluation) of the
  // objective function. I.e. this is called in
  // the variadic operator()(Args... args) method after
  // kernel->updateRuntimeArguments(args...)
  virtual double operator()() = 0;

public:
  // Publicly visible to clients for use in Optimization
  std::vector<double> current_gradient;

  // The Constructor
  ObjectiveFunction() = default;

  // Initialize this ObjectiveFunction with the problem
  // specific observable and CompositeInstruction
  virtual void initialize(Observable* obs,
                          std::shared_ptr<CompositeInstruction> qk) {
    observable = obs;
    kernel = qk;
    kernel_is_xacc_composite = true;
  }

  // Initialize this ObjectiveFunction with the problem
  // specific observable and pointer to quantum functor
  virtual void initialize(Observable* obs, void *qk) {
    observable = obs;
    pointer_to_functor = qk;
  }

  void set_options(HeterogeneousMap &opts) { options = opts; }

  // Set the results buffer
  void set_qreg(xacc::internal_compiler::qreg q) { qreg = q; }
  xacc::internal_compiler::qreg get_qreg() { return qreg; }

  // Evaluate this Objective function at the give parameters.
  // These variadic parameters must mirror the provided
  // quantum kernel
  template <typename... ArgumentTypes>
  double operator()(ArgumentTypes... args) {
    void (*functor)(ArgumentTypes...);
    if (pointer_to_functor) {
      functor =
          reinterpret_cast<void (*)(ArgumentTypes...)>(pointer_to_functor);
    }

    if (!qreg.results()) {
      // this hasn't been set, so set it
      qreg = std::get<0>(std::forward_as_tuple(args...));
    }

    if (kernel_is_xacc_composite) {
      kernel->updateRuntimeArguments(args...);
    } else {
      kernel = __internal__::kernel_as_composite_instruction(functor, args...);
    }
    return operator()();
  }
};

void set_backend(const std::string &backend) {
  xacc::internal_compiler::compiler_InitializeXACC();
  xacc::internal_compiler::setAccelerator(backend.c_str());
}

std::shared_ptr<CompositeInstruction> compile(const std::string &src);

// Public observe function, returns expected value of Observable
template <typename QuantumKernel, typename... Args>
auto observe(QuantumKernel &kernel, std::shared_ptr<Observable> obs,
             Args... args) {
  auto program = __internal__::kernel_as_composite_instruction(kernel, args...);
  return [program, obs](Args... args) {
    // Get the first argument, which should be a qreg
    auto q = std::get<0>(std::forward_as_tuple(args...));

    // Observe the program
    auto programs = obs->observe(program);

    xacc::internal_compiler::execute(q.results(), programs);

    // We want to contract q children buffer
    // exp-val-zs with obs term coeffs
    return q.weighted_sum(obs.get());
  }(args...);
}

template <typename QuantumKernel, typename... Args>
auto observe(QuantumKernel &kernel, Observable &obs,
             Args... args) {
  auto program = __internal__::kernel_as_composite_instruction(kernel, args...);
  return [program, &obs](Args... args) {

    // Get the first argument, which should be a qreg
    auto q = std::get<0>(std::forward_as_tuple(args...));

    // Observe the program
    auto programs = obs.observe(program);

    xacc::internal_compiler::execute(q.results(), programs);

    // We want to contract q children buffer
    // exp-val-zs with obs term coeffs
    return q.weighted_sum(&obs);
  }(args...);
}

// Create the desired Optimizer
std::shared_ptr<xacc::Optimizer>
createOptimizer(const std::string& type, HeterogeneousMap &&options = {});

// Create an observable from a string representation
std::shared_ptr<Observable> createObservable(const std::string& repr);

std::shared_ptr<ObjectiveFunction> createObjectiveFunction(
    const std::string & obj_name, std::shared_ptr<CompositeInstruction> kernel,
    std::shared_ptr<Observable> observable, HeterogeneousMap &&options = {}) 
    {
      std::cout<<"shared_ptr OBSERVABLE CALLED"<<std::endl;
      auto obj_func = qcor::__internal__::get_objective(obj_name);
      obj_func->initialize(observable.get(), kernel);
      obj_func->set_options(options);
  return obj_func;
}

std::shared_ptr<ObjectiveFunction> createObjectiveFunction(
    const std::string & obj_name, std::shared_ptr<CompositeInstruction> kernel,
    Observable& observable, HeterogeneousMap &&options = {}) {
  std::cout<<"OBSERVABLE& CALLED"<<std::endl;
  auto obj_func = qcor::__internal__::get_objective(obj_name);
  obj_func->initialize(&observable, kernel);
  obj_func->set_options(options);
  return obj_func;
}

// Create an Objective Function that makes calls to the
// provided Quantum Kernel, with measurements dictated by
// the provided Observable. Optionally can provide problem-specific
// options map.
template <typename QuantumKernel>
std::shared_ptr<ObjectiveFunction>
createObjectiveFunction(const std::string & obj_name, QuantumKernel &kernel,
                        std::shared_ptr<Observable> observable,
                        HeterogeneousMap &&options = {}) {
  auto obj_func = qcor::__internal__::get_objective(obj_name);
  // We can store this function pointer to a void* on ObjectiveFunction
  // to be converted to CompositeInstruction later
  void *kk = reinterpret_cast<void *>(kernel);
  obj_func->initialize(observable.get(), kk);
  obj_func->set_options(options);
  return obj_func;
}

template <typename QuantumKernel>
std::shared_ptr<ObjectiveFunction>
createObjectiveFunction(const std::string & obj_name, QuantumKernel &kernel,
                        Observable& observable,
                        HeterogeneousMap &&options = {}) {
  auto obj_func = qcor::__internal__::get_objective(obj_name);
  // We can store this function pointer to a void* on ObjectiveFunction
  // to be converted to CompositeInstruction later
  void *kk = reinterpret_cast<void *>(kernel);
  obj_func->initialize(&observable, kk);
  obj_func->set_options(options);
  return obj_func;
}
Handle taskInitiate(std::shared_ptr<ObjectiveFunction> objective,
                    std::shared_ptr<Optimizer> optimizer,
                    std::function<double(const std::vector<double>,
                                         std::vector<double> &)> &&opt_function,
                    const int nParameters);

Handle taskInitiate(std::shared_ptr<ObjectiveFunction> objective,
                    std::shared_ptr<Optimizer> optimizer,
                    qcor::OptFunction &&opt_function);

Handle taskInitiate(std::shared_ptr<ObjectiveFunction> objective,
                    std::shared_ptr<Optimizer> optimizer,
                    qcor::OptFunction &opt_function);
template <typename... Args>
Handle taskInitiate(std::shared_ptr<ObjectiveFunction> objective,
                    std::shared_ptr<Optimizer> optimizer,
                    TranslationFunctor<Args...> translation,
                    const int nParameters) {
  return taskInitiate(
      objective, optimizer,
      [=](const std::vector<double> x, std::vector<double> &dx) {
        auto translated_tuple = translation(x);
        return qcor::__internal__::call(objective, translated_tuple);
      },
      nParameters);
}

// Controlled-Op transform:
// Usage: Controlled::Apply(controlBit, QuantumKernel, Args...)
// where Args... are arguments that will be passed to the kernel.
// Note: we use a class with a static member function to enforce
// that the invocation is Controlled::Apply(...) (with the '::' separator), 
// hence the XASM compiler cannot mistakenly parse this as a XASM call.
class Controlled {
public:
  template <typename FunctorType, typename... ArgumentTypes>
  static void Apply(int ctrlIdx, FunctorType functor, ArgumentTypes... args) {
  __controlledIdx = { ctrlIdx };
  const auto __cached_execute_flag = __execute;
  __execute = false;
  functor(args...);
  __controlledIdx.clear();
  __execute = __cached_execute_flag;
}
};

template <typename QuantumKernel, typename... Args>
std::function<void(Args...)> measure_all(QuantumKernel &kernel, Args... args) {
  return [&](Args... args) {
    auto internal_kernel =
        qcor::__internal__::kernel_as_composite_instruction(kernel, args...);
    auto q = std::get<0>(std::forward_as_tuple(args...));
    auto q_name = q.name();
    auto nq = q.size();
    auto observable = allZs(nq);
    auto observed = observable.observe(internal_kernel)[0];
    auto visitor = std::make_shared<xacc_to_qrt_mapper>(q_name);
    quantum::clearProgram();
    xacc::InstructionIterator iter(observed);
    while (iter.hasNext()) {
      auto next = iter.next();
      if (!next->isComposite()) {
        next->accept(visitor);
      }
    }
    if (xacc::internal_compiler::__execute) {
      ::quantum::submit(q.results());
    }
    return;
  };
}

template <typename QuantumKernel, typename... Args>
std::function<void(Args...)>
apply_transformations(QuantumKernel &kernel,
                      std::vector<std::string> &&transforms, Args... args) {
  auto internal_kernel =
      qcor::__internal__::kernel_as_composite_instruction(kernel, args...);

  for (auto &transform : transforms) {

    auto xacc_transform = qcor::__internal__::get_transformation(transform);
    xacc_transform->apply(internal_kernel, xacc::internal_compiler::get_qpu());
  }

  return [internal_kernel](Args... args) {
    // map back to executable kernel
    quantum::clearProgram();
    auto q = std::get<0>(std::forward_as_tuple(args...));
    auto q_name = q.name();
    auto visitor = std::make_shared<xacc_to_qrt_mapper>(q_name);
    xacc::InstructionIterator iter(internal_kernel);
    while (iter.hasNext()) {
      auto next = iter.next();
      if (!next->isComposite()) {
        next->accept(visitor);
      }
    }
    if (xacc::internal_compiler::__execute) {
      ::quantum::submit(q.results());
    }
  };
}

template <typename QuantumKernel, typename... Args>
const std::size_t n_instructions(QuantumKernel &kernel, Args... args) {
  return qcor::__internal__::kernel_as_composite_instruction(kernel, args...)
      ->nInstructions();
}

template <typename QuantumKernel, typename... Args>
const std::size_t depth(QuantumKernel &kernel, Args... args) {
  return qcor::__internal__::kernel_as_composite_instruction(kernel, args...)
      ->depth();
}
} // namespace qcor

#endif
