#include "qcor.hpp"
#include <iomanip>
using namespace qcor;
__qpu__ void ansatz(qreg q, double theta) {
    X(q[0]);
    X(q[1]);
    Rx(q[0],1.5707);
    H(q[1]);
    H(q[2]);
    H(q[3]);
    CNOT(q[0],q[1]);
    CNOT(q[1],q[2]);
    CNOT(q[2],q[3]);
    Rz(q[3], theta);
    CNOT(q[2],q[3]);
    CNOT(q[1],q[2]);
    CNOT(q[0],q[1]);
    Rx(q[0],-1.5707);
    H(q[1]);
    H(q[2]);
    H(q[3]);
}

int main(int argc, char **argv){
    auto q = qalloc(4);

    auto H =  -0.165606823582 * adag(1)* adag(2) * a(1) * a(2) +
    0.120200490713*adag(1)*adag(0)*a(0)*a(1) - 0.0454063328691*adag(0)*adag(3)*a(1)*a(2) + 
    0.168335986252*adag(2)*a(0)*a(0)*a(2) + 0.0454063328691*adag(1)*adag(2)*a(3)*a(0) + 
    0.168335986252*adag(0)*adag(2)*a(2)*a(0) + 0.165606823582*adag(0)*adag(3)*a(3)*a(0) + 
    -0.0454063328691*adag(3)*adag(0)*a(2)*a(1) - 0.0454063328691*adag(1)*adag(3)*a(0)*a(2) +
    -0.0454063328691*adag(3)*adag(1)*a(2)*a(0) + 0.165606823582*adag(1)*adag(2)*a(2)*a(1) +
    -0.165606823582*adag(0)*adag(3)*a(0)*a(3) - 0.479677813134*adag(3)*a(3) + 
    -0.0454063328691*adag(1)*adag(2)*a(0)*a(3) - 0.174072892497*adag(1)*adag(3)*a(1)*a(3) + 
    -0.0454063328691*adag(0)*adag(2)*a(1)*a(3) + 0.120200490713*adag(0)*adag(1)*a(1)*a(0) + 
    0.0454063328691*adag(0)*adag(1)*a(3)*a(1)+ 0.174072892497*adag(1)*adag(3)*a(3)*a(1) + 
    0.165606823582*adag(2)*adag(1)*a(1)*a(2)  - 0.0454063328691*adag(2)*adag(1)*a(3)*a(0)+ 
    -0.120200490713*adag(2)*adag(3)*a(2)*a(3) + 0.120200490713*adag(2)*adag(3)*a(3)*a(2) + 
    -0.168335986252*adag(0)*adag(2)*a(0)*a(2) + 0.120200490713*adag(3)*adag(2)*a(2)*a(3) + 
    -0.120200490713*adag(3)*adag(2)*a(3)*a(2) + 0.0454063328691*adag(1)*adag(3)*a(2)*a(0) + 
    -1.2488468038*adag(0)*a(0) + 0.0454063328691*adag(3)*adag(1)*a(0)*a(2) + 
    -0.168335986252*adag(2)*adag(0)*a(2)*a(0) + 0.165606823582*adag(3)*adag(0)*a(0)*a(3) + 
    -0.0454063328691*adag(2)*adag(0)*a(3)*a(1) + 0.0454063328691*adag(2)*adag(0)*a(1)*a(3) + 
    -1.2488468038*adag(2)*a(2) + 0.0454063328691*adag(2)*adag(1)*a(0)*a(3) +
    0.174072892497*adag(3)*adag(1)*a(1)*a(3) + -0.479677813134*adag(1)*a(1) + 
    -0.174072892497*adag(3)*adag(1)*a(3)*a(1)  + 0.0454063328691*adag(3)*adag(0)*a(1)*a(2) + 
    -0.165606823582*adag(3)*adag(0)*a(3)*a(0) + 0.0454063328691*adag(0)*adag(3)*a(2)*a(1) + 
    -0.165606823582*adag(2)*adag(1)*a(2)*a(1) + -0.120200490713*adag(0)*adag(1)*a(0)*a(1) + 
    -0.120200490713*adag(1)*adag(0)*a(1)*a(0) + 0.7080240981;


    std::cout<<"OBSERVABLE: "<<std::endl<<std::setprecision(12)<<H.toString()<<std::endl;



    auto objective = qcor::createObjectiveFunction("vqe", ansatz, H);
    std::cout<<typeid(*objective).name()<<std::endl;
    auto optimizer = qcor::createOptimizer("nlopt");

    std::cout<<typeid(H).name() << std::endl;
    qcor::OptFunction f(
      [&](const std::vector<double> &x, std::vector<double> &grad){
        std::cout<< "HERE: "<< (*objective)(q, .59)<<std::endl;
        return(*objective)(q, x[0]);
      },
      10);
    std::cout<<typeid(f).name()<<std::endl;
    auto results = optimizer->optimize(f);
    printf("vqe energy = %f\n", results.first);
    return 0;
}





/*
    auto H =  -0.165606823582 * adag(1)* adag(2) * a(1) * a(2) +
    0.120200490713*adag(1)*adag(0)*a(0)*a(1) - 0.0454063328691*adag(0)*adag(3)*a(1)*a(2) + 
    0.168335986252*adag(2)*a(0)*a(0)*a(2) + 0.0454063328691*adag(1)*adag(2)*a(3)*a(0) + 
    0.168335986252*adag(0)*adag(2)*a(2)*a(0) + 0.165606823582*adag(0)*adag(3)*a(3)*a(0) + 
    -0.0454063328691*adag(3)*adag(0)*a(2)*a(1) - 0.0454063328691*adag(1)*adag(3)*a(0)*a(2) +
    -0.0454063328691*adag(3)*adag(1)*a(2)*a(0) + 0.165606823582*adag(1)*adag(2)*a(2)*a(1) +
    -0.165606823582*adag(0)*adag(3)*a(0)*a(3) - 0.479677813134*adag(3)*a(3) + 
    -0.0454063328691*adag(1)*adag(2)*a(0)*a(3) - 0.174072892497*adag(1)*adag(3)*a(1)*a(3) + 
    -0.0454063328691*adag(0)*adag(2)*a(1)*a(3) + 0.120200490713*adag(0)*adag(1)*a(1)*a(0) + 
    0.0454063328691*adag(0)*adag(1)*a(3)*a(1)+ 0.174072892497*adag(1)*adag(3)*a(3)*a(1) + 
    0.165606823582*adag(2)*adag(1)*a(1)*a(2)  - 0.0454063328691*adag(2)*adag(1)*a(3)*a(0)+ 
    -0.120200490713*adag(2)*adag(3)*a(2)*a(3) + 0.120200490713*adag(2)*adag(3)*a(3)*a(2) + 
    -0.168335986252*adag(0)*adag(2)*a(0)*a(2) + 0.120200490713*adag(3)*adag(2)*a(2)*a(3) + 
    -0.120200490713*adag(3)*adag(2)*a(3)*a(2) + 0.0454063328691*adag(1)*adag(3)*a(2)*a(0) + 
    -1.2488468038*adag(0)*a(0) + 0.0454063328691*adag(3)*adag(1)*a(0)*a(2) + 
    -0.168335986252*adag(2)*adag(0)*a(2)*a(0) + 0.165606823582*adag(3)*adag(0)*a(0)*a(3) + 
    -0.0454063328691*adag(2)*adag(0)*a(3)*a(1) + 0.0454063328691*adag(2)*adag(0)*a(1)*a(3) + 
    -1.2488468038*adag(2)*a(2) + 0.0454063328691*adag(2)*adag(1)*a(0)*a(3) +
    0.174072892497*adag(3)*adag(1)*a(1)*a(3) + -0.479677813134*adag(1)*a(1) + 
    -0.174072892497*adag(3)*adag(1)*a(3)*a(1)  + 0.0454063328691*adag(3)*adag(0)*a(1)*a(2) + 
    -0.165606823582*adag(3)*adag(0)*a(3)*a(0) + 0.0454063328691*adag(0)*adag(3)*a(2)*a(1) + 
    -0.165606823582*adag(2)*adag(1)*a(2)*a(1) + -0.120200490713*adag(0)*adag(1)*a(0)*a(1) + 
    -0.120200490713*adag(1)*adag(0)*a(1)*a(0) + 0.7080240981;

    */