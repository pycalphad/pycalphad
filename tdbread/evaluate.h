// evaluate.h -- header file for Gibbs energy minimization function
// This file is for use outside the optimizer engine

#ifndef INCLUDED_EVALUATE
#define INCLUDED_EVALUATE

class Database;
struct evalconditions;

void evaluate(const Database &DB, const evalconditions &conditions);

#endif