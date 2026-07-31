#ifndef DPHDC_READEXEINPUTS_HPP
#define DPHDC_READEXEINPUTS_HPP

struct Inputs {
    int vector_size = 10000;
    int n_gram = 3;
    int rt_steps = 0;
    float alpha = -1.0f;
    int regen_iterations = 0;
    float regen_ratio = 0.1f;
    bool host = false;
};

Inputs readExeInputs(int argc, char **argv);

#endif //DPHDC_READEXEINPUTS_HPP
