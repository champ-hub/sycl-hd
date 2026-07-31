#ifndef DPHCL_RESULTSHANDLER_H
#define DPHCL_RESULTSHANDLER_H

#include <chrono>
#include <string>
#include <syclhd.hpp>
#include "readExeInputs.hpp"

#define ACCURATE_TIME

class ResultsHandler {
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_training;
    std::chrono::time_point<std::chrono::high_resolution_clock> finish_training;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_testing;
    std::chrono::time_point<std::chrono::high_resolution_clock> finish_testing;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_gen;
    std::chrono::time_point<std::chrono::high_resolution_clock> finish_gen;
    std::string version = "v" PROJECT_VERSION_CMAKE;

    bool custom_data = false;
    std::string custom_data_text;

    cl::sycl::queue *q = nullptr;

    std::time_t getEndTime();

public:
    bool host = false;
    std::string accelerator;
    std::string model = "undf";
    std::string example_name = "Example";
    float success_rate = -1;
    int vector_size = -1;
    int rt_steps = -1;
    int regen_iterations = -1;
    float alpha = -1;
    float regen_ratio = -1;
    int ngram = -1;

    ResultsHandler(){};

    template<class Mn>
    ResultsHandler(Mn &m){this->autoset(m);};

    ResultsHandler(std::string example_name,Inputs &&inputs);
    ResultsHandler(std::string example_name);

    template<class Tp>
    void autoset(hd::model<Tp> &m){
        this->model = m.getModelName();
        this->accelerator = m.getDeviceName();
        this->vector_size = m.getVectorSize();
        this->q = &m.getQueue();
    }

    void autoset(Inputs &inputs);
    void startTraining();

    void stopTraining();

    void startTesting();

    void stopTesting();
    
    void startGen();

    void stopGen();

    void printToTerminal();

    void printToFile();
    void printToFile(const std::string &path_to_file);

    void setCustomData(const std::string &cdt);
};


#endif //DPHCL_RESULTSHANDLER_H
