#include "ResultsHandler.hpp"

#include <iostream>
#include <fstream>
#include <unistd.h>

#define NOW() std::chrono::high_resolution_clock::now();

constexpr void wait(cl::sycl::queue *q){
#ifdef ACCURATE_TIME
    if (q != nullptr)
        q->wait();
#endif
return;
}

bool file_exists (const std::string& name) {
    return ( access( name.c_str(), F_OK ) != -1 );
}

ResultsHandler::ResultsHandler(std::string example_name): example_name(example_name){}

ResultsHandler::ResultsHandler(std::string example_name,Inputs &&inputs):
    example_name(example_name)
{
    this->autoset(inputs);
}

void ResultsHandler::autoset(Inputs &inputs){
    this->vector_size = inputs.vector_size;
    this->rt_steps = inputs.rt_steps;
    this->alpha = inputs.alpha;
    this->regen_iterations = inputs.regen_iterations;
    this->regen_ratio = inputs.regen_ratio;
    this->host = inputs.host;
    this->ngram = inputs.n_gram;
}

void ResultsHandler::startTraining() {
    wait(this->q);
    this->start_training = NOW()
}

void ResultsHandler::stopTraining() {
    wait(this->q);
    this->finish_training = NOW()
}

void ResultsHandler::startTesting() {
    wait(this->q);
    this->start_testing = NOW()
}

void ResultsHandler::stopTesting() {
    wait(this->q);
    this->finish_testing = NOW()
}

void ResultsHandler::startGen() {
    wait(this->q);
    this->start_gen = NOW()
}

void ResultsHandler::stopGen() {
    wait(this->q);
    this->finish_gen = NOW()
}

void ResultsHandler::printToTerminal() {
    std::time_t end_time = this->getEndTime();

    std::cout << "\n\n\nResults for " << this->example_name <<"\n";
    std::cout << PROJECT_NAME_CMAKE " " << this->version << "\n";
    std::cout << "Finished at: " << std::ctime(&end_time);
    std::cout << "Accelerator: " << this->accelerator << "\n";
    std::cout << "Success Rate: " << this->success_rate << "%\n";
    std::cout << "HV Generation Time: " << (float) std::chrono::duration_cast<std::chrono::microseconds>(
            this->finish_gen - this->start_gen).count() / 1000000 << "s\n";
    std::cout << "Training Time: " << (float) std::chrono::duration_cast<std::chrono::microseconds>(
            this->finish_training - this->start_training).count() / 1000000 << "s\n";
    std::cout << "Testing Time: " << (float) std::chrono::duration_cast<std::chrono::microseconds>(
            this->finish_testing - this->start_testing).count() / 1000000 << "s\n";
    std::cout << "Model: " << this->model << "\n";
    std::cout << "Vector Size: " << this->vector_size << "\n";
    if (this->rt_steps > 0){
        std::cout << "Retraining steps: " << this->rt_steps << "\n";
        std::cout << "Alpha: " << this->alpha << "\n";
    }
    if (this->regen_iterations > 0){
        std::cout << "Regen Iterations: " << this->regen_iterations << "\n";
        std::cout << "Regen Ratio: " << this->regen_ratio << "\n";
    }
    std::cout << "\n\n";
}
void ResultsHandler::printToFile() {
    this->printToFile(PROJECT_PATH_CMAKE "/results/session/results.csv");
}

void ResultsHandler::printToFile(const std::string &path_to_file) {

    std::time_t end_time = this->getEndTime();
    std::string end_time_string = std::ctime(&end_time);
    end_time_string.pop_back();

    bool fe = file_exists(path_to_file);
    std::ofstream write_file;
    
    write_file.open(path_to_file, std::ios_base::app);

    if(!fe){
        write_file << "Example,Model,Version,End Time,Accelerator,Accuracy,Generation Time,Training Time,Testing Time,Dimensions,Retrain Steps,Alpha,Regen Iterations,Regen Ratio";
        if (this->custom_data) write_file << ",Custom Data";
        write_file << std::endl;
    }

    write_file << this->example_name << ",";
    write_file << this->model << ",";
    write_file << this->version << ",";
    write_file << end_time_string << ",";
    write_file << this->accelerator << ",";
    write_file << this->success_rate << ",";
    write_file << (float) std::chrono::duration_cast<std::chrono::microseconds>(
            this->finish_gen - this->start_gen).count() / 1000000 << ",";
    write_file << (float) std::chrono::duration_cast<std::chrono::microseconds>(
            this->finish_training - this->start_training).count() / 1000000 << ",";
    write_file << (float) std::chrono::duration_cast<std::chrono::microseconds>(
            this->finish_testing - this->start_testing).count() / 1000000 << ",";
    write_file << this->vector_size << ",";
    write_file << this->rt_steps << ",";
    write_file << this->alpha << ",";
    write_file << this->regen_iterations << ",";
    write_file << this->regen_ratio;
    
    if (this->custom_data) write_file << "," << this->custom_data_text; 

    write_file << std::endl;
    write_file.close();
}

std::time_t ResultsHandler::getEndTime() {
    std::time_t end_time;
    if ((this->finish_testing - this->start_testing).count() == 0) {
        end_time = std::chrono::system_clock::to_time_t(this->finish_training);
    } else {
        end_time = std::chrono::system_clock::to_time_t(this->finish_testing);
    }

    return end_time;
}


void ResultsHandler::setCustomData(const std::string &cdt){
    this->custom_data = true;
    this->custom_data_text = cdt;
}