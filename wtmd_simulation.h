// ######################################################################################
// Exposes public methods to perform an L-FTS simulation: equilibrate() and statistics()
// ######################################################################################

#pragma once
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include "diblock.h"
#include "anderson.h"
#include <random>
#include "field_generator.h"
#include "lfts_params.h"
#include "file_IO.h"
#include "langevin.h"
#include "metadynamics.h"

class wtmd_simulation {
    double *w_;                 // Array containing: N*w-(r), N*w+(r), phi-(r), phi+(r)
    diblock  *dbc_;             // Diblock object for calculating phi-(r) and phi+(r)
    anderson *AM_;              // Anderson mixing object to solve for w+(r)
    langevin *Langevin_;        // Langevin object to update w-(r) at each step
    metadynamics *WTMD_;        // Metadynamics object for biasing the Langevin step
    std::mt19937_64 RNG_;       // Random number generator
    lfts_params *P_;            // Object to hold the simulation parameters - automatically updates derived parameters

    int M_;                     // Total number of field mesh points (constant - contained in lfts_params object but copied for tidier code)

    public:
        wtmd_simulation(std::string inputFile, std::string biasFile) {

            // Check that input files exist before proceeding
            std::string s = "";
            if (!file_IO::isValidFile(inputFile))   s += "ERROR => Cannot open the L-FTS input file.\n";
            if (!file_IO::isValidFile(biasFile))    s += "ERROR => Cannot open the WTMD bias file.\n";
            if (s != "") {
                std::cout << s << std::endl;
                exit(1);
            }

            // Read simulation parameters from the input file
            P_ = new lfts_params(inputFile);
            P_->outputParameters();
            M_=P_->M();

            // Set up random number generator
            RNG_.seed(123456789);

            // Allocate memory for field array
            w_ = new double[4*M_];

            // Create a new diblock object
            std::cout << "creating diblock object..." << std::endl;
            dbc_ = new diblock(P_->NA(), P_->NB(), P_->m(), P_->L(), M_, P_->Mk());

            // Create a new anderson mixing object
            std::cout << "creating anderson object..." << std::endl;
            AM_ = new anderson(M_, 10);

            // Set up a langevin object to upstate w-(r) at each step
            std::cout << "creating langevin object..." << std::endl;
            Langevin_ = new langevin(RNG_, P_->sigma(), M_);

            // Set up a metadynamics object
            // Note: WTMD_ must be constructed before the fields are loaded into w_, as the fftw plan setup can destroy w-
            std::cout << "creating metadynamics object..." << std::endl;
            WTMD_ = new metadynamics(w_, P_->m(), P_->L(), P_->n(), P_->Mk(), M_, biasFile);

            // Read w-[r] and w+[r] from the input file
            if (P_->loadType() == 1) { 
                std::cout << "loading input field..." << std::endl;
                file_IO::readArray(w_, inputFile, 2*M_, 3);
            }
            else generate_field(w_, P_->loadType());

            // Perform an initial mix to get phi-(r) and phi+(r) from the input fields
            std::cout << "Initial Anderson mix..." << std::endl;
            AM_->mix(dbc_,200,1e-4,w_);

            // Output initial fields
            saveStdOutputFile("w_0");
            file_IO::saveArray(w_+2*M_, "phi_0", 2*M_);
        }

        // Destructor
        ~wtmd_simulation() {
            delete[] w_;
            delete dbc_;
            delete AM_;
            delete Langevin_;
            delete WTMD_;
            delete P_;
        }

        // Equilibration loop, during which statistics are NOT sampled
        void equilibrate() {
            int it;
            for (it=1; it<=P_->equil_its(); it++) {

                // Perform a Langevin step with symmetrised noise to update w-(r)
                Langevin_->step_wm(w_, RNG_, P_->XbN(), P_->sigma(), P_->dt(), WTMD_->get_fBias(w_));

                // Calculate saddle point value of w+(r), phi-(r) and phi+(r)
                AM_->mix(dbc_,200,1e-4,w_);
                std::cout << "Psi = " << WTMD_->get_Psi(w_) << std::endl;

                // Save to file every save_freq_ steps
                if (it%P_->save_freq()==0) { 
                    saveStdOutputFile("w_eq_" + std::to_string(it));
                    file_IO::saveArray(w_+2*M_, "phi_eq_"+std::to_string(it), 2*M_);
                }
            }
            // Final save to file at end of equilibration period
            saveStdOutputFile("w_eq_" + std::to_string(it-1));
            file_IO::saveArray(w_+2*M_, "phi_eq_"+std::to_string(it-1), 2*M_);
        }

        // Statistics loop, during which statistics are sampled
        void statistics() {
            int it;
            for (it=1; it<=P_->sim_its(); it++) {

                // Perform a Langevin step with symmetrised noise to update w-(r)
                Langevin_->step_wm(w_, RNG_, P_->XbN(), P_->sigma(), P_->dt(), WTMD_->get_fBias(w_));

                // Calculate saddle point value of w+(r), phi-(r) and phi+(r)
                AM_->mix(dbc_,200,1e-4,w_);
                std::cout << "Psi = " << WTMD_->get_Psi(w_) << std::endl;

                // Sample statistics every sample_freq_ steps
                if (it%P_->sample_freq()==0) {
                }

                // Update the bias potential every WTMD_->get_update_freq() steps
                if (it%WTMD_->get_update_freq()==0) {
                    WTMD_->update_bias_field(w_);
                }

                // Save fields to file every save_freq_ steps
                if (it%P_->save_freq()==0) { 
                    saveStdOutputFile("w_st_" + std::to_string(it));
                    file_IO::saveArray(w_+2*M_, "phi_st_"+std::to_string(it), 2*M_);
                    WTMD_->save_bias_std_output("bias_st_" + std::to_string(it));
                }
            }
            // Final save to file at end of equilibration period
            saveStdOutputFile("w_st_" + std::to_string(it-1));
            file_IO::saveArray(w_+2*M_, "phi_st_"+std::to_string(it-1), 2*M_);
            WTMD_->save_bias_std_output("bias_st_" + std::to_string(it-1));
        }

        // Calculate the diblock copolymer Hamiltonian
        double getH() {
            int r;
            double wp_sum=0.0, wm2_sum=0.0;

            // Calculate the natural log of the partition function
            double lnQ = dbc_->calc_concs(w_);

            // Calculate the sums of w+(r) and w-(r)^2
            for (r=0; r<M_; r++) wp_sum += w_[r+M_];
            for (r=0; r<M_; r++) wm2_sum += pow(w_[r],2.0);

            // Return the Hamiltonian
            return -lnQ + (wm2_sum/P_->XbN() - wp_sum)/M_;
        }

    private:

        // Save data in a standard format to be used as in input file
        void saveStdOutputFile(std::string fileName) {
            P_->saveOutputParams(fileName);
            file_IO::saveArray(w_, fileName, 2*M_, true);
        }

        void generate_field(double *w, int loadType) {
            switch (loadType) {
                case 2:
                    field_generator::create_lamellar(w, P_->XbN(), P_->m()); break;
                default:
                    // Create a random field with noise of amplitude XN/2
                    std::uniform_real_distribution<double> uDist(0.0, 1.0);
                    for (int r=0; r<M_; r++) {
                        w[r] = P_->XbN()*(uDist(RNG_)-0.5);
                        w[r+M_] = 0.0;
                    }
                    break;
            }
        }



        
};
