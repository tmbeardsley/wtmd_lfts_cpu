// #######################################################################################
// Provides public methods to update the bias potential and perform a modified (biased)
// Langevin step
// #######################################################################################

#pragma once
#include <stdlib.h>
#include <math.h>
#include <fstream>
#include <limits>
#include <iomanip>
#include <complex>
#include <fftw3.h>
#include "wtmd_params.h"

class metadynamics {

    // non-changing arrays (set in calc_wt_fk(...))
    int *wt_;                           // Weighting of contribution from wavevector k
    double *fk_;                        // Weighting function to reduce the contribution of large wavevectors in the order parameter

    // get_Psi()
    fftw_plan   wr_to_wk_;              // Plan handle for transforming w-(r) -> w-(k) with fftw

    // get_fBias()
    fftw_plan   dPsi_wk_to_dPsi_wr_;    // Plan handle for transforming (DPsi/Dwk) -> (DPsi/Dwr) with fftw

    // update_bias_field()
    double *u_;                         // Bias potential, U(Psi)
    double *up_;                        // Derivative of the bias potential, (DU(Psi)/DPsi)
    double *I0_;                        // Array to which Gaussians are added for computing <w-(r)^2>
    double *I1_;                        // Array to which Gaussians scaled by w-(r)^2 are added for computing <w-(r)^2>

    // langevin()
    double *fBias_;                     // The biasing force used in the modified Langevin step

    // get_fBias()
    std::complex<double> *dPsi_dwk_;    // (DPsi/Dwk)
    double *dPsi_dwr_;                  // (DPsi/Dwr)

    // get_fBias(), get_Psi()
    std::complex<double> *wk_;          // Composition field in reciprocal space, w-(k)

    bool nonZeroBias_;                  // Indicates whether there is a non-zero bias potential (for efficiency in calculating fBias_[])
    wtmd_params *B_;                    // Object to deal with bias-related parameters read from file

    // Simulation constants derived from the input file (see lfts_params.h for details)
    int M_;
    int Mk_;
    double n_;


    public:
        metadynamics(double *w, int *m, double *L, double CV, int Mk, int M, std::string biasFile) {
            Mk_ = Mk;
            M_ = M;
            n_ = CV;

            // Indicate that there is currently no bias potential
            nonZeroBias_ = false;

            // Read first line of the bias input file
            B_ = new wtmd_params(biasFile);
            B_->printInputParams();

            // Allocate memory for lookup arrays, f(k) and wt(k) and calculate them
            fk_ = new double[Mk_];
            wt_ = new int[Mk_];
            calc_wt_fk(wt_, fk_, m, L);

            // langevin() - Allocate memory for the biasing force and set to zero
            fBias_ = new double[M_];
            for (int r=0; r<M_; r++) fBias_[r] = 0.0;

            // get_fBias() - Set up the fftw plan and allocate memory for derivatives
            dPsi_dwk_ = new std::complex<double>[Mk_];
            dPsi_dwr_ = new double[M_];
            dPsi_wk_to_dPsi_wr_ = fftw_plan_dft_c2r(3, m, reinterpret_cast<fftw_complex*>(dPsi_dwk_), dPsi_dwr_, FFTW_PATIENT);

            // get_fBias(), get_Psi() - Allocate memory for w(k)
            wk_ = new std::complex<double>[Mk_];

            // get_Psi() - Set up fftw plan to transform w-(r) to reciprocal space.
            wr_to_wk_ = fftw_plan_dft_r2c(3, m, w, reinterpret_cast<fftw_complex*>(wk_), FFTW_PATIENT);

            // update_bias() - Allocate memory and pointers for bias-related arrays and set to zero
            int mPsi = B_->mPsi();
            u_ = new double[4*mPsi];
            for (int i=0; i<4*mPsi; i++) u_[i] = 0.0;
            up_ = u_ + mPsi;
            I0_ = u_ + 2*mPsi;
            I1_ = u_ + 3*mPsi;

            // Read the bias field if the flag was set in the input file
            if (B_->read_bias() != 0) read_Bias_Fields(biasFile);
        }

        // Destructor
        ~metadynamics() {
            delete[] fk_;
            delete[] wt_;
            delete[] fBias_;
            delete[] dPsi_dwk_;
            delete[] dPsi_dwr_;
            delete[] wk_;
            delete[] u_;
            fftw_destroy_plan(dPsi_wk_to_dPsi_wr_);
            fftw_destroy_plan(wr_to_wk_);
            delete B_;
        }

        // Calculate the order parameter 
        double get_Psi(double *w)
        {
            double Psi = 0.0;

            // Fourier transform w-(r) to get w-(k) and do calculation
            fftw_execute(wr_to_wk_);
            for (int k=0; k<Mk_; k++) Psi += pow(std::norm(wk_[k]), 0.5*B_->ell())*fk_[k]*wt_[k]/M_;
            Psi = pow(Psi/M_, 1.0/B_->ell());

            return Psi;
        }


        // Update u(Psi), up(Psi), I0(Psi) and I1(Psi) - function overloaded so psi doesn't have to be recalculated if already known
        void update_bias_field(double *w) { update_bias_field(get_Psi(w), w); }
        void update_bias_field(double Psi_hat, double *w)
        {
            double Psi, X, A, G;

            // Bias field exists as the field has been updated
            nonZeroBias_ = true;

            // Calculate current value of w-^2
            double w2_hat = 0.0;
            for (int r=0; r<M_; r++) w2_hat += w[r]*w[r];
            w2_hat /= M_;

            // Update the bias fields
            for (int i=0; i < B_->mPsi(); i++) {
                Psi = B_->Psi_min() + i*B_->dPsi();
                X = (Psi_hat-Psi)/(B_->sigma_Psi());
                A = exp(-n_*u_[i]/(B_->DT()))/n_;
                G = exp(-0.5*X*X);
                u_[i]  += A*G;
                up_[i] += ( X/(B_->sigma_Psi()) - n_*up_[i]/(B_->DT()) )*A*G;
                I0_[i] += G;
                I1_[i] += w2_hat*G;
            }
        }

        // Make update_freq publicly accessible for wtmd_simulation.h
        int get_update_freq() {
            return B_->update_freq();
        }

        // Calculate the biasing force for the modified Langevin step.
        double* get_fBias(double *w)
        {
            // fBias_[]=0 if there is no bias potential, so don't waste resources computing it
            if (!nonZeroBias_) return fBias_;

            int    i, k;
            double Psi, x, up_hat;

            // Start calculating current value of DU(Psi)/DPsi
            Psi = get_Psi(w);
            x = (Psi - B_->Psi_min()) / B_->dPsi();
            i = floor(x);
            if (i < 0) {printf("Error: Psi = %lf < Psi_min\n",Psi); exit(1);}
            if (i >= B_->mPsi()) {printf("Error: Psi = %lf > Psi_max\n",Psi); exit(1); }
            x = x-i;

            // Linear interpolation of the (DU(Psi)/DPsi) due to discrete mesh
            up_hat = (1.0-x)*up_[i] + x*up_[i+1];

            // Calculate derivative of order parameter with respect to wk.
            // Note: wk_ was evaluated in the above call to get_Psi(w)
            for (k=0; k<Mk_; k++) dPsi_dwk_[k] = pow(std::norm(wk_[k]),B_->ell()/2.0-1.0) * pow(Psi, 1.0-B_->ell())*wk_[k]*fk_[k];

            // Calculate derivative of order parameter with respect to w
            fftw_execute(dPsi_wk_to_dPsi_wr_);

            // Multiply array elements by a constant
            for (i=0; i<M_; i++) fBias_[i] = dPsi_dwr_[i]*up_hat/M_;

            return fBias_;
        }

        // Save standard bias output file
        void save_bias_std_output(std::string fileName) {
            B_->saveBiasParams(fileName);
            save_bias_fields(fileName, true);
        }

        // Save the bias fields to file in the same format as the bias input file
        void save_bias_fields(std::string fileName, bool append=false) 
        {
            std::ofstream outstream;
            if (append) outstream.open(fileName,std::ios_base::app);
            else outstream.open(fileName);
            outstream.precision(6);
            outstream << std::fixed;

            // Write the output fields to file
            for (int i=0; i<B_->mPsi(); i++) {
                outstream   << B_->Psi_min()+i*B_->dPsi()   << " " << std::scientific
                            << u_[i]                        << " " 
                            << up_[i]                       << " " 
                            << I0_[i]                       << " " 
                            << I1_[i]                       << " " << std::fixed
                            << std::endl;
            }
            outstream.close();
        }

    private:
        // Read the bias fields from the input bias file
        void read_Bias_Fields(std::string fileName) {
            double Psi;
            int mPsi = B_->mPsi();

            std::ifstream inFile;
            inFile.open(fileName);

            // Ignore parameters on first line (already read with read_Bias_Params(...))
            inFile.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            for (int i=0; i<mPsi; i++) {
                inFile >> Psi >> u_[i] >> up_[i] >> I0_[i] >> I1_[i];
            }
            inFile.close();

            // Bias field exists as the field has been updated
            nonZeroBias_ = true;
        }

        // Calculate wt(k) and f(k) lookup tables
        void calc_wt_fk(int *wt, double *fk, int *m, double *L) {
            int K0, K1, k;
            double kx_sq, ky_sq, kz_sq, K;
            double kc = B_->kc();

            for (k=0; k<Mk_; k++) wt[k]=2;

            for (int k0=-(m[0]-1)/2; k0<=m[0]/2; k0++) {
                K0 = (k0<0)?(k0+m[0]):k0;
                kx_sq = k0*k0/(L[0]*L[0]);

                for (int k1=-(m[1]-1)/2; k1<=m[1]/2; k1++) {
                    K1 = (k1<0)?(k1+m[1]):k1;
                    ky_sq = k1*k1/(L[1]*L[1]);

                    for (int k2=0; k2<=m[2]/2; k2++) {
                        kz_sq = k2*k2/(L[2]*L[2]);
                        k = k2 + (m[2]/2+1)*(K1+m[1]*K0);
                        K = 2*M_PI*pow(kx_sq+ky_sq+kz_sq,0.5);
                        fk[k] = 1.0/(1.0 + exp(12.0*(K-kc)/kc));
                        if ((k2==0)||(k2==m[2]/2)) wt[k]=1;
                    }
                }
            }
        }

};
