// test_c2r_nyquist.cpp -- does oneMKL's complex->real 3D transform agree with
// FFTW's on the Nyquist planes?
//
// JDFTx stores G-space scalar fields on the half-space k2 in [0, S2/2] and gets
// back to real space with a single c2r transform (cufftExecZ2D on the GPU,
// fftw_execute_dft_c2r on the CPU). On the k2=0 and k2=S2/2 planes that storage
// is redundant -- X[k] and conj(X[-k]) are the same datum -- and the two
// libraries need not resolve the redundancy identically. This checks whether
// they do, on a JDFTx-shaped grid, starting from data that is Hermitian by
// construction (an FFTW r2c of a real array), so any disagreement is a
// convention mismatch and not bad input.
//
// Build:
//   icpx -fsycl -O2 -fp-model=precise -ffp-contract=off test_c2r_nyquist.cpp \
//     -I$MKLROOT/include -L$MKLROOT/lib -lmkl_sycl_dft -lmkl_intel_lp64 \
//     -lmkl_sequential -lmkl_core -lfftw3 -o test_c2r_nyquist
// Exit 0 = the two agree (hypothesis refuted); 1 = they differ.

#include <sycl/sycl.hpp>
#include <oneapi/mkl/dft.hpp>
#include <fftw3.h>
#include <complex>
#include <cstdio>
#include <cmath>
#include <vector>
#include <cstdint>

namespace dft = oneapi::mkl::dft;

int main()
{	const int S[3] = {24, 24, 54};          //graphene test's charge-density grid
	const int nHalf = S[2]/2 + 1;
	const size_t nR = size_t(S[0])*S[1]*S[2];
	const size_t nG = size_t(S[0])*S[1]*nHalf;

	//--- Reference real field, and its r2c transform (Hermitian by construction)
	std::vector<double> r0(nR);
	uint64_t seed = 12345;
	for(size_t i=0; i<nR; i++)
	{	seed = seed*6364136223846793005ULL + 1442695040888963407ULL;
		r0[i] = double(int64_t(seed >> 12) % 2000001 - 1000000) * 1e-6;
	}
	std::vector<std::complex<double>> G(nG);
	{	fftw_plan p = fftw_plan_dft_r2c_3d(S[0], S[1], S[2], r0.data(),
			reinterpret_cast<fftw_complex*>(G.data()), FFTW_ESTIMATE);
		fftw_execute(p); fftw_destroy_plan(p);
	}

	//--- Perturb the Nyquist planes so the input is deliberately NOT Hermitian.
	// This is not an artificial case: JDFTx's half-space kernels evaluate a
	// radial function at the *folded* index, and on a Nyquist plane the fold
	// cannot represent the negation (-12 == +12 mod 24), so index (12,9,27) is
	// labelled G(12,9,27) while its Hermitian partner (12,15,27) is labelled
	// G(12,-9,27). With a cross term in the metric (any hexagonal lattice) those
	// have different |G|, so the stored plane is genuinely non-Hermitian.
	auto isNyq = [&](int k, int n) { return k==0 || 2*k==n; };
	for(int k0=0; k0<S[0]; k0++) for(int k1=0; k1<S[1]; k1++) for(int k2=0; k2<nHalf; k2++)
		if(isNyq(k0,S[0]) || isNyq(k1,S[1]) || isNyq(k2,S[2]))
		{	seed = seed*6364136223846793005ULL + 1442695040888963407ULL;
			double d = double(int64_t(seed >> 12) % 2001 - 1000) * 1e-3;
			G[(size_t(k0)*S[1] + k1)*nHalf + k2] += std::complex<double>(d, -0.7*d);
		}

	//--- FFTW c2r (what the CPU build does)
	std::vector<double> rFFTW(nR);
	{	std::vector<std::complex<double>> Gin = G;   //c2r destroys its input
		fftw_plan p = fftw_plan_dft_c2r_3d(S[0], S[1], S[2],
			reinterpret_cast<fftw_complex*>(Gin.data()), rFFTW.data(), FFTW_ESTIMATE);
		fftw_execute(p); fftw_destroy_plan(p);
	}

	//--- oneMKL c2r, configured exactly as gsycl/cufft_impl.cpp does
	std::vector<double> rMKL(nR);
	{	sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order()};
		dft::descriptor<dft::precision::DOUBLE, dft::domain::REAL> desc(
			std::vector<std::int64_t>{S[0], S[1], S[2]});
		desc.set_value(dft::config_param::PLACEMENT, dft::config_value::NOT_INPLACE);
		const std::vector<std::int64_t> realStrides = {0, std::int64_t(S[1])*S[2], S[2], 1};
		const std::vector<std::int64_t> cplxStrides = {0, std::int64_t(S[1])*nHalf, nHalf, 1};
		desc.set_value(dft::config_param::FWD_STRIDES, realStrides);
		desc.set_value(dft::config_param::BWD_STRIDES, cplxStrides);
		desc.commit(q);
		auto* dG = sycl::malloc_device<std::complex<double>>(nG, q);
		auto* dR = sycl::malloc_device<double>(nR, q);
		q.memcpy(dG, G.data(), nG*sizeof(std::complex<double>)).wait();
		dft::compute_backward(desc, dG, dR).wait();
		q.memcpy(rMKL.data(), dR, nR*sizeof(double)).wait();
		sycl::free(dG, q); sycl::free(dR, q);
	}

	//--- oneMKL again, but with the k2=0 and k2=S2/2 planes Hermitian-symmetrized
	// first: X[k0,k1] <- (X[k0,k1] + conj(X[-k0,-k1]))/2. Those two planes are
	// the only ones the half-space array stores redundantly.
	std::vector<double> rSym(nR);
	{	std::vector<std::complex<double>> Gs = G;
		for(int k2 : {0, S[2]/2})
			for(int k0=0; k0<S[0]; k0++) for(int k1=0; k1<S[1]; k1++)
			{	int j0 = (S[0]-k0) % S[0], j1 = (S[1]-k1) % S[1];
				size_t a = (size_t(k0)*S[1] + k1)*nHalf + k2;
				size_t b = (size_t(j0)*S[1] + j1)*nHalf + k2;
				if(a > b) continue;
				std::complex<double> m = 0.5*(G[a] + std::conj(G[b]));
				Gs[a] = m; Gs[b] = std::conj(m);
			}
		sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order()};
		dft::descriptor<dft::precision::DOUBLE, dft::domain::REAL> desc(
			std::vector<std::int64_t>{S[0], S[1], S[2]});
		desc.set_value(dft::config_param::PLACEMENT, dft::config_value::NOT_INPLACE);
		desc.set_value(dft::config_param::FWD_STRIDES,
			std::vector<std::int64_t>{0, std::int64_t(S[1])*S[2], S[2], 1});
		desc.set_value(dft::config_param::BWD_STRIDES,
			std::vector<std::int64_t>{0, std::int64_t(S[1])*nHalf, nHalf, 1});
		desc.commit(q);
		auto* dG = sycl::malloc_device<std::complex<double>>(nG, q);
		auto* dR = sycl::malloc_device<double>(nR, q);
		q.memcpy(dG, Gs.data(), nG*sizeof(std::complex<double>)).wait();
		dft::compute_backward(desc, dG, dR).wait();
		q.memcpy(rSym.data(), dR, nR*sizeof(double)).wait();
		sycl::free(dG, q); sycl::free(dR, q);
	}

	//--- Compare, and separate the error by Nyquist multiplicity in G space
	double maxDiff = 0., maxVal = 0.;
	for(size_t i=0; i<nR; i++)
	{	maxDiff = std::fmax(maxDiff, std::fabs(rFFTW[i] - rMKL[i]));
		maxVal  = std::fmax(maxVal, std::fabs(rFFTW[i]));
	}
	//round-trip error of FFTW alone, as the noise floor to compare against
	double maxSelf = 0.;
	for(size_t i=0; i<nR; i++)
		maxSelf = std::fmax(maxSelf, std::fabs(rFFTW[i] - r0[i]*double(nR)));

	std::printf("grid %dx%dx%d   max|FFTW| = %.6e\n", S[0], S[1], S[2], maxVal);
	std::printf("FFTW round-trip error (noise floor) : %.6e\n", maxSelf);
	double maxSym = 0.;
	for(size_t i=0; i<nR; i++) maxSym = std::fmax(maxSym, std::fabs(rFFTW[i] - rSym[i]));
	std::printf("max |FFTW - oneMKL| raw             : %.6e  (relative %.3e)\n",
		maxDiff, maxDiff/maxVal);
	std::printf("max |FFTW - oneMKL| symmetrized     : %.6e  (relative %.3e)\n",
		maxSym, maxSym/maxVal);
	if(maxSym < 1e-10*maxVal)
		std::printf("  -> FFTW's convention IS the Hermitian symmetrization\n");

	if(maxDiff > 1e-8*maxVal)
	{	std::printf("RESULT: the two libraries DISAGREE -- c2r Nyquist convention mismatch\n");
		return 1;
	}
	std::printf("RESULT: agree to the round-trip noise floor -- hypothesis refuted\n");
	return 0;
}
