// Numerical check: gsycl cuFFT shim (oneMKL DFT) vs FFTW3, on the exact
// transforms JDFTx uses: Z2Z out-of-place/in-place both directions, D2Z, Z2D.
#include "cufft.h"
#include <sycl/sycl.hpp>
#include <fftw3.h>
#include <complex>
#include <vector>
#include <cstdio>
#include <cmath>
#include <cstdlib>

static double maxAbsDiff(const std::complex<double>* a, const std::complex<double>* b, size_t n)
{	double m = 0.;
	for(size_t i=0; i<n; i++) m = std::max(m, std::abs(a[i]-b[i]));
	return m;
}
static double maxAbsDiff(const double* a, const double* b, size_t n)
{	double m = 0.;
	for(size_t i=0; i<n; i++) m = std::max(m, std::fabs(a[i]-b[i]));
	return m;
}
static double maxAbs(const std::complex<double>* a, size_t n)
{	double m = 0.; for(size_t i=0;i<n;i++) m = std::max(m, std::abs(a[i])); return m;
}

int main()
{	// Deliberately non-cubic and odd/even mixed, to catch stride/order mistakes.
	const int S0 = 5, S1 = 6, S2 = 8;
	const size_t nR = size_t(S0)*S1*S2;
	const int S2h = S2/2 + 1;
	const size_t nG = size_t(S0)*S1*S2h;

	sycl::queue& q = jdftx_sycl::queue();
	std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());

	// --- host reference data ---
	std::vector<std::complex<double>> cIn(nR), cRef(nR), cGot(nR);
	std::vector<double> rIn(nR), rRef(nR), rGot(nR);
	std::vector<std::complex<double>> gRef(nG), gGot(nG), gIn(nG);
	std::srand(12345);
	auto rnd = []{ return (std::rand()/double(RAND_MAX))*2.-1.; };
	for(size_t i=0;i<nR;i++) { cIn[i] = {rnd(), rnd()}; rIn[i] = rnd(); }

	// --- FFTW references (unnormalised, same sign convention as cuFFT) ---
	{	fftw_plan p = fftw_plan_dft_3d(S0,S1,S2, (fftw_complex*)cIn.data(), (fftw_complex*)cRef.data(), FFTW_FORWARD, FFTW_ESTIMATE);
		fftw_execute(p); fftw_destroy_plan(p);
	}
	std::vector<std::complex<double>> cRefInv(nR);
	{	fftw_plan p = fftw_plan_dft_3d(S0,S1,S2, (fftw_complex*)cIn.data(), (fftw_complex*)cRefInv.data(), FFTW_BACKWARD, FFTW_ESTIMATE);
		fftw_execute(p); fftw_destroy_plan(p);
	}
	{	std::vector<double> tmp = rIn;
		fftw_plan p = fftw_plan_dft_r2c_3d(S0,S1,S2, tmp.data(), (fftw_complex*)gRef.data(), FFTW_ESTIMATE);
		fftw_execute(p); fftw_destroy_plan(p);
	}
	// c2r reference: use gRef as input (FFTW c2r may destroy input, so copy)
	{	gIn = gRef;
		std::vector<std::complex<double>> tmp = gIn;
		fftw_plan p = fftw_plan_dft_c2r_3d(S0,S1,S2, (fftw_complex*)tmp.data(), rRef.data(), FFTW_ESTIMATE);
		fftw_execute(p); fftw_destroy_plan(p);
	}

	// --- device buffers ---
	auto* dC1 = sycl::malloc_device<std::complex<double>>(nR, q);
	auto* dC2 = sycl::malloc_device<std::complex<double>>(nR, q);
	auto* dR  = sycl::malloc_device<double>(nR, q);
	auto* dG  = sycl::malloc_device<std::complex<double>>(nG, q);

	cufftHandle planZ2Z, planD2Z, planZ2D;
	if(cufftPlan3d(&planZ2Z, S0,S1,S2, CUFFT_Z2Z)) { std::printf("FAIL: cufftPlan3d Z2Z\n"); return 1; }
	if(cufftPlan3d(&planD2Z, S0,S1,S2, CUFFT_D2Z)) { std::printf("FAIL: cufftPlan3d D2Z\n"); return 1; }
	if(cufftPlan3d(&planZ2D, S0,S1,S2, CUFFT_Z2D)) { std::printf("FAIL: cufftPlan3d Z2D\n"); return 1; }

	int nFail = 0;
	auto check = [&](const char* what, double err, double scale)
	{	double tol = 1e-10 * std::max(scale, 1.0);
		std::printf("%-28s maxAbsErr=%9.3e tol=%9.3e  %s\n", what, err, tol, err<=tol?"OK":"FAIL");
		if(!(err<=tol)) nFail++;
	};

	// Z2Z forward, out of place
	q.memcpy(dC1, cIn.data(), nR*sizeof(std::complex<double>)).wait();
	cufftExecZ2Z(planZ2Z, (const cuDoubleComplex*)dC1, (cuDoubleComplex*)dC2, CUFFT_FORWARD);
	q.wait(); q.memcpy(cGot.data(), dC2, nR*sizeof(std::complex<double>)).wait();
	check("Z2Z forward out-of-place", maxAbsDiff(cGot.data(), cRef.data(), nR), maxAbs(cRef.data(), nR));

	// Z2Z inverse, out of place
	q.memcpy(dC1, cIn.data(), nR*sizeof(std::complex<double>)).wait();
	cufftExecZ2Z(planZ2Z, (const cuDoubleComplex*)dC1, (cuDoubleComplex*)dC2, CUFFT_INVERSE);
	q.wait(); q.memcpy(cGot.data(), dC2, nR*sizeof(std::complex<double>)).wait();
	check("Z2Z inverse out-of-place", maxAbsDiff(cGot.data(), cRefInv.data(), nR), maxAbs(cRefInv.data(), nR));

	// Z2Z forward, in place
	q.memcpy(dC1, cIn.data(), nR*sizeof(std::complex<double>)).wait();
	cufftExecZ2Z(planZ2Z, (const cuDoubleComplex*)dC1, (cuDoubleComplex*)dC1, CUFFT_FORWARD);
	q.wait(); q.memcpy(cGot.data(), dC1, nR*sizeof(std::complex<double>)).wait();
	check("Z2Z forward in-place", maxAbsDiff(cGot.data(), cRef.data(), nR), maxAbs(cRef.data(), nR));

	// Z2Z inverse, in place
	q.memcpy(dC1, cIn.data(), nR*sizeof(std::complex<double>)).wait();
	cufftExecZ2Z(planZ2Z, (const cuDoubleComplex*)dC1, (cuDoubleComplex*)dC1, CUFFT_INVERSE);
	q.wait(); q.memcpy(cGot.data(), dC1, nR*sizeof(std::complex<double>)).wait();
	check("Z2Z inverse in-place", maxAbsDiff(cGot.data(), cRefInv.data(), nR), maxAbs(cRefInv.data(), nR));

	// D2Z (real -> half-complex)
	q.memcpy(dR, rIn.data(), nR*sizeof(double)).wait();
	cufftExecD2Z(planD2Z, dR, (cuDoubleComplex*)dG);
	q.wait(); q.memcpy(gGot.data(), dG, nG*sizeof(std::complex<double>)).wait();
	check("D2Z (r2c)", maxAbsDiff(gGot.data(), gRef.data(), nG), maxAbs(gRef.data(), nG));

	// Z2D (half-complex -> real)
	q.memcpy(dG, gIn.data(), nG*sizeof(std::complex<double>)).wait();
	cufftExecZ2D(planZ2D, (const cuDoubleComplex*)dG, dR);
	q.wait(); q.memcpy(rGot.data(), dR, nR*sizeof(double)).wait();
	{	double scale=0; for(size_t i=0;i<nR;i++) scale=std::max(scale,std::fabs(rRef[i]));
		check("Z2D (c2r)", maxAbsDiff(rGot.data(), rRef.data(), nR), scale);
	}

	// Round trip D2Z -> Z2D should give N * original
	q.memcpy(dR, rIn.data(), nR*sizeof(double)).wait();
	cufftExecD2Z(planD2Z, dR, (cuDoubleComplex*)dG);
	cufftExecZ2D(planZ2D, (const cuDoubleComplex*)dG, dR);
	q.wait(); q.memcpy(rGot.data(), dR, nR*sizeof(double)).wait();
	{	std::vector<double> want(nR);
		for(size_t i=0;i<nR;i++) want[i] = rIn[i]*double(nR);
		double scale=0; for(size_t i=0;i<nR;i++) scale=std::max(scale,std::fabs(want[i]));
		check("D2Z->Z2D round trip = N*x", maxAbsDiff(rGot.data(), want.data(), nR), scale);
	}

	cufftDestroy(planZ2Z); cufftDestroy(planD2Z); cufftDestroy(planZ2D);
	sycl::free(dC1,q); sycl::free(dC2,q); sycl::free(dR,q); sycl::free(dG,q);
	std::printf(nFail ? "\n%d CHECK(S) FAILED\n" : "\nALL FFT CHECKS PASSED\n", nFail);
	return nFail ? 1 : 0;
}
