// gsycl/cufft_impl.cpp -- oneMKL DPC++ DFT implementation of the cufft.h shim
// declarations. Sole translation unit that includes <oneapi/mkl/dft.hpp>
// (which transitively pulls in mkl_cblas.h); it never includes gsl_cblas.h,
// so the CBLAS enum redefinition clash cannot arise.
//
// Layout note: JDFTx uses cuFFT's default layouts, and oneMKL's real-domain
// CCE format matches them -- an S0 x S1 x S2 real grid transforms to
// S0 x S1 x (S2/2+1) complex, contiguous. Both libraries leave the transforms
// unnormalised in both directions, so no scale factors are needed.

#include "cufft.h"
#include <sycl/sycl.hpp>
#include <oneapi/mkl/dft.hpp>
#include <complex>
#include <cstdint>
#include <new>
#include <vector>

namespace {

namespace dft = oneapi::mkl::dft;
using DescC = dft::descriptor<dft::precision::DOUBLE, dft::domain::COMPLEX>;
using DescR = dft::descriptor<dft::precision::DOUBLE, dft::domain::REAL>;

//! oneMKL fixes in-place vs out-of-place at commit time, while a cuFFT plan
//! serves both (JDFTx uses the Z2Z plan each way), hence the two descriptors.
struct FftPlan
{	DescC* cOutOfPlace = nullptr;
	DescC* cInPlace = nullptr;
	DescR* real = nullptr;
	int S[3] = {0, 0, 0};                     //grid dimensions
	std::complex<double>* planeScratch = nullptr;  //S[0]*S[1] scratch, see hermitianSymmetrizePlane()

	~FftPlan()
	{	delete cOutOfPlace;
		delete cInPlace;
		delete real;
		if(planeScratch) sycl::free(planeScratch, jdftx_sycl::queue());
	}
};

DescC* makeComplex(const std::vector<std::int64_t>& n, bool inPlace)
{	DescC* desc = new DescC(n);
	desc->set_value(dft::config_param::PLACEMENT,
		inPlace ? dft::config_value::INPLACE : dft::config_value::NOT_INPLACE);
	desc->commit(jdftx_sycl::queue());
	return desc;
}

DescR* makeReal(const std::vector<std::int64_t>& n)
{	DescR* desc = new DescR(n);
	desc->set_value(dft::config_param::PLACEMENT, dft::config_value::NOT_INPLACE);
	//Match cuFFT's contiguous layouts explicitly rather than relying on the
	//oneMKL defaults, which have changed across releases for the real domain.
	const std::int64_t nHalf = n[2]/2 + 1;
	const std::vector<std::int64_t> realStrides = { 0, n[1]*n[2], n[2], 1 };
	const std::vector<std::int64_t> cplxStrides = { 0, n[1]*nHalf, nHalf, 1 };
	desc->set_value(dft::config_param::FWD_STRIDES, realStrides);
	desc->set_value(dft::config_param::BWD_STRIDES, cplxStrides);
	desc->commit(jdftx_sycl::queue());
	return desc;
}

//! Make the redundantly-stored planes of a half-space (CCE) array Hermitian
//! before a complex->real transform.
//!
//! A c2r transform's input is only half the spectrum; the k2=0 and (for even
//! S2) k2=S2/2 planes are the two where the array stores BOTH X[k] and X[-k],
//! so a well-formed input must have X[k0,k1] == conj(X[-k0,-k1]) there. JDFTx's
//! half-space kernels do not always produce that. They evaluate radial
//! functions at the *folded* index, and on the k2=S2/2 plane the fold cannot
//! represent the negation -- -S2/2 == +S2/2 modulo S2 -- so entry (k0,k1,S2/2)
//! is labelled G(k0,k1,+S2/2) while its stored Hermitian partner (-k0,-k1,S2/2)
//! is labelled G(-k0,-k1,+S2/2) instead of G(k0,k1,-S2/2). Whenever the metric
//! has a cross term (any hexagonal cell, e.g. the graphene test) those two have
//! different |G|, and the plane comes out genuinely non-Hermitian.
//!
//! For such input the two libraries do not agree, and neither is wrong -- the
//! input is out of contract. Measured on the graphene grid (24x24x54), FFTW's
//! result is exactly what you get by Hermitian-symmetrizing these planes first,
//! X[k] <- (X[k] + conj(X[-k]))/2, while oneMKL's is not; the two then differ
//! by ~1e-3 relative on those planes. That propagates to a ~1e-4 error in
//! Vlocps and shifts every eigenvalue by up to 6e-4 Eh, which is what made the
//! GPU miss graphene's Fermi-level check. Applying the symmetrization here
//! makes oneMKL reproduce FFTW to 1e-15, so the CPU and GPU builds agree.
//! See gsycl/tests/test_c2r_nyquist.cpp for the standalone demonstration.
//!
//! Mutating the input is allowed: cuFFT and FFTW both document that c2r may
//! destroy it, and JDFTx honours that (core/Operators.cpp's I() clones first
//! unless it was handed an expiring value).
void hermitianSymmetrizePlane(FftPlan* p, std::complex<double>* data, int k2)
{	const int S0 = p->S[0], S1 = p->S[1];
	const std::int64_t nHalf = p->S[2]/2 + 1;
	const size_t nPlane = size_t(S0)*S1;
	std::complex<double>* scratch = p->planeScratch;
	sycl::queue& q = jdftx_sycl::queue();
	//Gather the (strided) plane, so the symmetrization reads pre-update values
	//only -- writing X[k] and X[-k] in one pass would race on the partner read.
	q.parallel_for(sycl::range<1>(nPlane), [=](sycl::id<1> idx)
	{	size_t j = idx[0];
		scratch[j] = data[j*nHalf + k2];
	});
	q.parallel_for(sycl::range<1>(nPlane), [=](sycl::id<1> idx)
	{	size_t j = idx[0];
		int k0 = int(j / S1), k1 = int(j % S1);
		int j0 = (S0 - k0) % S0, j1 = (S1 - k1) % S1;   //index of -k
		std::complex<double> partner = scratch[size_t(j0)*S1 + j1];
		//Self-partnered entries (k0 and k1 both 0 or Nyquist) collapse to Re(X),
		//which is what this same expression gives.
		data[j*nHalf + k2] = 0.5*(scratch[j] + std::conj(partner));
	});
}

inline FftPlan* planOf(const cufftHandle& plan)
{	return static_cast<FftPlan*>(plan.desc);
}

//! oneMKL only instantiates compute_forward/backward for non-const data types,
//! so strip const from the input pointers (they are read-only regardless).
template<typename T> inline T* unConst(const T* p) { return const_cast<T*>(p); }

//! cuFFT reports failures through a return code; oneMKL throws. Convert, and
//! record the message where cudaGetErrorString()/gpuErrorCheck() will find it.
template<typename F> inline cudaError_t guardedDft(F&& f)
{	cudaError_t status = cudaSuccess;
	jdftx_sycl::guarded([&]{ f(); });
	if(not jdftx_sycl::lastError().empty()) status = 1;
	return status;
}

} // namespace

cudaError_t cufftPlan3d(cufftHandle* plan, int nx, int ny, int nz, int type)
{	const std::vector<std::int64_t> n = { std::int64_t(nx), std::int64_t(ny), std::int64_t(nz) };
	FftPlan* p = new FftPlan();
	try
	{	if(type == CUFFT_Z2Z)
		{	p->cOutOfPlace = makeComplex(n, false);
			p->cInPlace = makeComplex(n, true);
		}
		else if(type == CUFFT_D2Z or type == CUFFT_Z2D)
		{	p->real = makeReal(n);
			p->S[0] = nx; p->S[1] = ny; p->S[2] = nz;
			p->planeScratch = sycl::malloc_device<std::complex<double>>(
				size_t(nx)*ny, jdftx_sycl::queue());
			if(not p->planeScratch) throw std::bad_alloc();
		}
		else
		{	delete p;
			plan->desc = nullptr; plan->type = type; plan->initialized = false;
			return 1;
		}
	}
	catch(const std::exception&)
	{	delete p;
		plan->desc = nullptr; plan->type = type; plan->initialized = false;
		return 1;
	}
	plan->desc = p;
	plan->type = type;
	plan->initialized = true;
	return cudaSuccess;
}

cudaError_t cufftExecZ2Z(cufftHandle plan, const cuDoubleComplex* in,
	cuDoubleComplex* out, int direction)
{	if(not plan.initialized) return 1;
	FftPlan* p = planOf(plan);
	std::complex<double>* inPtr = unConst(reinterpret_cast<const std::complex<double>*>(in));
	std::complex<double>* outPtr = reinterpret_cast<std::complex<double>*>(out);
	const bool inPlace = (inPtr == outPtr);
	DescC* desc = inPlace ? p->cInPlace : p->cOutOfPlace;
	if(not desc) return 1;
	return guardedDft([&]
	{	if(direction == CUFFT_FORWARD)
		{	if(inPlace) dft::compute_forward(*desc, inPtr);
			else dft::compute_forward(*desc, inPtr, outPtr);
		}
		else
		{	if(inPlace) dft::compute_backward(*desc, inPtr);
			else dft::compute_backward(*desc, inPtr, outPtr);
		}
	});
}

cudaError_t cufftExecD2Z(cufftHandle plan, const double* in, cuDoubleComplex* out)
{	if(not plan.initialized) return 1;
	DescR* desc = planOf(plan)->real;
	if(not desc) return 1;
	return guardedDft([&]
	{	dft::compute_forward(*desc, unConst(in), reinterpret_cast<std::complex<double>*>(out));
	});
}

cudaError_t cufftExecZ2D(cufftHandle plan, const cuDoubleComplex* in, double* out)
{	if(not plan.initialized) return 1;
	FftPlan* p = planOf(plan);
	DescR* desc = p->real;
	if(not desc) return 1;   //only a real plan has p->S and p->planeScratch set
	std::complex<double>* inPtr = unConst(reinterpret_cast<const std::complex<double>*>(in));
	return guardedDft([&]
	{	//Bring the redundantly-stored planes into contract before transforming,
		//so oneMKL and FFTW resolve them the same way (see above).
		hermitianSymmetrizePlane(p, inPtr, 0);
		if(p->S[2] % 2 == 0) hermitianSymmetrizePlane(p, inPtr, p->S[2]/2);
		dft::compute_backward(*desc, inPtr, out);
	});
}

cudaError_t cufftDestroy(cufftHandle plan)
{	delete planOf(plan);
	return cudaSuccess;
}
