// ============================================================================
// HYPOTHESIS REFUTED (2026-08-21) -- kept only as a record. Do not re-chase.
//
// The graphene/metalSurface failures were NOT FMA contraction in
// CoulombSlab_calc. Two independent checks killed it: rewriting operator() to
// take Gplane from the zeroed vector (the sibling latticeGradient()'s form),
// and compiling the whole tree with -ffp-contract=off, each left graphene's mu
// unchanged to 1e-9. The Slab correlation was also coincidence -- plain
// `coulomb-interaction Periodic` graphene showed the same error.
//
// The real cause was a c2r FFT Nyquist-plane convention mismatch between FFTW
// and oneMKL; see test_c2r_nyquist.cpp and the fix in ../cufft_impl.cpp.
// ============================================================================

// gsycl/tests/test_slab_kernel_precision.cpp
//
// Probe for the suspected root cause of the graphene / metalSurface GPU
// failures (the only two JDFTx tests using `coulomb-interaction Slab`).
//
// CoulombSlab_calc::operator() in core/Coulomb_internal.h computes the
// in-plane |G| by SUBTRACTION:
//
//     double Gsq    = GGT.metric_length_squared(iG);
//     double Gplane = Gsq - GGT(iDir,iDir) * iG[iDir]*iG[iDir];
//     Gplane = Gplane>0. ? sqrt(Gplane) : 0.;   // "safe sqrt" per upstream
//
// For grid points with no in-plane component (iG[0]==iG[1]==0) the true value
// is exactly 0. But `Gsq` is a rounded double, so if the compiler contracts
// the subtraction into fma(-GGT22*iGz, iGz, Gsq) -- which icpx MAY do, since
// this build sets -fp-model=precise but never sets -ffp-contract, leaving
// contraction at the default `fast` -- the product is evaluated exactly and
// the leftover is the rounding error of Gsq, i.e. ~1e-16*Gsq > 0.
//
// sqrt() then amplifies that to ~1e-8, so exp(-Gplane*hlfL) is no longer
// exactly 1, and the kernel value
//     4*pi * (1 - exp(-Gplane*hlfL)*cos(pi*iGz)) / Gsq
// which must be EXACTLY 0 for even iGz, becomes nonzero. Those (0,0,even)
// points carry the planar-averaged potential, i.e. the slab's absolute
// potential reference -- so the whole band structure shifts rigidly while the
// total energy (neutral cell) barely moves. That is precisely the observed
// signature: graphene energy correct to 4e-6 Ha but mu off by 4.6e-4 Ha.
//
// Note the sibling method latticeGradient() in the SAME struct already avoids
// this by zeroing the component first:
//     vector3<int> iGplane = iG; iGplane[iDir] = 0;
//     double Gplane = sqrt(GGT.metric_length_squared(iGplane));
//
// Build (no JDFTx headers needed):
//   icpx -fsycl -O3 -fp-model=precise -fsycl-targets=spir64_gen \
//        -Xsycl-target-backend=spir64_gen "-device pvc" \
//        -o test_slab_kernel_precision test_slab_kernel_precision.cpp
// Run:
//   ZE_AFFINITY_MASK=0.0 ./test_slab_kernel_precision
//
// Exit code 0 = device matches host exactly (hypothesis refuted).
// Exit code 1 = device differs (hypothesis supported).

#include <sycl/sycl.hpp>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {

// graphene test cell: `lattice Hexagonal 4.651 11`, truncated along 001.
// GGT = G * G^T for the hexagonal reciprocal lattice; only the diagonal (2,2)
// and the in-plane 2x2 block are needed here.
constexpr double aLat = 4.651, cLat = 11.0;
constexpr int iDir = 2;

struct GGTmat
{   double m[3][3];
    double operator()(int i, int j) const { return m[i][j]; }
};

GGTmat makeGGT()
{   // Hexagonal: a1=(a,0,0), a2=(-a/2, a*sqrt(3)/2, 0), a3=(0,0,c)
    // Reciprocal metric GGT(i,j) = dot(b_i, b_j), b = 2*pi*inv(R)^T
    const double twoPi = 2*M_PI;
    const double gx = twoPi/aLat, gy = twoPi/(aLat*std::sqrt(3.0)/2.0), gz = twoPi/cLat;
    GGTmat g{};
    // b1 = (gx, gy/... ) -- exact form is not critical; what matters is that
    // GGT is block-diagonal (in-plane 2x2 + separate zz), which slab mode
    // guarantees via checkOrthogonality().
    g.m[0][0] = gx*gx + 0.25*gy*gy;
    g.m[0][1] = g.m[1][0] = -0.5*gy*gy;
    g.m[1][1] = gy*gy;
    g.m[2][2] = gz*gz;
    g.m[0][2] = g.m[2][0] = 0.0;
    g.m[1][2] = g.m[2][1] = 0.0;
    return g;
}

// Mirrors core/matrix3.h metric_length_squared(): full double sum over i,j.
inline double metricLenSq(const GGTmat& g, int v0, int v1, int v2)
{   const double v[3] = { double(v0), double(v1), double(v2) };
    double r = 0.;
    for(int i=0; i<3; i++)
        for(int j=0; j<3; j++)
            r += g(i,j) * v[i] * v[j];
    return r;
}

} // namespace

int main()
{   sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order()};
    std::printf("device: %s\n\n", q.get_device().get_info<sycl::info::device::name>().c_str());

    const GGTmat g = makeGGT();
    const double hlfL = 0.5*cLat;
    const int nz = 96;   //iG[2] range for a realistic slab grid

    // ---- host evaluation, in-plane-zero line iG = (0,0,n) ----------------
    std::vector<double> hGplaneRaw(nz), hKernel(nz);
    for(int n=0; n<nz; n++)
    {   const double Gsq = metricLenSq(g, 0, 0, n);
        double Gplane = Gsq - g(iDir,iDir) * double(n)*double(n);
        hGplaneRaw[n] = Gplane;
        Gplane = Gplane>0. ? std::sqrt(Gplane) : 0.;
        hKernel[n] = Gsq ? (4*M_PI)*(1. - std::exp(-Gplane*hlfL)*std::cos(M_PI*n))/Gsq
                         : (4*M_PI)*(-0.5*hlfL*hlfL);
    }

    // ---- device evaluation, identical source expressions ------------------
    double* dGplaneRaw = sycl::malloc_device<double>(nz, q);
    double* dKernel    = sycl::malloc_device<double>(nz, q);
    const GGTmat gDev = g;
    q.parallel_for(sycl::range<1>(nz), [=](sycl::id<1> id)
    {   const int n = int(id[0]);
        const double v[3] = { 0., 0., double(n) };
        double Gsq = 0.;
        for(int i=0; i<3; i++)
            for(int j=0; j<3; j++)
                Gsq += gDev.m[i][j] * v[i] * v[j];
        double Gplane = Gsq - gDev.m[iDir][iDir] * double(n)*double(n);
        dGplaneRaw[n] = Gplane;
        Gplane = Gplane>0. ? sycl::sqrt(Gplane) : 0.;
        dKernel[n] = Gsq ? (4*M_PI)*(1. - sycl::exp(-Gplane*hlfL)*sycl::cos(M_PI*double(n)))/Gsq
                         : (4*M_PI)*(-0.5*hlfL*hlfL);
    }).wait();

    std::vector<double> dGraw(nz), dK(nz);
    q.memcpy(dGraw.data(), dGplaneRaw, nz*sizeof(double)).wait();
    q.memcpy(dK.data(), dKernel, nz*sizeof(double)).wait();

    // ---- report -----------------------------------------------------------
    std::printf("in-plane-zero line iG=(0,0,n): Gplane BEFORE sqrt must be exactly 0\n");
    std::printf("%4s %24s %24s %14s\n", "n", "host Gsq-GGT22*n^2", "device Gsq-GGT22*n^2", "device sqrt");
    int nHostNonZero=0, nDevNonZero=0;
    for(int n=0; n<nz; n++)
    {   if(hGplaneRaw[n] != 0.0) nHostNonZero++;
        if(dGraw[n] != 0.0) nDevNonZero++;
        if(n < 12 || dGraw[n] != 0.0)
            if(n < 24)
                std::printf("%4d %24.17g %24.17g %14.6e\n", n, hGplaneRaw[n], dGraw[n],
                    dGraw[n] > 0 ? std::sqrt(dGraw[n]) : 0.0);
    }
    std::printf("\n  host  nonzero residuals: %d / %d\n", nHostNonZero, nz);
    std::printf("  device nonzero residuals: %d / %d\n\n", nDevNonZero, nz);

    std::printf("kernel value at iG=(0,0,even) must be EXACTLY 0\n");
    double worstEven = 0.; int nWorst = 0;
    for(int n=2; n<nz; n+=2)
        if(std::fabs(dK[n]) > worstEven) { worstEven = std::fabs(dK[n]); nWorst = n; }
    double worstEvenHost = 0.;
    for(int n=2; n<nz; n+=2) worstEvenHost = std::max(worstEvenHost, std::fabs(hKernel[n]));
    std::printf("  host   max |kernel| over even n>0 = %.6e\n", worstEvenHost);
    std::printf("  device max |kernel| over even n>0 = %.6e  (at n=%d)\n\n", worstEven, nWorst);

    double maxKdiff = 0.;
    for(int n=0; n<nz; n++) maxKdiff = std::max(maxKdiff, std::fabs(dK[n]-hKernel[n]));
    std::printf("  max |device-host| kernel on this line = %.6e\n\n", maxKdiff);

    const bool clean = (nDevNonZero == 0) && (worstEven == 0.0) && (maxKdiff == 0.0);
    std::printf("VERDICT: %s\n", clean
        ? "device reproduces host exactly -- FMA-contraction hypothesis REFUTED"
        : "device differs from host -- FMA contraction in the Gplane subtraction is REAL;\n"
          "         fix by computing Gplane from the zeroed in-plane vector, as\n"
          "         CoulombSlab_calc::latticeGradient() already does.");

    sycl::free(dGplaneRaw, q); sycl::free(dKernel, q);
    return clean ? 0 : 1;
}
