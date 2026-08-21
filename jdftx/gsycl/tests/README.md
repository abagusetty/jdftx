# gsycl standalone checks

Small self-contained programs that validate one shim mapping each, without
building JDFTx. They are not part of the CMake build; compile them by hand.

## fft_check.cpp

Compares the `cufft.h` shim (oneMKL DFT) against FFTW3 on exactly the
transforms JDFTx performs — Z2Z forward/inverse, out-of-place and in-place,
plus D2Z and Z2D — on a deliberately non-cubic 5x6x8 grid, which is what
catches dimension-order and stride mistakes.

```bash
R=<jdftx-source-dir>            # the directory containing gsycl/
F=$FFTW3_ROOT
icpx -fsycl -std=c++17 -DGPU_ENABLED -DUSE_SYCL \
     -I$R/gsycl -I$R -I$F/include \
     fft_check.cpp $R/gsycl/cufft_impl.cpp -o fft_check \
     -L$F/lib -lfftw3 -qmkl=sequential -lmkl_sycl_dft -Wl,-rpath,$F/lib
./fft_check
```

Expected output ends with `ALL FFT CHECKS PASSED`.

## test_c2r_nyquist.cpp

`fft_check.cpp` above always feeds the inverse transform a spectrum produced by
a forward r2c, which is Hermitian-consistent by construction — so it cannot see
the one place FFTW and oneMKL genuinely differ. This check does: it perturbs
the redundantly-stored k2=0 and k2=S2/2 planes, where a half-space array holds
both X[k] and X[-k], and compares the two libraries on a JDFTx-shaped grid
(24x24x54, the graphene test's charge-density grid). It also checks the
candidate fix, Hermitian-symmetrizing those planes first.

Without the symmetrization the two disagree by ~2e-3 relative; with it they
agree to ~1e-15. `cufft_impl.cpp` therefore applies it inside `cufftExecZ2D`.
Left unfixed this shifted graphene's eigenvalues by up to 6e-4 Eh.

```bash
icpx -fsycl -O2 -fp-model=precise -ffp-contract=off test_c2r_nyquist.cpp \
     -I$MKLROOT/include -L$MKLROOT/lib \
     -lmkl_sycl_dft -lmkl_intel_lp64 -lmkl_sequential -lmkl_core \
     -I$FFTW3_ROOT/include -L$FFTW3_ROOT/lib -lfftw3 -o test_c2r_nyquist
./test_c2r_nyquist
```

Exit 1 (`the two libraries DISAGREE`) is the *expected* result: it demonstrates
the raw mismatch that motivates the fix, and prints the symmetrized comparison
to show the fix closes it. It is a demonstration, not a pass/fail gate on the
shim.

## test_slab_kernel_precision.cpp

Records a refuted hypothesis (FMA contraction in `CoulombSlab_calc`). Kept only
so it is not chased again; see the banner at the top of the file.
