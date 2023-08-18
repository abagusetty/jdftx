/*-------------------------------------------------------------------
Copyright 2011 Ravishankar Sundararaman, Kendra Letchworth Weaver

This file is part of JDFTx.

JDFTx is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

JDFTx is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with JDFTx.  If not, see <http://www.gnu.org/licenses/>.
-------------------------------------------------------------------*/

#ifndef JDFTX_CORE_MINIMIZE_H
#define JDFTX_CORE_MINIMIZE_H

#include <core/MinimizeParams.h>
#include <electronic/PerturbationParams.h>
#include <core/Util.h>
#include <deque>
#include <cmath>
#include <cfloat>
#include <algorithm>

//! @addtogroup Algorithms
//! @{

/**
@file Minimize.h
@brief Nonlinear minimization and linear solve templates
*/

/** Interface (abstract base class) for the minimization algorithm template
	@tparam Vector A data type that represents a direction in the tangent space of the parameter manifold,
	which must have the following functions/operators defined: \n
		- Vector& operator*=(double); (scalar multiply) \n
		- void axpy(double alpha, const Vector& X, Vector& Y); (accumulate operation: Y += alpha*X) \n
		- double dot(const Vector& X, const Vector& Y); (inner product) \n
		- Vector clone(const Vector& X); (create a copy) \n
		- void randomize(Vector&); (initialize with random numbers, used for auto-fdtests)
*/
template<typename Vector> struct Minimizable
{
	//! Move the state in parameter space along direction dir with scale alpha
	virtual void step(const Vector& dir, double alpha)=0;
	
	//! Returns the objective function at the current state and store the gradient in grad and preconditioned gradient in Kgrad, if non-null.
	virtual double compute(Vector* grad, Vector* Kgrad)=0;
	
	//! Override for optional processing/reporting after each/every few iterations
	//! It should return whether the state was modified
	virtual bool report(int iter) { return false; }
	
	//! Constrain search directions to the space of free directions for minimize.
	virtual void constrain(Vector&) {}
	
	//! Override to synchronize scalars over MPI processes (if the same minimization is happening in sync over many processes)
	virtual double sync(double x) const { return x; }
	
	//! Override to return maximum safe step size along a given direction. Steps can be arbitrarily large by default.
	virtual double safeStepSize(const Vector& dir) const { return DBL_MAX; }
	
	//! Minimize this objective function with algorithm controlled by params and return the minimized value
	double minimize(const MinimizeParams& params);
	
	//! Checks the consistency of the value and gradient returned by compute.
	//! params is used primarily to control output
	void fdTest(const MinimizeParams& params);
	
private:
	typedef bool (*Linmin)(Minimizable<Vector>&, const MinimizeParams&, const Vector&, double, double&, double&, Vector&, Vector&);
	Linmin getLinmin(const MinimizeParams& params) const; //!< Return function pointer to appropriate linmin method based on MinimizeParams
	double lBFGS(const MinimizeParams& params); //!< limited memory BFGS implementation (differs sufficiently from CG to be justify a separate implementation)
};

/** Interface (abstract base class) for linear conjugate gradients template which
	minimizes (1/2) dot(state, hessian*state) - dot(state, rhs) which is equivalent
	to the linear solve hessian * state == rhs
	@tparam Vector Same requirements as the Vector for #Minimizable
*/
template<typename Vector> struct LinearSolvable
{
	Vector state; //!< the location of the minimum, obtained by solving hessian * state == rhs
	
	//! Return vector multiplied by the hessian of the objective function
	virtual Vector hessian(const Vector&) const=0;
	
	//! Override to enable preconditioning: return the preconditioned vector, given a vector
	virtual Vector precondition(const Vector& v) const { return clone(v); }
	
	//! Override to synchronize scalars over MPI processes (if the same minimization is happening in sync over many processes)
	virtual double sync(double x) const { return x; }
	
	//! Solve the linear system hessian * state == rhs using conjugate gradients:
	//! @return the number of iterations taken to achieve target tolerance
	int solve(const Vector& rhs, const MinimizeParams& params);
};

/** Interface for symmetric (possibly indefinite) linear systems of the form Ax+b = 0.
	Choose between conjugate gradient or MINRES algorithms. Used in variational perturbation solver.
	CG implementation more memory-efficient compared to LinearSolvable. MINRES uses more memory but converges smoothly.
	@tparam Vector Same requirements as the Vector for #Minimizable
*/
template<typename Vector> struct LinearSolvableIndefinite 
{
	Vector state; //!< Solution to the system A.state+b = 0
	
	//! Linear operator A applied to the vector v.
	virtual void hessian(Vector& Av, const Vector& v) = 0;
	
	//! Apply precondioner to vector v. Required for CG algorithm
	virtual void precondition(Vector& v) = 0;
	
	//! Apply square root of preconditioner to v. Required for MINRES algorithm
	virtual void Ksqrt(Vector& v) = 0;
	
	//! Override to synchronize scalars over MPI processes (if the same minimization is happening in sync over many processes)
	virtual double sync(double x) const { return x; }
	
	//! Solve the linear system Ax+b=0 using algorithm specified by params
	//! @return the number of iterations taken to achieve target tolerance
	int solve(const Vector& b, const PerturbationParams& params);
	
	//! Solve Ax+b=0 using linear conjugate gradients.
	//! @return the number of iterations taken to achieve target tolerance
	int CG(const Vector& b, const PerturbationParams& params);
	
	//! Solve Ax+b=0 using MINRES algorithm.
	//! @return the number of iterations taken to achieve target tolerance
	int MINRES(const Vector& b, const PerturbationParams& params);
};


//! Energy difference convergence check
class EdiffCheck : std::deque<double>
{	unsigned nDiff;
	double threshold;
public:
	EdiffCheck(unsigned nDiff, double threshold); //!< Energy change within threshold nDiff consecutive times
	bool checkConvergence(double E); //!< Given new energy, and based on past history of calls, return whether convergence has been achieved
};

//! @}
//---------------------- Implementation ----------------------------
//!@cond

#include <core/Minimize_linmin.h>
#include <core/Minimize_lBFGS.h>

template<typename Vector> double Minimizable<Vector>::minimize(const MinimizeParams& p)
{	if(p.fdTest) fdTest(p); // finite difference test
	if(p.dirUpdateScheme == MinimizeParams::LBFGS) return lBFGS(p);
	
	Vector g, gPrev, Kg; //current, previous and preconditioned gradients
	double E = sync(compute(&g, &Kg)); //get initial energy and gradient
	EdiffCheck ediffCheck(p.nEnergyDiff, p.energyDiffThreshold); //list of past energies
	
	Vector d = clone(Kg); //step direction (will be reset in first iteration)
	constrain(d); //restrict search direction to allowed subspace
	bool forceGradDirection = true; //whether current direction is along the gradient
	MinimizeParams::DirectionUpdateScheme currentDirUpdateScheme = p.dirUpdateScheme; //initially use the specified scheme, may switch to SD on trouble
	bool gPrevUsed;
	switch(currentDirUpdateScheme)
	{	case MinimizeParams::FletcherReeves:
		case MinimizeParams::SteepestDescent:
			gPrevUsed = false;
			break;
		default:
			gPrevUsed = true;
	}
	
	double alphaT = p.alphaTstart; //test step size
	double alpha = alphaT; //actual step size
	double beta = 0.0; //CG prev search direction mix factor
	double gKNorm = 0.0, gKNormPrev = 0.0; //current and previous norms of the preconditioned gradient

	//Select the linmin method:
	Linmin linmin = getLinmin(p);
	
	//Iterate until convergence, max iteration count or kill signal
	int iter=0;
	for(iter=0; !killFlag; iter++)
	{
		if(report(iter)) //optional reporting/processing
		{	E = sync(compute(&g, &Kg)); //update energy and gradient if state was modified
			fprintf(p.fpLog, "%s\tState modified externally: resetting search direction.\n", p.linePrefix);
			fflush(p.fpLog);
			forceGradDirection = true; //reset search direction
		}
		
		gKNorm = sync(dot(g,Kg));
		fprintf(p.fpLog, "%sIter: %3d  %s: ", p.linePrefix, iter, p.energyLabel);
		fprintf(p.fpLog, p.energyFormat, E);
		fprintf(p.fpLog, "  |grad|_K: %10.3le  alpha: %10.3le", sqrt(gKNorm/p.nDim), alpha);

		//Print prev step stats and set CG direction parameter if necessary
		beta = 0.0;
		if(!forceGradDirection)
		{	double dotgd = sync(dot(g,d));
			double dotgPrevKg = gPrevUsed ? sync(dot(gPrev, Kg)) : 0.;

			fprintf(p.fpLog, "  linmin: %10.3le", dotgd/sqrt(sync(dot(g,g))*sync(dot(d,d))));
			if(gPrevUsed)
				fprintf(p.fpLog, "  cgtest: %10.3le", dotgPrevKg/sqrt(gKNorm*gKNormPrev));
			fprintf(p.fpLog, "  t[s]: %9.2lf", clock_sec());

			//Update beta:
			switch(currentDirUpdateScheme)
			{	case MinimizeParams::FletcherReeves:  beta = gKNorm/gKNormPrev; break;
				case MinimizeParams::PolakRibiere:    beta = (gKNorm-dotgPrevKg)/gKNormPrev; break;
				case MinimizeParams::HestenesStiefel: beta = (gKNorm-dotgPrevKg)/(dotgd-sync(dot(d,gPrev))); break;
				case MinimizeParams::SteepestDescent: beta = 0.0; break;
				case MinimizeParams::LBFGS: break; //Should never encounter since LBFGS handled separately; just to eliminate compiler warnings
			}
			if(beta<0.0)
			{	fprintf(p.fpLog, "\n%sEncountered beta<0, resetting CG.", p.linePrefix);
				beta=0.0;
			}
		}
		forceGradDirection = false;
		fprintf(p.fpLog, "\n"); fflush(p.fpLog);
		if(sqrt(gKNorm/p.nDim) < p.knormThreshold)
		{	fprintf(p.fpLog, "%sConverged (|grad|_K<%le).\n", p.linePrefix, p.knormThreshold);
			fflush(p.fpLog); return E;
		}
		if(ediffCheck.checkConvergence(E))
		{	fprintf(p.fpLog, "%sConverged (|Delta %s|<%le for %d iters).\n",
				p.linePrefix, p.energyLabel, p.energyDiffThreshold, p.nEnergyDiff);
			fflush(p.fpLog); return E;
		}
		if(!std::isfinite(gKNorm))
		{	fprintf(p.fpLog, "%s|grad|_K=%le. Stopping ...\n", p.linePrefix, gKNorm);
			fflush(p.fpLog); return E;
		}
		if(!std::isfinite(E))
		{	fprintf(p.fpLog, "%sE=%le. Stopping ...\n", p.linePrefix, E);
			fflush(p.fpLog); return E;
		}
		if(iter>=p.nIterations) break;
		if(gPrevUsed) gPrev = g;
		gKNormPrev = gKNorm;

		//Update search direction
		d *= beta; axpy(-1.0, Kg, d);  // d = beta*d - Kg
		constrain(d); //restrict search direction to allowed subspace
	
		//Line minimization
		alphaT = std::min(alphaT, safeStepSize(d));
		if(linmin(*this, p, d, alphaT, alpha, E, g, Kg))
		{	//linmin succeeded:
			if(p.updateTestStepSize)
			{	alphaT = alpha;
				if(alphaT<p.alphaTmin) //bad step size
					alphaT = p.alphaTstart; //make sure next test step size is not too bad
			}
		}
		else
		{	//linmin failed:
			fprintf(p.fpLog, "%s\tUndoing step.\n", p.linePrefix);
			step(d, -alpha);
			E = sync(compute(&g, &Kg));
			if(beta)
			{	//Failed, but not along the gradient direction:
				fprintf(p.fpLog, "%s\tStep failed: resetting search direction.\n", p.linePrefix);
				fflush(p.fpLog);
				forceGradDirection = true; //reset search direction
			}
			else
			{	//Failed along the gradient direction
				fprintf(p.fpLog, "%s\tStep failed along negative gradient direction.\n", p.linePrefix);
				fprintf(p.fpLog, "%sProbably at roundoff error limit. (Stopping)\n", p.linePrefix);
				fflush(p.fpLog);
				return E;
			}
		}
	}
	fprintf(p.fpLog, "%sNone of the convergence criteria satisfied after %d iterations.\n", p.linePrefix, iter);
	return E;
}


template<typename Vector> void Minimizable<Vector>::fdTest(const MinimizeParams& p)
{	
	const double deltaMin = 1e-9;
	const double deltaMax = 1e+1;
	const double deltaScale = 1e+1;
	string fdPrefixString = p.linePrefix + string("fdTest: ");
	const char* fdPrefix = fdPrefixString.c_str();
	fprintf(p.fpLog, "%s--------------------------------------\n", fdPrefix);
	Vector g, Kg;
	double E0 = sync(compute(&g, &Kg));
	
	Vector dx;
	{	// Set the direction to be a random vector of the same norm
		// as the preconditioned gradient times the initial test step size
		dx = clone(Kg);
		randomize(dx);
		constrain(dx);
		dx *= p.alphaTstart * sqrt(sync(dot(Kg,Kg))/sync(dot(dx,dx)));
	}
	double dE_ddelta = sync(dot(dx, g)); //directional derivative at delta=0

	double deltaPrev=0;
	for(double delta=deltaMin; delta<=deltaMax; delta*=deltaScale)
	{	double dE = dE_ddelta*delta;
		step(dx, delta-deltaPrev); deltaPrev=delta;
		double deltaE = sync(compute(0,0)) - E0;
		fprintf(p.fpLog, "%s   delta=%le:\n", fdPrefix, delta);
		fprintf(p.fpLog, "%s      d%s Ratio: %19.16lf\n", fdPrefix, p.energyLabel, deltaE/dE);
		fprintf(p.fpLog, "%s      d%s Error: %19.16lf\n", fdPrefix, p.energyLabel, sqrt(p.nDim)*1.1e-16/fabs(dE));
	}
	fprintf(p.fpLog, "%s--------------------------------------\n", fdPrefix);
	step(dx, -deltaPrev); //restore state to original value
}

//Return function pointer to appropriate linmin method based on MinimizeParams
template<typename Vector> typename Minimizable<Vector>::Linmin Minimizable<Vector>::getLinmin(const MinimizeParams& p) const
{	using namespace MinimizeLinmin;
	switch(p.linminMethod)
	{	case MinimizeParams::DirUpdateRecommended:
		{	switch(p.dirUpdateScheme)
			{	case MinimizeParams::SteepestDescent: return linminRelax<Vector>;
				case MinimizeParams::LBFGS: return linminCubicWolfe<Vector>;
				default: return linminQuad<Vector>; //Default for all nonlinear CG methods
			}
		}
		case MinimizeParams::Relax: return linminRelax<Vector>;
		case MinimizeParams::Quad: return linminQuad<Vector>;
		case MinimizeParams::CubicWolfe: return linminCubicWolfe<Vector>;
	}
	return 0;
}


template<typename Vector> int LinearSolvable<Vector>::solve(const Vector& rhs, const MinimizeParams& p)
{	//Initialize:
	Vector r = clone(rhs); axpy(-1.0, hessian(state), r); //residual r = rhs - A.state;
	Vector z = precondition(r), d = r; //the preconditioned residual and search direction
	double beta=0.0, rdotzPrev=0.0, rdotz = sync(dot(r, z));

	//Check initial residual
	double rzNorm = sqrt(fabs(rdotz)/p.nDim);
	fprintf(p.fpLog, "%sInitial:  sqrt(|r.z|): %12.6le\n", p.linePrefix, rzNorm); fflush(p.fpLog);
	if(rzNorm<p.knormThreshold) { fprintf(p.fpLog, "%sConverged sqrt(r.z)<%le\n", p.linePrefix, p.knormThreshold); fflush(p.fpLog); return 0; }

	//Main loop:
	int iter;
	for(iter=0; iter<p.nIterations && !killFlag; iter++)
	{	//Update search direction:
		if(rdotzPrev)
		{	beta = rdotz/rdotzPrev;
			d *= beta; axpy(1.0, z, d); // d = z + beta*d
		}
		else d = clone(z); //fresh search direction (along gradient)
		//Step:
		Vector w = hessian(d);
		double alpha = rdotz/sync(dot(w,d));
		axpy(alpha, d, state);
		axpy(-alpha, w, r);
		z = precondition(r);
		rdotzPrev = rdotz;
		rdotz = sync(dot(r, z));
		//Print info:
		double rzNorm = sqrt(fabs(rdotz)/p.nDim);
		fprintf(p.fpLog, "%sIter: %3d  sqrt(|r.z|): %12.6le  alpha: %12.6le  beta: %13.6le  t[s]: %9.2lf\n",
			p.linePrefix, iter, rzNorm, alpha, beta, clock_sec()); fflush(p.fpLog);
		//Check convergence:
		if(rzNorm<p.knormThreshold) { fprintf(p.fpLog, "%sConverged sqrt(r.z)<%le\n", p.linePrefix, p.knormThreshold); fflush(p.fpLog); return iter; }
	}
	fprintf(p.fpLog, "%sGradient did not converge within threshold in %d iterations\n", p.linePrefix, iter); fflush(p.fpLog);
	return iter;
}


template<typename Vector> int LinearSolvableIndefinite<Vector>::solve(const Vector& b, const PerturbationParams& p)
{
	switch(p.algorithm) {
		case PerturbationParams::CG: return CG(b, p);
		case PerturbationParams::MINRES: return MINRES(b,p);
		default: return 0;
	}
}

template<typename Vector> int LinearSolvableIndefinite<Vector>::CG(const Vector& b, const PerturbationParams& p)
{	fprintf(p.fpLog, "\nBeginning conjugate gradient minimization.\n");
	
	//Initialize:
	Vector r; hessian(r, state); axpy(1.0, b, r); r *= -1.0; //residual r = -(A.state + b);
	Vector z = clone(r); precondition(z);
	Vector d, w;
	double beta=0.0, rdotzPrev=0.0, rdotz = sync(dot(r, z));

	Vector optstate; double optresidual = rdotz; int optiter = 0;
	int consecErrIncreases = 0;
	
	double rKr;
	{
		Vector Kb = clone(b);
		precondition(Kb);
		rKr = sync(dot(b, Kb));
	}
	
	//Check initial residual
	double relErr = sqrt(fabs(rdotz)/fabs(rKr));
	double relErrPrev = relErr;
	fprintf(p.fpLog, "%sInitial:  ||r||_k/||b||_k: %12.6le\n", p.linePrefix, relErr); fflush(p.fpLog);
	if(relErr<p.residualTol) { fprintf(p.fpLog, "%sConverged ||r||_k/||b||_k<%le\n", p.linePrefix, p.residualTol); fflush(p.fpLog); return 0; }

	//Main loop:
	int iter;
	for(iter=0; iter<p.nIterations && !killFlag; iter++)
	{	//Update search direction:
		if(rdotzPrev)
		{	beta = rdotz/rdotzPrev;
			d *= beta; axpy(1.0, z, d); // d = z + beta*d
		}
		else d = clone(z); //fresh search direction (along gradient)
		//Step:
		hessian(w, d);
		double alpha = rdotz/sync(dot(w,d));
		axpy(alpha, d, state);
		axpy(-alpha, w, r);
		
		if (iter > 1 && iter % 10 == 0 && p.recomputeResidual) {
			hessian(r, state); axpy(1.0, b, r); r *= -1.0;
			logPrintf("Recomputing residual.\n");
		}
		
		z = clone(r);
		precondition(z);
		rdotzPrev = rdotz;
		rdotz = sync(dot(r, z));
		
		//Print info:
		double relErr = sqrt(fabs(rdotz)/fabs(rKr));
		//double relErr = sqrt(fabs(sync(dot(z,z)))/fabs(rKr));
		fprintf(p.fpLog, "%sIter: %3d  ||r||_k/||b||_k: %12.6le  alpha: %12.6le  beta: %13.6le  t[s]: %9.2lf\n",
			p.linePrefix, iter, relErr, alpha, beta, clock_sec()); fflush(p.fpLog);

		//Check convergence:
		if(relErr<p.residualTol) { fprintf(p.fpLog, "%sConverged ||r||_k/||b||_k<%le\n", p.linePrefix, p.residualTol); fflush(p.fpLog); return iter; }
		
		if(relErr > relErrPrev) {
			if (consecErrIncreases == 0 && rdotzPrev < optresidual) {
				axpy(-alpha, d, state);
				optstate = state;
				optiter = iter-1;
				optresidual = rdotzPrev;
				axpy(alpha, d, state);
			}
			consecErrIncreases++;
		}
		else
			consecErrIncreases = 0;
		
		if (consecErrIncreases >= 3 && !p.CGBypass) { logPrintf("Terminating solver as relative error has increased 3 times in a row.\n"); ; fflush(p.fpLog); break; }
		relErrPrev = relErr;
	}
	fprintf(p.fpLog, "%sGradient did not converge within threshold in %d iterations\n", p.linePrefix, iter); fflush(p.fpLog);
	if (optiter > 0) {
		state = optstate;
		fprintf(p.fpLog, "Setting final state to output of iteration %d with |r||_k/||b||_k = %g.\n", optiter, sqrt(optresidual)/fabs(rKr));
	}
	return iter;
}



template<typename Vector> int LinearSolvableIndefinite<Vector>::MINRES(const Vector& b, const PerturbationParams& p)
{	fprintf(p.fpLog, "\nBeginning MINRES solve.\n");
	Vector r = clone(state);
	
	Ksqrt(r); hessian(r, r); r *= -1.0; axpy(-1.0, b, r); Ksqrt(r); //r = -Ksqrt.b - Ksqrt.A.Ksqrt.x
	double rho=sqrt(dot(r,r));
	Vector v = clone(r); v *= (1/rho);
	double beta=0; double betatilde=0; double c=-1; double s=0;
	Vector vold=clone(state); Vector w=clone(state); Vector wtt=clone(v);
	Vector vhat, wt;
	
	int consecErrIncreases = 0;
	
	double rKr;
	{
		Vector Kb = clone(b);
		precondition(Kb);
		rKr = sync(dot(b, Kb));
	}
	double relErr = rho/sqrt(fabs(rKr));
	double relErrPrev = relErr;
	
	int iter;
	for(iter=0; iter<p.nIterations && !killFlag; iter++)
	{
		vhat = clone(v);
		Ksqrt(vhat); hessian(vhat, vhat); Ksqrt(vhat);
		axpy(-beta, vold, vhat);
		
		double alpha=dot(v,vhat);
		axpy(-alpha, v, vhat);
		beta=sqrt(dot(vhat,vhat));
		vold=clone(v);
		v = clone(vhat); v *= (1/beta);
		double l1=s*alpha-c*betatilde;
		double l2=s*beta;
		double alphatilde=-s*betatilde-c*alpha;
		betatilde=c*beta;
		double l0=sqrt(alphatilde*alphatilde+beta*beta);
		c=alphatilde/l0;
		s=beta/l0;
		wt=clone(wtt); axpy(-l1, w, wt);
		wtt=clone(v); axpy(-l2,w, wtt);
		w = clone(wt); w *= (1/l0);
		axpy(rho*c,w,state);
		rho=s*rho;
		
		double relErr = rho/sqrt(fabs(rKr));
		fprintf(p.fpLog, "%sIter: %3d  ||r||_k/||b||_k: %12.6le t[s]: %9.2lf\n",
			p.linePrefix, iter, relErr, clock_sec()); fflush(p.fpLog);
		if (relErr < p.residualTol) { fprintf(p.fpLog, "%sConverged ||r||_k/||b||_k<%le\n", p.linePrefix, p.residualTol); fflush(p.fpLog); Ksqrt(state); return iter; }
		
		if((relErrPrev - relErr)/relErr < p.residualDiffThreshold)
			consecErrIncreases++;
		else
			consecErrIncreases = 0;
		
		if (consecErrIncreases >= 2) { logPrintf("Terminating solver since (||r_prev||-||r||)/||r||<%g for two iterations in a row.\n", p.residualTol); fflush(p.fpLog); break; }
		relErrPrev = relErr;
	}
	
	Ksqrt(state);
	fprintf(p.fpLog, "%sGradient did not converge within threshold in %d iterations\n", p.linePrefix, iter); fflush(p.fpLog);
	return iter;
}

//--- Implementation of EdiffCheck ---
inline EdiffCheck::EdiffCheck(unsigned nDiff, double threshold) : nDiff(nDiff), threshold(fabs(threshold)) {}
inline bool EdiffCheck::checkConvergence(double E)
{	if(!size()) { push_back(E); return false; } //first element
	if(E >= back()) { clear(); push_back(E); return false; } //energy increased, reset converge list
	//Have atleast one energy difference in list:
	push_back(E);
	if(size()==nDiff+2) pop_front(); //discard old unneeded elements
	if(size()==nDiff+1)
	{	for(unsigned i=0; i<nDiff; i++)
			if(at(i+1) < at(i)-threshold)
				return false;
		return true;
	}
	else return false;
}

//!@endcond
#endif // JDFTX_CORE_MINIMIZE_H
