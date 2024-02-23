/*-------------------------------------------------------------------
Copyright 2011 Ravishankar Sundararaman, Kendra Letchworth Weaver, Deniz Gunceler

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

#include <electronic/Everything.h>
#include <fluid/NonlinearPCM.h>
#include <fluid/LinearPCM.h>
#include <fluid/PCM_internal.h>
#include <core/ScalarFieldIO.h>
#include <core/Util.h>


//Initialize Kkernel to square-root of the inverse kinetic operator

inline void setPreconditioner(int i, double Gsq, double* preconditioner, double epsBulk, double kappaSq)
{	preconditioner[i] = (Gsq || kappaSq) ? 1./(epsBulk*Gsq + kappaSq) : 0.;
}


NonlinearPCM::NonlinearPCM(const Everything& e, const FluidSolverParams& fsp)
: PCM(e, fsp), NonlinearCommon(fsp, epsBulk)
{
	//Initialize preconditioner:
	preconditioner = std::make_shared<RealKernel>(gInfo);
	applyFuncGsq(gInfo, setPreconditioner, preconditioner->data(), epsBulk, k2factor);
}

NonlinearPCM::~NonlinearPCM()
{
}

void NonlinearPCM::loadState(const char* filename)
{	ScalarField Iphi(ScalarFieldData::alloc(gInfo));
	loadRawBinary(Iphi, filename); //saved data is in real space
	phiTot = J(Iphi);
}

void NonlinearPCM::saveState(const char* filename) const
{	if(mpiWorld->isHead()) saveRawBinary(I(phiTot), filename); //saved data is in real space
}

void NonlinearPCM::dumpDensities(const char* filenamePattern) const
{	PCM::dumpDensities(filenamePattern);
	die("Not yet implemented.\n");
}

void NonlinearPCM::minimizeFluid()
{	//Info:
	logPrintf("\tNonlinear fluid (bulk dielectric constant: %g) occupying %lf of unit cell\n",
		epsBulk, integral(shape[0])/gInfo.detR);
	if(k2factor)
		logPrintf("\tNonlinear screening (bulk screening length: %g bohrs) occupying %lf of unit cell\n",
		sqrt(epsBulk/k2factor), integral(shape.back())/gInfo.detR);
	logFlush();

	//Minimize:
	minimize(e.fluidMinParams);
	logPrintf("\tNonlinear solve completed after %d iterations at t[s]: %9.2lf\n", iterLast, clock_sec());
}

void NonlinearPCM::step(const ScalarFieldTilde& dir, double alpha)
{	::axpy(alpha, dir, phiTot);
}

double NonlinearPCM::compute(ScalarFieldTilde* grad, ScalarFieldTilde* Kgrad)
{
	ScalarField A; nullToZero(A, gInfo);
	VectorField Dphi = I(gradient(phiTot)), A_Dphi_null;
	VectorField& A_Dphi = grad ? Dphi : A_Dphi_null; //Retrieve gradient in place (since Dphi no longer needed)
	callPref(dielectricEval->apply)(gInfo.nr, dielEnergyLookup,
		shape[0]->dataPref(), Dphi.const_dataPref(), A->dataPref(), A_Dphi.dataPref(), NULL);
	
	ScalarField A_phi;
	if(screeningEval)
	{	ScalarField phi = I(phiTot);
		if(grad) nullToZero(A_phi, gInfo);
		callPref(screeningEval->apply)(gInfo.nr, ionEnergyLookup,
			shape.back()->dataPref(), phi->dataPref(),
			A->dataPref(), grad ? A_phi->dataPref() : NULL, NULL);
	}
	
	if(grad)
	{	ScalarFieldTilde A_phiTilde = -divergence(J(A_Dphi));
		if(A_phi) A_phiTilde += J(A_phi);
		*grad = O(A_phiTilde - rhoExplicitTilde);
		if(Kgrad)
		{	*Kgrad = (*preconditioner) * (*grad);
		}
	}
	return A0 + integral(A) - dot(rhoExplicitTilde, O(phiTot));
}

bool NonlinearPCM::report(int iter)
{	iterLast = iter;
	return false;
}


void NonlinearPCM::set_internal(const ScalarFieldTilde& rhoExplicitTilde, const ScalarFieldTilde& nCavityTilde)
{	//Store the explicit system charge:
	this->rhoExplicitTilde = rhoExplicitTilde; zeroNyquist(this->rhoExplicitTilde);
	A0 = 0.5 * dot(rhoExplicitTilde, O(coulomb(rhoExplicitTilde)));
	
	//Update cavity:
	this->nCavity = I(nCavityTilde + getFullCore());
	updateCavity();
	
	//Initialize the state if it hasn't been loaded:
	if(!phiTot) nullToZero(phiTot, gInfo);
}

double NonlinearPCM::get_Adiel_and_grad_internal(ScalarFieldTilde& Adiel_rhoExplicitTilde, ScalarFieldTilde& Adiel_nCavityTilde, IonicGradient* extraForces, matrix3<>* Adiel_RRT) const
{
	EnergyComponents& Adiel = ((NonlinearPCM*)this)->Adiel;
	ScalarFieldArray Adiel_shape; nullToZero(Adiel_shape, gInfo, shape.size());
	ScalarField F; nullToZero(F, gInfo); //Hessian energy contributions

	//Compute dielectric contributions:
	{	const VectorField Dphi = I(gradient(phiTot));
		VectorField F_Dphi; vector3<double*> F_DphiData;
		if(Adiel_RRT)
		{	nullToZero(F_Dphi, gInfo); //only needed for stress
			F_DphiData = F_Dphi.dataPref();
		}
		callPref(dielectricEval->apply)(gInfo.nr, dielEnergyLookup,
			shape[0]->dataPref(), Dphi.const_dataPref(), F->dataPref(),
			F_DphiData, Adiel_shape[0]->dataPref());
		if(Adiel_RRT) *Adiel_RRT += gInfo.dV * dotOuter(F_Dphi, Dphi);
	}

	//Compute screening contributions
	if(screeningEval)
	{	const ScalarField phi = I(phiTot);
		callPref(screeningEval->apply)(gInfo.nr, ionEnergyLookup,
			shape.back()->dataPref(), phi->dataPref(), F->dataPref(),
			NULL, Adiel_shape.back()->dataPref());
	}

	//Compute the energy:
	ScalarFieldTilde phiExplicitTilde = coulomb(rhoExplicitTilde);
	Adiel["Electrostatic"] = -integral(F) + dot(phiTot - 0.5*phiExplicitTilde, O(rhoExplicitTilde));
	if(Adiel_RRT)
	{	*Adiel_RRT += Adiel["Electrostatic"] * matrix3<>(1,1,1) //volume contribution
			- 0.5*coulombStress(rhoExplicitTilde, rhoExplicitTilde); //through coulomb in phiExt
	}
	
	//Derivatives w.r.t electronic charge and density:
	Adiel_rhoExplicitTilde = phiTot - phiExplicitTilde;
	ScalarField Adiel_nCavity;
	propagateCavityGradients(Adiel_shape, Adiel_nCavity, Adiel_rhoExplicitTilde, extraForces, Adiel_RRT);
	Adiel_nCavityTilde = J(Adiel_nCavity);
	accumExtraForces(extraForces, Adiel_nCavityTilde);
	return Adiel;
}

/*
//Utility functions to extract/set the members of a MuEps
inline ScalarField& getMuPlus(ScalarFieldMuEps& X) { return X[0]; }
inline const ScalarField& getMuPlus(const ScalarFieldMuEps& X) { return X[0]; }
inline ScalarField& getMuMinus(ScalarFieldMuEps& X) { return X[1]; }
inline const ScalarField& getMuMinus(const ScalarFieldMuEps& X) { return X[1]; }
inline VectorField getEps(ScalarFieldMuEps& X) { return VectorField(&X[2]); }
inline const VectorField getEps(const ScalarFieldMuEps& X) { return VectorField(&X[2]); }
inline void setMuEps(ScalarFieldMuEps& mueps, ScalarField muPlus, ScalarField muMinus, VectorField eps) { mueps[0]=muPlus; mueps[1]=muMinus; for(int k=0; k<3; k++) mueps[k+2]=eps[k]; }

inline void setMetric(int i, double Gsq, double qMetricSq, double* metric)
{	metric[i] = qMetricSq ? Gsq / (qMetricSq + Gsq) : 1.;
}

void NonlinearPCM::set_internal(const ScalarFieldTilde& rhoExplicitTilde, const ScalarFieldTilde& nCavityTilde)
{	
	bool setPhiFromState = false; //whether to set linearPCM phi from state (first time in SCF version when state has been read in)
	
	//Initialize state if required:
	if(!state)
	{	logPrintf("Initializing state of NonlinearPCM using a similar LinearPCM: "); logFlush();
		FILE*& fpLog = ((MinimizeParams&)e.fluidMinParams).fpLog;
		fpLog = nullLog; //disable iteration log from LinearPCM
		linearPCM->minimizeFluid();
		fpLog = globalLog; //restore usual iteration log
		logFlush();
		//Guess nonlinear states based on the electrostatic potential of the linear version:
		//mu:
		ScalarField mu;
		if(screeningEval && screeningEval->linear)
		{	mu = (-ionZ/fsp.T) * I(linearPCM->state);
			mu -= integral(mu)/gInfo.detR; //project out G=0
		}
		else initZero(mu, gInfo); //initialization logic does not work well with hard sphere limit
		//eps:
		VectorField eps = (-pMol/fsp.T) * I(gradient(linearPCM->state));
		ScalarField E = sqrt(eps[0]*eps[0] + eps[1]*eps[1] + eps[2]*eps[2]);
		ScalarField Ecomb = 0.5*((dielectricEval->alpha-3.) + E);
		ScalarField epsByE = inv(E) * (Ecomb + sqrt(Ecomb*Ecomb + 3.*E));
		eps *= epsByE; //enhancement due to correlations
		//collect:
		setMuEps(state, mu, clone(mu), eps);
	}
	
	this->rhoExplicitTilde = rhoExplicitTilde; zeroNyquist(this->rhoExplicitTilde);
	this->nCavity = I(nCavityTilde + getFullCore());
	
	updateCavity();
	
	if(setPhiFromState)
	{	ScalarFieldMuEps gradUnused;
		ScalarFieldTilde phiFluidTilde;
		(*this)(state, gradUnused, &phiFluidTilde);
		linearPCM->state = phiFluidTilde + coulomb(rhoExplicitTilde);
	}
}

double NonlinearPCM::operator()(const ScalarFieldMuEps& state, ScalarFieldMuEps& Adiel_state,
	ScalarFieldTilde* Adiel_rhoExplicitTilde, ScalarFieldTilde* Adiel_nCavityTilde, IonicGradient* forces, matrix3<>* Adiel_RRT) const
{
	EnergyComponents& Adiel = ((NonlinearPCM*)this)->Adiel;
	ScalarFieldArray Adiel_shape; if(Adiel_nCavityTilde) nullToZero(Adiel_shape, gInfo, shape.size());
	
	ScalarFieldTilde rhoFluidTilde;
	ScalarField muPlus, muMinus, Adiel_muPlus, Adiel_muMinus;
	initZero(Adiel_muPlus, gInfo); initZero(Adiel_muMinus, gInfo);
	double mu0 = 0., Qexp = 0., Adiel_Qexp = 0.;
	if(screeningEval)
	{	//Get neutrality Lagrange multiplier:
		muPlus = getMuPlus(state);
		muMinus = getMuMinus(state);
		Qexp = integral(rhoExplicitTilde);
		mu0 = screeningEval->neutralityConstraint(muPlus, muMinus, shape.back(), Qexp);
		//Compute ionic free energy and bound charge
		ScalarField Aout, rhoIon;
		initZero(Aout, gInfo);
		initZero(rhoIon, gInfo);
		callPref(screeningEval->freeEnergy)(gInfo.nr, mu0, muPlus->dataPref(), muMinus->dataPref(), shape.back()->dataPref(),
			rhoIon->dataPref(), Aout->dataPref(), Adiel_muPlus->dataPref(), Adiel_muMinus->dataPref(), Adiel_shape.size() ? Adiel_shape.back()->dataPref() : 0);
		Adiel["Akappa"] = integral(Aout);
		rhoFluidTilde += J(rhoIon); //include bound charge due to ions
	}
	
	//Compute the dielectric free energy and bound charge:
	VectorField eps = getEps(state), p, Adiel_eps;
	{	ScalarField Aout;
		initZero(Aout, gInfo);
		nullToZero(p, gInfo);
		nullToZero(Adiel_eps, gInfo);
		callPref(dielectricEval->freeEnergy)(gInfo.nr, eps.const_dataPref(), shape[0]->dataPref(),
			p.dataPref(), Aout->dataPref(), Adiel_eps.dataPref(), Adiel_shape.size() ? Adiel_shape[0]->dataPref() : 0);
		Adiel["Aeps"] = integral(Aout);
		rhoFluidTilde -= divergence(J(p)); //include bound charge due to dielectric
		if(!Adiel_RRT) p = 0; //only need later for lattice derivative
	} //scoped to automatically deallocate temporaries
	
	//Compute the electrostatic terms:
	ScalarFieldTilde phiFluidTilde = coulomb(rhoFluidTilde);
	ScalarFieldTilde phiExplicitTilde = coulomb(rhoExplicitTilde);
	Adiel["Coulomb"] = dot(rhoFluidTilde, O(0.5*phiFluidTilde + phiExplicitTilde));
	if(Adiel_RRT)
		*Adiel_RRT += matrix3<>(1,1,1)*(Adiel["Coulomb"] + Adiel["Aeps"] + Adiel["Akappa"])
			+ coulombStress(rhoFluidTilde, 0.5*rhoFluidTilde+rhoExplicitTilde);
	
	if(screeningEval)
	{	//Propagate gradients from rhoIon to mu, shape
		ScalarField Adiel_rhoIon = I(phiFluidTilde+phiExplicitTilde);
		callPref(screeningEval->convertDerivative)(gInfo.nr, mu0, muPlus->dataPref(), muMinus->dataPref(), shape.back()->dataPref(),
			Adiel_rhoIon->dataPref(), Adiel_muPlus->dataPref(), Adiel_muMinus->dataPref(), Adiel_shape.size() ? Adiel_shape.back()->dataPref() : 0);
		//Propagate gradients from mu0 to mu, shape, Qexp:
		double Adiel_mu0 = integral(Adiel_muPlus) + integral(Adiel_muMinus), mu0_Qexp;
		ScalarField mu0_muPlus, mu0_muMinus, mu0_shape;
		screeningEval->neutralityConstraint(muPlus, muMinus, shape.back(), Qexp, &mu0_muPlus, &mu0_muMinus, &mu0_shape, &mu0_Qexp);
		Adiel_muPlus += Adiel_mu0 * mu0_muPlus;
		Adiel_muMinus += Adiel_mu0 * mu0_muMinus;
		if(Adiel_shape.size()) Adiel_shape.back() += Adiel_mu0 * mu0_shape;
		Adiel_Qexp = Adiel_mu0 * mu0_Qexp;
	}
	
	//Propagate gradients from p to eps, shape
	{	VectorField Adiel_p = I(gradient(phiFluidTilde+phiExplicitTilde)); //Because dagger(-divergence) = gradient
		callPref(dielectricEval->convertDerivative)(gInfo.nr, eps.const_dataPref(), shape[0]->dataPref(),
			Adiel_p.const_dataPref(), Adiel_eps.dataPref(), Adiel_shape.size() ? Adiel_shape[0]->dataPref() : 0);
		if(Adiel_RRT) *Adiel_RRT -= gInfo.dV * dotOuter(p, Adiel_p);
	}
	
	//Optional outputs:
	if(Adiel_rhoExplicitTilde)
	{	if(fsp.nonlinearSCF && !useGummel())
		{	//Non-variational version (from inner solve):
			*Adiel_rhoExplicitTilde = linearPCM->state - phiExplicitTilde;
		}
		else
		{	//Variational version (from bound charge with neutrality constraint):
			*Adiel_rhoExplicitTilde = phiFluidTilde;
			(*Adiel_rhoExplicitTilde)->setGzero(Adiel_Qexp);
		}
	}
	if(Adiel_nCavityTilde)
	{	ScalarField Adiel_nCavity;
		propagateCavityGradients(Adiel_shape, Adiel_nCavity, *Adiel_rhoExplicitTilde, forces, Adiel_RRT);
		*Adiel_nCavityTilde = J(Adiel_nCavity);
	}
	
	//Collect energy and gradient pieces:
	setMuEps(Adiel_state, Adiel_muPlus, Adiel_muMinus, Adiel_eps);
	Adiel_state *= gInfo.dV; //converts variational derivative to total derivative
	return Adiel;
}

void NonlinearPCM::minimizeFluid()
{	if(fsp.nonlinearSCF)
	{	clearState();
		Pulay<ScalarFieldTilde>::minimize(compute(0,0));
	}
	else
		Minimizable<ScalarFieldMuEps>::minimize(e.fluidMinParams);
}

void NonlinearPCM::loadState(const char* filename)
{	nullToZero(state, gInfo);
	state.loadFromFile(filename);
}

void NonlinearPCM::saveState(const char* filename) const
{	if(mpiWorld->isHead()) state.saveToFile(filename);
}

double NonlinearPCM::get_Adiel_and_grad_internal(ScalarFieldTilde& Adiel_rhoExplicitTilde, ScalarFieldTilde& Adiel_nCavityTilde, IonicGradient* extraForces, matrix3<>* Adiel_RRT) const
{	ScalarFieldMuEps Adiel_state;
	double A = (*this)(state, Adiel_state, &Adiel_rhoExplicitTilde, &Adiel_nCavityTilde, extraForces, Adiel_RRT);
	accumExtraForces(extraForces, Adiel_nCavityTilde);
	return A;
}

void NonlinearPCM::step(const ScalarFieldTilde& dir, double alpha)
{	::axpy(alpha, dir, phiTot);
}

double NonlinearPCM::compute(ScalarFieldMuEps* grad, ScalarFieldMuEps* Kgrad)
{	ScalarFieldMuEps gradUnused;
	double E = (*this)(state, grad ? *grad : gradUnused);
	//Compute preconditioned gradient:
	if(Kgrad)
	{	const ScalarFieldMuEps& in = grad ? *grad : gradUnused;
		double dielPrefac = 1./(gInfo.dV * dielectricEval->NT);
		double ionsPrefac = screeningEval ? 1./(gInfo.dV * screeningEval->NT) : 0.;
		setMuEps(*Kgrad,
			ionsPrefac * I(preconditioner*J(getMuPlus(in))),
			ionsPrefac * I(preconditioner*J(getMuMinus(in))),
			dielPrefac * getEps(in));
	}
	return E;
}


void NonlinearPCM::dumpDensities(const char* filenamePattern) const
{	PCM::dumpDensities(filenamePattern);

	//Output dielectric bound charge:
	string filename;
	{	ScalarField Aout; initZero(Aout, gInfo);
		VectorField p; nullToZero(p, gInfo);
		VectorField Adiel_eps; nullToZero(Adiel_eps, gInfo);
		callPref(dielectricEval->freeEnergy)(gInfo.nr, getEps(state).const_dataPref(), shape[0]->dataPref(),
			p.dataPref(), Aout->dataPref(), Adiel_eps.dataPref(), 0);
		ScalarField rhoDiel = -divergence(p); //include bound charge due to dielectric
		FLUID_DUMP(rhoDiel, "RhoDiel");
	}
	
	//Output ionic bound charge (if any):
	if(screeningEval)
	{	ScalarField Nplus, Nminus;
		{	ScalarField muPlus = getMuPlus(state);
			ScalarField muMinus = getMuMinus(state);
			double Qexp = integral(rhoExplicitTilde);
			double mu0 = screeningEval->neutralityConstraint(muPlus, muMinus, shape.back(), Qexp);
			Nplus = ionNbulk * shape.back() * (fsp.linearScreening ? 1.+(mu0+muPlus) : exp(mu0+muPlus));
			Nminus = ionNbulk * shape.back() * (fsp.linearScreening ? 1.-(mu0+muMinus) : exp(-(mu0+muMinus)));
		}
		FLUID_DUMP(Nplus, "N+");
		FLUID_DUMP(Nminus, "N-");
		FLUID_DUMP(ionZ*(Nplus-Nminus), "RhoIon");
	}
}

//--------- Interface for Pulay<ScalarFieldTilde> ---------

double NonlinearPCM::cycle(double dEprev, std::vector<double>& extraValues)
{	//Update epsilon / kappaSq based on current phi:
	phiToState(false);
	//Inner linear solve
	FILE*& fpLog = ((MinimizeParams&)e.fluidMinParams).fpLog;
	fpLog = nullLog; //disable iteration log from LinearPCM
	linearPCM->minimizeFluid();
	fpLog = globalLog; //restore usual iteration log
	//Update state from new phi:
	phiToState(true);
	return compute(0,0);
}


void NonlinearPCM::readVariable(ScalarFieldTilde& X, FILE* fp) const
{	nullToZero(X, gInfo);
	loadRawBinary(X, fp);
}

void NonlinearPCM::writeVariable(const ScalarFieldTilde& X, FILE* fp) const
{	saveRawBinary(X, fp);
}

ScalarFieldTilde NonlinearPCM::getVariable() const
{	return clone(linearPCM->state);
}

void NonlinearPCM::setVariable(const ScalarFieldTilde& X)
{	linearPCM->state = clone(X);
}

ScalarFieldTilde NonlinearPCM::precondition(const ScalarFieldTilde& X) const
{	return fsp.scfParams.mixFraction * X;	
}

ScalarFieldTilde NonlinearPCM::applyMetric(const ScalarFieldTilde& X) const
{	return (*metric) * X;
}

void NonlinearPCM::phiToState(bool setState)
{	//Initialize inputs:
	const ScalarField phi = I(linearPCM->state);
	const VectorField Dphi = I(gradient(linearPCM->state));
	//Prepare outputs:
	ScalarField epsilon, kappaSq;
	if(!setState)
	{	nullToZero(epsilon, gInfo);
		if(screeningEval)
			nullToZero(kappaSq, gInfo);
	}
	VectorField eps = getEps(state);
	ScalarField& muPlus = getMuPlus(state);
	ScalarField& muMinus = getMuMinus(state);
	//Calculate eps/mu or epsilon/kappaSq as needed:
	vector3<double*> vecDataUnused(0,0,0); double* dataUnused=0;
	callPref(dielectricEval->phiToState)(gInfo.nr, Dphi.dataPref(), shape[0]->dataPref(), gLookup, setState,
		setState ? eps.dataPref() : vecDataUnused,
		setState ? dataUnused : epsilon->dataPref() );
	if(screeningEval)
		callPref(screeningEval->phiToState)(gInfo.nr, phi->dataPref(), shape.back()->dataPref(), xLookup, setState,
			setState ? muPlus->dataPref() : dataUnused,
			setState ? muMinus->dataPref() : dataUnused, 
			setState ? dataUnused : kappaSq->dataPref() );
	//Save to global state or linearPCM as required:
	if(setState)
		setMuEps(state, muPlus, muMinus, eps);
	else
		linearPCM->override(epsilon, kappaSq);
}
*/
