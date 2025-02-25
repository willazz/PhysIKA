#include "PositionBasedFluidModel.h"
#include "Framework/Topology/PointSet.h"
#include "Framework/Framework/Node.h"
#include "DensityPBD.h"
#include "ParticleIntegrator.h"
#include "DensitySummation.h"
#include "ImplicitViscosity.h"
#include "Framework/Framework/MechanicalState.h"
#include "Framework/Mapping/PointSetToPointSet.h"
#include "Framework/Topology/FieldNeighbor.h"
#include "Framework/Topology/NeighborQuery.h"
#include "Dynamics/ParticleSystem/Helmholtz.h"
#include "Dynamics/ParticleSystem/Attribute.h"
#include "Core/Utility.h"

namespace Physika
{
	IMPLEMENT_CLASS_1(PositionBasedFluidModel, TDataType)

	template<typename TDataType>
	PositionBasedFluidModel<TDataType>::PositionBasedFluidModel()
		: NumericalModel()
		, m_restRho(Real(1000))
		, m_pNum(0)
	{
		m_smoothingLength.setValue(Real(0.0075));

		attachField(&m_smoothingLength, "smoothingLength", "Smoothing length", false);

		attachField(&m_position, "position", "Storing the particle positions!", false);
		attachField(&m_velocity, "velocity", "Storing the particle velocities!", false);
		attachField(&m_forceDensity, "force_density", "Storing the particle force densities!", false);
	}

	template<typename TDataType>
	PositionBasedFluidModel<TDataType>::~PositionBasedFluidModel()
	{
		
	}

	template<typename TDataType>
	bool PositionBasedFluidModel<TDataType>::initializeImpl()
	{
		m_nbrQuery = this->getParent()->addComputeModule<NeighborQuery<TDataType>>("neighborhood");
		m_smoothingLength.connect(m_nbrQuery->m_radius);
		m_position.connect(m_nbrQuery->m_position);
		m_nbrQuery->initialize();

		m_pbdModule = this->getParent()->addConstraintModule<DensityPBD<TDataType>>("density_constraint");
		m_smoothingLength.connect(m_pbdModule->m_smoothingLength);
		m_position.connect(m_pbdModule->m_position);
		m_velocity.connect(m_pbdModule->m_velocity);
		m_nbrQuery->m_neighborhood.connect(m_pbdModule->m_neighborhood);
		m_pbdModule->initialize();

		m_integrator = this->getParent()->setNumericalIntegrator<ParticleIntegrator<TDataType>>("integrator");
		m_position.connect(m_integrator->m_position);
		m_velocity.connect(m_integrator->m_velocity);
		m_forceDensity.connect(m_integrator->m_forceDensity);
		m_integrator->initialize();

		m_visModule = this->getParent()->addConstraintModule<ImplicitViscosity<TDataType>>("viscosity");
		m_visModule->setViscosity(Real(1));
		m_smoothingLength.connect(m_visModule->m_smoothingLength);
		m_position.connect(m_visModule->m_position);
		m_velocity.connect(m_visModule->m_velocity);
		m_nbrQuery->m_neighborhood.connect(m_visModule->m_neighborhood);
		m_visModule->initialize();

		return true;
	}

	template<typename TDataType>
	void PositionBasedFluidModel<TDataType>::step(Real dt)
	{
		Node* parent = getParent();
		if (parent == NULL)
		{
			Log::sendMessage(Log::Error, "Parent not set for ParticleSystem!");
			return;
		}
		m_integrator->begin();

		m_nbrQuery->compute();
		m_integrator->integrate();
		
		m_pbdModule->constrain();

		m_visModule->constrain();
		
		m_integrator->end();
	}

	template<typename TDataType>
	void PositionBasedFluidModel<TDataType>::setIncompressibilitySolver(std::shared_ptr<ConstraintModule> solver)
	{
		if (!m_incompressibilitySolver)
		{
			getParent()->deleteConstraintModule(m_incompressibilitySolver);
		}
		m_incompressibilitySolver = solver;
		getParent()->addConstraintModule(solver);
	}


	template<typename TDataType>
	void PositionBasedFluidModel<TDataType>::setViscositySolver(std::shared_ptr<ConstraintModule> solver)
	{
		if (!m_viscositySolver)
		{
			getParent()->deleteConstraintModule(m_viscositySolver);
		}
		m_viscositySolver = solver;
		getParent()->addConstraintModule(solver);
	}



	template<typename TDataType>
	void PositionBasedFluidModel<TDataType>::setSurfaceTensionSolver(std::shared_ptr<ForceModule> solver)
	{
		if (!m_surfaceTensionSolver)
		{
			getParent()->deleteForceModule(m_surfaceTensionSolver);
		}
		m_surfaceTensionSolver = solver;
		getParent()->addForceModule(m_surfaceTensionSolver);
	}

}