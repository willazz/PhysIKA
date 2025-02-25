#include "Peridynamics.h"
#include "Framework/Framework/DeviceContext.h"
#include "Framework/Framework/MechanicalState.h"
#include "Framework/Framework/Node.h"
#include "Framework/Topology/PointSet.h"
#include "Framework/Mapping/PointSetToPointSet.h"
#include "ParticleIntegrator.h"
#include "Framework/Topology/NeighborQuery.h"
#include "Core/Utility.h"

namespace Physika
{
	IMPLEMENT_CLASS_1(Peridynamics, TDataType)

	template<typename TDataType>
	Peridynamics<TDataType>::Peridynamics()
		: NumericalModel()
	{
		attachField(&m_horizon, "horizon", "Supporting radius", false);

		attachField(&m_position, "position", "Storing the particle positions!", false);
		attachField(&m_velocity, "velocity", "Storing the particle velocities!", false);
		attachField(&m_forceDensity, "force_density", "Storing the particle force densities!", false);

		m_horizon.setValue(0.0085);
	}

	template<typename TDataType>
	bool Peridynamics<TDataType>::initializeImpl()
	{
		if (!isAllFieldsReady())
		{
			std::cout << "Exception: " << std::string("Peridynamics's fields are not fully initialized!") << "\n";
			return false;
		}

		m_integrator = std::make_shared<ParticleIntegrator<TDataType>>();
		m_position.connect(m_integrator->m_position);
		m_velocity.connect(m_integrator->m_velocity);
		m_forceDensity.connect(m_integrator->m_forceDensity);
		m_integrator->initialize();

		m_nbrQuery = std::make_shared<NeighborQuery<TDataType>>();
		m_horizon.connect(m_nbrQuery->m_radius);
		m_position.connect(m_nbrQuery->m_position);
		m_nbrQuery->initialize();
		m_nbrQuery->compute();

		m_elasticity = std::make_shared<ElasticityModule<TDataType>>();
		m_position.connect(m_elasticity->m_position);
		m_velocity.connect(m_elasticity->m_velocity);
		m_horizon.connect(m_elasticity->m_horizon);
		m_nbrQuery->m_neighborhood.connect(m_elasticity->m_neighborhood);
		m_elasticity->initialize();

		m_nbrQuery->setParent(getParent());
		m_integrator->setParent(getParent());
		m_elasticity->setParent(getParent());

		return true;
	}

	template<typename TDataType>
	void Peridynamics<TDataType>::step(Real dt)
	{
		Node* parent = getParent();
		if (parent == NULL)
		{
			Log::sendMessage(Log::Error, "Parent not set for ParticleSystem!");
			return;
		}

		m_integrator->begin();

		m_integrator->integrate();

		m_elasticity->constrain();

		m_integrator->end();
	}
}