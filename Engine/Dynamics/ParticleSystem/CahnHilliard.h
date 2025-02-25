#pragma once


#include "Core/Array/Array.h"

#include "Framework/Framework/Base.h"
#include "Framework/Framework/Node.h"
#include "Framework/Framework/NumericalModel.h"
#include "Framework/Framework/Module.h"
#include "Framework/Framework/FieldVar.h"
#include "Framework/Framework/FieldArray.h"
#include "Framework/Topology/FieldNeighbor.h"

namespace Physika
{
    template<typename TDataType, int PhaseCount = 2>
	class CahnHilliard : public Module
    {
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
        using PhaseVector = Vector<Real, PhaseCount>;

		CahnHilliard();
		~CahnHilliard() override;

		bool initializeImpl() override;

		bool integrate();

		std::string getModuleType() override { return "NumericalIntegrator"; }

        NeighborField<int> m_neighborhood;

		VarField<Real> m_particleVolume;
		VarField<Real> m_smoothingLength;

		VarField<Real> m_degenerateMobilityM;

        DeviceArrayField<Coord> m_position;
		DeviceArrayField<PhaseVector> m_chemicalPotential;
        DeviceArrayField<PhaseVector> m_concentration;
	};
#ifdef PRECISION_FLOAT
	template class CahnHilliard<DataType3f>;
#else
	template class CahnHilliard<DataType3d>;
#endif
}

