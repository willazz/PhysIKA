#pragma once
#include "ParticleSystem.h"

namespace Physika
{
	/*!
	*	\class	ParticleFluid
	*	\brief	Position-based fluids.
	*
	*	This class implements a position-based fluid solver.
	*	Refer to Macklin and Muller's "Position Based Fluids" for details
	*
	*/
	template<typename TDataType>
	class ParticleFluid : public ParticleSystem<TDataType>
	{
		DECLARE_CLASS_1(ParticleFluid, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		ParticleFluid(std::string name = "default");
		virtual ~ParticleFluid();

		void advance(Real dt) override;
	private:
	};

#ifdef PRECISION_FLOAT
	template class ParticleFluid<DataType3f>;
#else
	template class ParticleFluid<DataType3d>;
#endif
}