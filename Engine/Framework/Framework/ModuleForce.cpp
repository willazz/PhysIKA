#include "ModuleForce.h"
#include "Framework/Framework/Node.h"

namespace Physika
{
IMPLEMENT_CLASS(ForceModule)

ForceModule::ForceModule()
	: Module()
	, m_forceID(MechanicalState::force())
	, m_torqueID(MechanicalState::torque())
{
}

ForceModule::~ForceModule()
{
}

}