#pragma once
#include "Framework/Framework/ModuleController.h"
#include "Core/Platform.h"

namespace Physika
{

class AnimationController : public ControllerModule
{
	DECLARE_CLASS(AnimationController)

public:
	AnimationController();
	virtual ~AnimationController();

	bool execute() override;
private:

};

}
