#include <iostream>
#include <memory>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <GL/glew.h>
#include <GL/freeglut.h>

#include "GUI/GlutGUI/GLApp.h"

#include "Framework/Framework/SceneGraph.h"
#include "Framework/Topology/PointSet.h"
#include "Framework/Framework/Log.h"

#include "Dynamics/ParticleSystem/ParticleElastoplasticBody.h"
#include "Dynamics/ParticleSystem/StaticBoundary.h"
#include "Dynamics/RigidBody/RigidBody.h"
#include "Dynamics/ParticleSystem/FractureModule.h"

using namespace std;
using namespace Physika;

void RecieveLogMessage(const Log::Message& m)
{
	switch (m.type)
	{
	case Log::Info:
		cout << ">>>: " << m.text << endl; break;
	case Log::Warning:
		cout << "???: " << m.text << endl; break;
	case Log::Error:
		cout << "!!!: " << m.text << endl; break;
	case Log::User:
		cout << ">>>: " << m.text << endl; break;
	default: break;
	}
}

void CreateScene()
{
	SceneGraph& scene = SceneGraph::getInstance();

	std::shared_ptr<StaticBoundary<DataType3f>> root = scene.createNewScene<StaticBoundary<DataType3f>>();
	root->loadSDF("../Media/bar/bar.sdf", false);
	root->translate(Vector3f(0.2f, 0.2f, 0));
	root->loadCube(Vector3f(0), Vector3f(1), true);

	std::shared_ptr<ParticleElastoplasticBody<DataType3f>> child3 = std::make_shared<ParticleElastoplasticBody<DataType3f>>();
	root->addParticleSystem(child3);
	child3->getRenderModule()->setColor(Vector3f(0, 1, 1));
	child3->setMass(1.0);
  	child3->loadParticles("../Media/bunny/bunny_points.obj");
  	child3->loadSurface("../Media/bunny/bunny_mesh.obj");
	child3->translate(Vector3f(0.3, 0.4, 0.5));
	auto fracture = std::make_shared<FractureModule<DataType3f>>();
	child3->setElastoplasticitySolver(fracture);

	std::shared_ptr<RigidBody<DataType3f>> rigidbody = std::make_shared<RigidBody<DataType3f>>();
	root->addRigidBody(rigidbody);
	rigidbody->loadShape("../Media/bar/bar.obj");
	rigidbody->setActive(false);
	rigidbody->translate(Vector3f(0.2f, 0.2f, 0));
}


int main()
{
	CreateScene();

	Log::setOutput("console_log.txt");
	Log::setLevel(Log::Info);
	Log::setUserReceiver(&RecieveLogMessage);
	Log::sendMessage(Log::Info, "Simulation begin");

	GLApp window;
	window.createWindow(1024, 768);

	window.mainLoop();

	Log::sendMessage(Log::Info, "Simulation end!");
	return 0;
}


