#pragma once

#include "Framework/Framework/ModuleVisual.h"
#include "Rendering/PointRender.h"
#include "Rendering/LineRender.h"
#include "Rendering/TriangleRender.h"
#include "Framework/Framework/FieldArray.h"
#include "Framework/Framework/FieldVar.h"

namespace Physika
{
	class PointRenderModule : public VisualModule
	{
		DECLARE_CLASS(PointRenderModule)
	public:
		PointRenderModule();
		~PointRenderModule();

		enum RenderMode {
			POINT = 0,
			SPRITE
		};

		void display() override;
		void setRenderMode(RenderMode mode);
		void setColor(Vector3f color);

		void setColorRange(float min, float max);

	public:
		VarField<float> m_minIndex;
		VarField<float> m_maxIndex;

		DeviceArrayField<Vector3f> m_vecIndex;
		DeviceArrayField<Vector3f> m_scalarIndex;

	protected:
		bool  initializeImpl() override;

		void updateRenderingContext() override;

	private:
		RenderMode m_mode;
		Vector3f m_color;

		DeviceArray<glm::vec3> m_colorArray;

// 		std::shared_ptr<PointRenderUtil> point_render_util;
// 		std::shared_ptr<PointRenderTask> point_render_task;
		std::shared_ptr<PointRender> m_pointRender;
		std::shared_ptr<LineRender> m_lineRender;
		std::shared_ptr<TriangleRender> m_triangleRender;
	};

}