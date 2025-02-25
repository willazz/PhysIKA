#pragma once
#include <iostream>
#include "Core/Typedef.h"
#include "Field.h"
#include "Base.h"
#include "Framework/Framework/Log.h"
#include "Core/Array/MemoryManager.h"

namespace Physika {

/*!
*	\class	Variable
*	\brief	Variables of build-in data types.
*/
template<typename T>
class VarField : public Field
{
public:
	typedef T VarType;
	typedef T FieldType;

	VarField();
	VarField(T value);
	VarField(T value, std::string name, std::string description);

public:
	~VarField() override;

	size_t getElementCount() override { return 1; }
	const std::string getTemplateName() override { return std::string(typeid(T).name()); }
	const std::string getClassName() override { return std::string("Variable"); }
		
	T& getValue();
	void setValue(T val);

	inline std::shared_ptr<T> getReference();

//	void reset() override;

	bool isEmpty() override {
		return getReference() == nullptr;
	}

	bool connect(VarField<T>& field2);

private:
	std::shared_ptr<T> m_data = nullptr;
};

template<typename T>
VarField<T>::VarField()
	: Field("", "")
{
}

template<typename T>
Physika::VarField<T>::VarField(T value)
	: Field("", "")
{
	m_data = std::make_shared<T>(value);
}

template<typename T>
VarField<T>::VarField(T value, std::string name, std::string description)
	: Field(name, description)
{
	m_data = std::make_shared<T>(value);
}

template<typename T>
VarField<T>::~VarField()
{
};


// template<typename T>
// void Physika::VarField<T>::reset()
// {
// 	*m_data = T(0);
// }

template<typename T>
T& VarField<T>::getValue()
{
	return *(getReference());
}


template<typename T>
void VarField<T>::setValue(T val)
{
	std::shared_ptr<T> data = getReference();
	if (data == nullptr)
	{
		m_data = std::make_shared<T>(val);
	}
	else
	{
		*data = val;
	}
}

template<typename T>
std::shared_ptr<T> VarField<T>::getReference()
{
	Field* source = this->getSource();
	if (source == nullptr)
	{
		return m_data;
	}
	else
	{
		VarField<T>* var = dynamic_cast<VarField<T>*>(source);
		if (var != nullptr)
		{
			return var->getReference();
		}
		else
		{
			return nullptr;
		}
	}
}

template<typename T>
bool VarField<T>::connect(VarField<T>& field2)
{
// 	if (this->isEmpty())
// 	{
// 		Log::sendMessage(Log::Warning, "The parent field " + this->getObjectName() + " is empty!");
// 		return false;
// 	}
	field2.setDerived(true);
	field2.setSource(this);
	//field2.m_data = m_data;
	return true;
}


template<typename T>
using HostVarField = VarField<T>;

template<typename T>
using DeviceVarField = VarField<T>;

template<typename T>
using HostVariablePtr = std::shared_ptr< HostVarField<T> >;

template<typename T>
using DeviceVariablePtr = std::shared_ptr< DeviceVarField<T> >;
}