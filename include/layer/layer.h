#ifndef LAYER_H
#define LAYER_H

#include <list>

#include "arm_compute/runtime/NEON/NEFunctions.h"
#include "utils/Utils.h"

using namespace arm_compute;
using namespace utils;

struct LayerAttributes {
  int id = -1;
};

class Layer {
protected:
	int id_;

public:
	Layer() = default;
	explicit Layer(const LayerAttributes& attrs) : id_(attrs.id) {}
	explicit Layer(int id) : id_(id) {}
	virtual ~Layer() = default;
	void setID(int id) { id_ = id; }
	int getID() const { return id_; }
	virtual std::string getInfoString() const;
	//virtual void exec(Tensor& input, Tensor& output) = 0;
	//virtual void exec(Tensor& input1, Tensor& input2, Tensor& output) = 0;
	virtual void exec() = 0;
	//virtual Shape get_output_shape() = 0;

	virtual std::string get_type_name() const = 0;
	void addNeighbor(Layer* neighbor);
	void removeNeighbor(Layer* neighbor);
	std::list<Layer*> neighbors_;
};
#endif