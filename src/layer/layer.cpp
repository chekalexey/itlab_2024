#include "layer/layer.h"

void Layer::addNeighbor(Layer* neighbor) {
  if (neighbor != nullptr) {
    neighbors_.push_back(neighbor);
  }
}

void Layer::removeNeighbor(Layer* neighbor) { neighbors_.remove(neighbor); }

std::string Layer::getInfoString() const {
  return "Layer (ID: " + std::to_string(id_) + ")";
}