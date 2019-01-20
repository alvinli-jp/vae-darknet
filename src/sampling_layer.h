#ifndef SAMPLING_LAYER_H
#define SAMPLING_LAYER_H
#include "layer.h"
#include "network.h"

layer make_sampling_layer(int batch, int inputs, int outputs, float beta);
void forward_sampling_layer(const layer l, network net);
void backward_sampling_layer(const layer l, network net);

#ifdef GPU
void forward_sampling_layer_gpu(const layer l, network net);
void backward_sampling_layer_gpu(const layer l, network net);
#endif

#endif
