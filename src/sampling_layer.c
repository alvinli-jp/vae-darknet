#include "sampling_layer.h"
#include "activations.h"
#include "utils.h"
#include "cuda.h"
#include "blas.h"
#include "gemm.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

float *tmp_sd_part = NULL;

layer make_sampling_layer(int batch, int inputs, int outputs, float beta)
{
    fprintf(stderr, "sampling layer                               %4d   ->   %4d\n",  inputs, outputs);
    layer l = {0};
    l.type = SAMPLING;
    l.batch = batch;
    l.inputs = inputs;
    l.outputs = outputs;
    l.output = calloc(outputs *batch, sizeof(float));
    l.scales = calloc(outputs *batch, sizeof(float));

    // has to be size of l.outputs, because it is set in the next layer
    l.delta = calloc(outputs * batch, sizeof(float));

    l.forward = forward_sampling_layer;
    l.backward = backward_sampling_layer;
    #ifdef GPU
    l.forward_gpu = forward_sampling_layer_gpu;
    l.backward_gpu = backward_sampling_layer_gpu;

    l.output_gpu = cuda_make_array(l.output, outputs*batch); 
    l.scales_gpu = cuda_make_array(l.output, outputs*batch); 
    l.delta_gpu = cuda_make_array(l.delta, outputs*batch); 
    #endif

    tmp_sd_part = calloc(l.outputs*l.batch, sizeof(float));

    l.beta = beta;
    return l;
}

void forward_sampling_layer(const layer l, network net)
{
    fill_cpu(l.outputs*l.batch, 0, l.output, 1);

    int n_z = l.inputs / 2;

    // convert cross input from route layer to sd and mean
    float *sd = calloc(n_z * l.batch, sizeof(float));
    float *mean = calloc(n_z * l.batch, sizeof(float));
    int i;
    for (i = 0; i < l.inputs * l.batch; i+=n_z) {
        if (i % (n_z*2) == 0) {
            copy_cpu(n_z, net.input + i, 1, sd + i / 2, 1);
        } else {
            copy_cpu(n_z, net.input + i, 1, mean + (i-n_z) / 2, 1);
        }
    }
    
    // numpy would be: mean + np.exp(sd / 2) * epsilon (mean.shape=n_z,sd.shape=n:z,epsilon.shape=l.outputs,n_z) (element wise multiplication and addition numpy like)
    int k;
    for (k = 0; k < l.batch; k++) {
        for (i = k * l.outputs; i < k * l.outputs + l.outputs; i++) {
            tmp_sd_part[i] = exp(sd[i] / 2) * rand_normal();

            l.output[i] = mean[i] + tmp_sd_part[i];
        }
    }

    free(sd);
    free(mean);
}

void backward_sampling_layer(const layer l, network net)
{  
    int n_z = l.inputs / 2;

    // derivation of sd part with sd in it saved from forward prop in tmp_sd_part
    float *d_sd = tmp_sd_part;
    scal_cpu(l.outputs*l.batch, 0.5, d_sd, 1);
    
    int i;
    for (i = 0; i < l.outputs*l.batch; i++) {
        d_sd[i] *= l.delta[i];
    }

    // first part d_sd goes to layer -1, partial derivative for sd * l.delta
    // second part goes to layer -3, partial derivative for mean( which is 1) * l.delta

    // convert cross input from route layer to sd and mean
    
    // note: acutally only pointers have to change, it doesnt have to be copied, but doesnt affect perf because n_z is small
    float *sd = calloc(n_z * l.batch, sizeof(float));
    float *mean = calloc(n_z * l.batch, sizeof(float));

    for (i = 0; i < l.inputs * l.batch; i+=n_z) {
        if (i % (n_z*2) == 0) {
            copy_cpu(n_z, net.input + i, 1, sd + i / 2, 1);
        } else {
            copy_cpu(n_z, net.input + i, 1, mean + (i-n_z) / 2, 1);
        }
    }

    // calculate additional influence/ derivative of sd and mean for cost_function due to additionial latent_loss as second part
    // derivative of sd, with respect to latent_loss
    int x;
    for (x = 0; x < l.outputs * l.batch; x++) {
        d_sd[x] += - l.beta * (exp(sd[x]) - 1) / l.batch;
    }
        
    // derivative of mean, with respect to latent_loss
    for (x = 0; x < l.outputs * l.batch; x++) {
        l.delta[x] += - l.beta * mean[x] / l.batch;
    }

    free(sd);
    free(mean);

    // convert to cross-form in net.delta for route layer
    int d_sd_write_point = 0;
    int delta_write_point = n_z;
    for (i = 0; i < l.batch; i++) {
        copy_cpu(n_z, d_sd + i*n_z, 1, net.delta + d_sd_write_point, 1);
        copy_cpu(n_z, l.delta + i*n_z, 1, net.delta + delta_write_point, 1);
        d_sd_write_point += n_z*2;
        delta_write_point += n_z*2;
    }
}

#ifdef GPU

void forward_sampling_layer_gpu(const layer l, network net)
{
    cuda_pull_array(net.input_gpu, net.input, l.inputs*net.batch);
    forward_sampling_layer(l, net);
    cuda_push_array(l.output_gpu, l.output, l.outputs*net.batch);
}

void backward_sampling_layer_gpu(const layer l, network net)
{
    cuda_pull_array(l.delta_gpu, l.delta, l.outputs*l.batch);
    cuda_pull_array(net.delta_gpu, net.delta, l.inputs*l.batch);
    backward_sampling_layer(l, net);
    cuda_push_array(net.delta_gpu, net.delta, l.inputs*l.batch);
}

#endif
