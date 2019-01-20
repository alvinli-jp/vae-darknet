#include "darknet.h"

#include <sys/time.h>
#include <assert.h>
#include <math.h>
#include <utils.h>
#include <blas.h>
#include <image.h>

image *image_evolution = NULL;

int write_float_array_to_file(float *array, int linebreak, int len, char *filename) {
    FILE *filePtr;
 
    filePtr = fopen(filename, "w");
    if (!filePtr) {
        printf("failed to write file %s\n", filename);
        return 0;
    }

    int i = 0;
    for (i = 0; i < len; i++) {
        if (i % linebreak == 0 && i != 0)
            fprintf(filePtr, "\n");
        fprintf(filePtr, "%f,", array[i]);
    }

    return 1;
}

void write_training_to_file(network *net, float suffix_number, char *filename, int save_inputs) {
    char buff[265];
    
    if (save_inputs) {
        sprintf(buff, "%s_input_%d.txt", filename, (int)suffix_number);

        write_float_array_to_file(net->input,
            net->inputs,
            net->inputs*net->batch,
            buff);

    }

    sprintf(buff, "%s_output_%d.txt", filename, (int)suffix_number);

    layer last_layer = net->layers[net->n - 1];
    
    int i;
    for (i = net->n - 1; i >= 0; i--) {
        if (net->layers[i].type != COST) {
            last_layer = net->layers[i];
            break;
        }
    }

    write_float_array_to_file(last_layer.output,
        last_layer.outputs,
        last_layer.outputs*net->batch,
        buff);
}

void visualize_training(network *net, int number_images, int evo_i, int max_i) {
    int i;
    int n = number_images;
    int batch = net->batch;

    if (!image_evolution)
        image_evolution = calloc(max_i, sizeof(image));

    if (n > batch) {
        printf("cant visualize %d images, if batch size is only %d", n, batch);
        return;
    }

    image *all_images = calloc(n*2*max_i, sizeof(image));

    for (i = evo_i*n; i < evo_i*n + n; i ++) {
        layer last_layer = net->layers[net->n-2];

        int y;
        for (y = 0; y < 784; y++){
            if (y % 28 == 0)
                printf("\n");
            if (*(net->input + (i-evo_i*n)*784 + y) < 0.001) printf(" ");
            else if (*(net->input + (i-evo_i*n)*784 + y) < 0.005) printf(".");
            else if (*(net->input + (i-evo_i*n)*784 + y) < 0.05) printf("$");
            else printf("#");
        }

        all_images[i*2] = copy_image(float_to_image(28, 28, 1, net->input + (i-evo_i*n)*784));
        normalize_image(all_images[i*2]);

        for (y = 0; y < 784; y++){
            if (y % 28 == 0)
                printf("\n");
            if (*(last_layer.output + (i-evo_i*n)*784 + y) < 0.001) printf(" ");
            else if (*(last_layer.output + (i-evo_i*n)*784 + y) < 0.005) printf(".");
            else if (*(last_layer.output + (i-evo_i*n)*784 + y) < 0.05) printf("$");
            else printf("#");
        }

        all_images[i*2+1] = copy_image(float_to_image(28, 28, 1, last_layer.output + (i-evo_i*n)*784));
        normalize_image(all_images[i*2+1]);
    }

    
    
    image_evolution[evo_i] = collapse_images_vert(all_images + n*2*evo_i, n*2);
    

    if (evo_i == max_i - 1) {
        char buff[265];
        sprintf(buff, "training_autoencoder");
        image final_image = collapse_images_horz(image_evolution, max_i);
        save_image(final_image, buff);
        show_image(final_image, buff, 1);
        free_image(final_image);
    }
    
}

void save_latent_space(network *net, data train, int max_batches) {
    assert(train.X.rows % net->batch == 0);
    int batch = net->batch;

    layer sampling_layer = net->layers[net->n - 2];
    
    int i;
    for(i = 0; i < net->n; i++) {
        if (net->layers[i].type == SAMPLING) {
            sampling_layer = net->layers[i];
            break;
        }
    }

    int n_z = sampling_layer.outputs;

    float *latent_variables = calloc(n_z * batch * max_batches, sizeof(float));

    for(i = 0; i < max_batches; i++) {
        get_next_batch(train, batch, i*batch, net->input, NULL);
        get_next_batch(train, batch, i*batch, net->truth, NULL);
        
        forward_network(net);
        int x;
        for (x = 0; x < n_z * batch; x++) {
            latent_variables[i*batch*n_z + x] = sampling_layer.output[x];
        }

    }
    char buff[265];
    sprintf(buff, "latent_space.txt");
    write_float_array_to_file(latent_variables, n_z * batch, n_z * batch * max_batches, buff);

    free(latent_variables);   
}


float calc_latent_loss(network *net) {

    int batch = net->batch;
    int i_sampling = 0;

    int i;
    for (i = 0; i < net->n; i++) {
        if (net->layers[i].type == SAMPLING) {
            i_sampling = i;
            break;
        }
    }

    int n_z = net->layers[i_sampling].outputs;

#ifdef GPU
    cuda_pull_array(net->layers[i_sampling - 2].output_gpu, net->layers[i_sampling - 2].output, net->layers[i_sampling - 2].outputs * batch);
    cuda_pull_array(net->layers[i_sampling - 4].output_gpu, net->layers[i_sampling - 4].output, net->layers[i_sampling - 4].outputs * batch);
#endif

    float *sd = net->layers[i_sampling - 2].output;
    float *mean = net->layers[i_sampling - 4].output;

    float sum = 0;
    for (i = 0; i < n_z * batch; i++) {
        sum += 0.5 * (exp(sd[i]) + mean[i] * mean[i] - 1.0 - sd[i]);
    }

    return sum;
}

void only_forward_net(network *net, data train, int batch_offset) {
    
    assert(train.X.rows % net->batch == 0);
    int batch = net->batch;

    get_next_batch(train, batch, batch_offset*batch, net->input, NULL);
    get_next_batch(train, batch, batch_offset*batch, net->truth, NULL);

    forward_network(net);
    
    layer last_layer = net->layers[net->n - 1];
    
    int i;
    for (i = net->n - 1; i >= 0; i--) {
        if (net->layers[i].type != COST) {
            last_layer = net->layers[i];
            break;
        }
    }

    int mi = max_index(last_layer.output, last_layer.outputs);
    float max_val = last_layer.output[mi];

    float *norm_outputs = calloc(last_layer.outputs, sizeof(float));
    copy_cpu(last_layer.outputs, last_layer.output, 1, norm_outputs, 1);
    scal_cpu(last_layer.outputs, 1/ max_val, norm_outputs, 1);

    printf("input          output          normalized_output:\n");
    int x;
    for(x = 0; x < 28; x++) {
        int y;
        for (y = x * 28; y < x*28 + 28; y++){
            if (net->input[y] < 0.001) printf(" ");
            else if (net->input[y] < 0.005) printf(".");
            else if ((net->input[y] < 0.05)) printf("$");
            else printf("#");
        }
        printf("          ");
        for (y = x * 28; y < x*28 + 28; y++){
            if (last_layer.output[y] < 0.25) printf(" ");
            else if (last_layer.output[y] < 0.5) printf(".");
            else if (last_layer.output[y] < 0.75) printf("$");
            else printf("#");
        }
        printf("          ");
        for (y = x * 28; y < x*28 + 28; y++){
            if (norm_outputs[y] < 0.25) printf(" ");
            else if (norm_outputs[y] < 0.5) printf(".");
            else if (norm_outputs[y] < 0.75) printf("$");
            else printf("#");
        }
        printf("\n");
    }
    printf("\n");

    free(norm_outputs);      

}

float train_autoencoder(network *net, data train) {
    assert(train.X.rows % net->batch == 0);
    int n = train.X.rows / net->batch;
    int batch = net->batch;
    float sum = 0;
    int i;
    for (i = 0; i < n; i++) {
        get_next_batch(train, batch, i * batch, net->input, NULL);
        get_next_batch(train, batch, i * batch, net->truth, NULL);
        *net->seen += net->batch;
        
        forward_network(net);
        float error = *net->cost;

        // layer first_conv = net->layers[0];
        // printf("layer[0] activation maps:\n");
        // int x;
        // if (i == 0){
        //     for(x = 0; x < first_conv.outputs; x++) {
        //         if (x % first_conv.out_w == 0) printf("\n");
        //         if (x % (first_conv.out_w * first_conv.out_h) == 0) printf("\n----------------------\n");
        //         if (net->input[x] < 0.001) printf(" ");
        //         else if (net->input[x] < 0.005) printf(".");
        //         else if ((net->input[x] < 0.05)) printf("$");
        //         else printf("#");
        //     }
        // }

        backward_network(net);
        if(((*net->seen)/net->batch)%net->subdivisions == 0) update_network(net);
        sum += error;
    }
    return (float)sum/(n*net->batch);
}

void sample_vae(network *netp, float *latent_vector, int latent_vector_len) {
    network net = *netp;
    
    int i_sampling = 0;
    int i_output_layer = net.n - 1;

    int i;
    for (i = 0; i < net.n; i++) {
        if (net.layers[i].type == SAMPLING) {
            i_sampling = i;
        }
        if (net.layers[i].type == COST) {
            i_output_layer = i - 1;
            break;
        }
    }

    int n_z = net.layers[i_sampling].outputs;

    assert(n_z == latent_vector_len);

    copy_cpu(n_z, latent_vector, 1, net.input, 1);
    
    for(i = i_sampling + 1; i <= i_output_layer; ++i){
        net.index = i;
        layer l = net.layers[i];
        if(l.delta){
            fill_cpu(l.outputs * l.batch, 0, l.delta, 1);
        }
        l.forward(l, net);
        net.input = l.output;
        if(l.truth) {
            net.truth = l.output;
        }
    }

    layer last_layer = net.layers[i_output_layer];

    int mi = max_index(last_layer.output, last_layer.outputs);
    float max_val = last_layer.output[mi];

    float *norm_outputs = calloc(last_layer.outputs, sizeof(float));
    copy_cpu(last_layer.outputs, last_layer.output, 1, norm_outputs, 1);
    scal_cpu(last_layer.outputs, 1/ max_val, norm_outputs, 1);

    printf("output          normalized_output:\n");
    int x;
    for(x = 0; x < 28; x++) {
        int y;
        for (y = x * 28; y < x*28 + 28; y++){
            if (last_layer.output[y] < 0.25) printf(" ");
            else if (last_layer.output[y] < 0.5) printf(".");
            else if (last_layer.output[y] < 0.75) printf("$");
            else printf("#");
        }
        printf("          ");
        for (y = x * 28; y < x*28 + 28; y++){
            if (norm_outputs[y] < 0.25) printf(" ");
            else if (norm_outputs[y] < 0.5) printf(".");
            else if (norm_outputs[y] < 0.75) printf("$");
            else printf("#");
        }
        printf("\n");
    }
    printf("\n");

    free(norm_outputs);      
}

float train_vae(network *net, data train) {
    assert(train.X.rows % net->batch == 0);
    int n = train.X.rows / net->batch;
    int batch = net->batch;
    float sum_img_loss = 0;
    float sum_latent_loss = 0;
    int i;
    for (i = 0; i < n; i++) {
        get_next_batch(train, batch, i * batch, net->input, NULL);
        get_next_batch(train, batch, i * batch, net->truth, NULL);

        *net->seen += net->batch;
        forward_network(net);
       
        float error = *net->cost;
        sum_latent_loss += calc_latent_loss(net);
        sum_img_loss += error;

        backward_network(net);

        if (((*net->seen)/net->batch)%net->subdivisions == 0) update_network(net);
    }
    printf("latent_loss: %f, img_loss: %f\n", sum_latent_loss/(n*net->batch), sum_img_loss/(n*net->batch));
    return (float)sum_img_loss/(n*net->batch);
}

void run_autoencoder(int argc, char **argv, int vae) {
    srand(0);
    if(argc < 4){
        fprintf(stderr, "usage: %s %s [train/sample] [cfg] [train_file/latent_vector_file] [-weights (optional)] [train ? [-N (optional)] [-output (optional)]] \n", argv[0], argv[1]);
        return;
    }

    char *cfg = argv[3];
    char *weight_file = find_char_arg(argc, argv, "-weights", 0);

    char *output_file = find_char_arg(argc, argv, "-output", 0);
    if (output_file) {
        size_t len = strlen(output_file);
        output_file[len-4] = '\0';
    }

#ifdef GPU
    cuda_set_device(0);
#endif

    network *net = load_network(cfg, weight_file, 0);

    if (0==strcmp(argv[2], "train")) {

        char *train_file = argv[4];

        data train = {0};
        train.shallow = 0;
        matrix X = csv_to_matrix(train_file);
        train.X = X;

        int N = find_int_arg(argc, argv, "-N", 10);

        train.X.rows = N;

        float loss = 0;
        float epoch = 0;
        while(get_current_batch(net) < net->max_batches || net->max_batches == 0){
            only_forward_net(net, train, 1);
            
            if (output_file)
                write_training_to_file(net, epoch, output_file, 1);

            if (vae) {
                train_vae(net, train);
                epoch = (float)(*net->seen)/N;
                printf("epoch %f finished\n", epoch);
            }
            else {
                loss = train_autoencoder(net, train);
                epoch = (float)(*net->seen)/N;
                printf("epoch %f finished with loss: %f\n", epoch, loss);
            }
        }
        int n;
        for (n = 0; n < 10; n++)
            only_forward_net(net, train, n);

        if (vae)
            save_latent_space(net, train, 500);

        free_data(train);

        char buff[256];
        sprintf(buff, "vae.weights");
        save_weights(net, buff);

    } else if (0==strcmp(argv[2], "sample")) {
        
        char *latent_vector_filename = argv[4];
        matrix latent_vectors = csv_to_matrix(latent_vector_filename);
        
        int i;
        for(i = 0; i < latent_vectors.rows; i++)  {
            printf("sampling latent_vectors[%d]:", i);
            int x;
            for (x = 0; x < latent_vectors.cols; x++)
                printf("%.3f, ", latent_vectors.vals[i][x]);
            printf("\n");

            sample_vae(net, latent_vectors.vals[i], latent_vectors.cols);

            if (output_file)
                write_training_to_file(net, i, output_file, 1);
        }

    }

    free_network(net);
}