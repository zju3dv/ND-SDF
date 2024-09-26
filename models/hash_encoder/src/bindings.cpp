#include<pybind11/pybind11.h>
#include "hash_encoder.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("hash_encoder_forward",&hash_encoder_forward);
    m.def("hash_encoder_backward",&hash_encoder_backward);
    m.def("hash_encoder_second_backward",&hash_encoder_second_backward);
}

