#include<torch/extension.h>
#include<torch/torch.h>

void hash_encoder_forward(const at::Tensor x, const at::Tensor embeddings, const at::Tensor offsets, const at::Tensor resolution_list, const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L, const bool if_cal_grad_x, const uint32_t AL, at::Tensor y, at::Tensor dy_dx);

void hash_encoder_backward(const at::Tensor grad_y, const at::Tensor x, const at::Tensor dy_dx, const at::Tensor offsets, const at::Tensor resolution_list,  const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L, const bool if_cal_grad_x, const uint32_t AL, at::Tensor grad_x, at::Tensor grad_embeddings);


void hash_encoder_second_backward(const at::Tensor grad2_grad_x, const at::Tensor x, const at::Tensor embeddings, const at::Tensor grad_y, const at::Tensor dy_dx,  const at::Tensor offsets, const at::Tensor resolution_list,  const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L, const bool if_cal_hessian_x, const uint32_t AL, at::Tensor grad2_embeddings, at::Tensor grad2_grad_y, at::Tensor grad2_x);