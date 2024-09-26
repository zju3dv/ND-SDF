/*
by author: github.com/dawnzyt
mail: tangziyu@zju.edu.cn
TODO: 1. 优化循环, #pragma unroll[√];
      2. 实现grad2_x;[√]
FIXME：大per level dim(C=8)下比tcnn慢了11%

*/
#include<cuda.h>
#include<cuda_fp16.h>
#include<cuda_runtime.h>

#include<ATen/cuda/CUDAContext.h>
#include<torch/extension.h>
#include<torch/torch.h>

#include<algorithm>
#include<cstdio>
#include<stdio.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x" IS NOT A CUDA TENSOR!!!!")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x" IS NOT CONTIGUOUS!!!!")
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type()==at::ScalarType::Float||x.scalar_type()==at::ScalarType::Half||x.scalar_type()==at::ScalarType::Double, #x" IS NOT A FLOATING TENSOR(HALF, FLOAT or DOUBLE)!!!!")
#define CHECK_INT(x) TORCH_CHECK(x.scalar_type()==at::ScalarType::Int, #x" IS NOT A INTEGER TYPE TENSOR!!!!")
#define MIN(x,y) ((x)<(y)?(x):(y))


// cuda supports __half not torch's c10::Half, so we need to overload
// at::Float和at::Double和cuda的float、double兼容不用重载
// 注意: half的atomicAdd非常慢, 最好别出现使用该half的atomicAdd的情况
static inline __device__ c10::Half atomicAdd(c10::Half *address, c10::Half val){
    return atomicAdd(reinterpret_cast<__half*>(address), val);
}


template<typename T>
static inline __host__ T divide_round_up(T v,T d){return (v+d-1)/d;}

// hash function: {x_idx,y_idx,z_idx} -> uint32_t 
template<uint32_t D>
static inline __device__ uint32_t fast_grid_hash(const uint32_t grid_idx[D]){
    static const uint32_t primes[7]={1, 2654435761, 805459861, 3674653429, 2097192037, 1434869437, 2165219737};
    uint32_t h=1;
    #pragma unroll
    for(uint32_t i=0;i<D;i++) h^=(primes[i]*grid_idx[i]);
    return h;
}

// get_hash_idx
// when res**D == offset, means no hash, just use flatten grid_idx as hash_idx
template<uint32_t D>
static inline __device__ uint32_t get_hash_idx(const uint32_t grid_idx[D], const uint32_t res, const uint32_t offset){
    uint32_t idx=0;
    uint32_t stride=1;
    #pragma unroll
    for(uint32_t d=0;d<D;d++){
        idx+=stride*grid_idx[d]; // format: [z][y][x]
        stride*=res;
    }
    if(stride==offset) return idx;
    return fast_grid_hash<D>(grid_idx)%offset;
}

static inline __device__ float smooth(float x){return x*x*(3.0f-2*x);}
static inline __device__ float smooth_derivative(float x){return 6.0f*x*(1.0f-x);} // d(smooth(x))/dx
static inline __device__ float smooth_second_derivative(float x){return 6.0f-12.0f*x;} // d2(smooth(x))/dx2

/* ------------------------------------------- Kernel ------------------------------------------- */
// forward kernel: element{b, level}
template<typename scalar_t, uint32_t D, uint32_t C, uint32_t N_C>
__global__ void kernel_forward(const float * __restrict__ x, 
const scalar_t * __restrict__ embeddings, 
const uint32_t * __restrict__ offsets, 
const uint32_t * __restrict__ resolution_list, 
const uint32_t  __restrict__ B, 
const uint32_t  __restrict__ L, 
const bool if_cal_grad_x,
scalar_t * __restrict__ y,
scalar_t * __restrict__ dy_dx){

    /**
     * parallel: b,level,C/N_C
     * x: B,D
     * embeddings: total,C
     * y: B,L,C
     * dy_dx: B,L,D,C
    */
    const uint32_t b=blockIdx.x*blockDim.x+threadIdx.x;
    // ATTENTION: thread-b must < B
    if(b>=B)return ;
    const uint32_t Ci=blockIdx.z;
    const uint32_t level=blockIdx.y;
    const uint32_t res=resolution_list[level];
    const uint32_t offset=offsets[level+1]-offsets[level];
    
    // locate
    x+=b*D;
    embeddings+=offsets[level]*C; 
    y+=b*L*C+level*C+Ci*N_C;
    dy_dx+=b*L*D*C+level*D*C;

    // find voxel and voxel idx
    uint32_t voxel_idx[D]; // local voxel's grid idx(bottom-left corner)
    float voxel[D]; // coordinate in local voxel, \in [0,1) 
    float voxel_smooth[D]; // smooth
    float voxel_smooth_derivative[D]; // smooth derivative
    #pragma unroll
    for(uint32_t d=0;d<D;d++){
        voxel_idx[d] = floorf(x[d]*(res-1));
        voxel_idx[d] = MIN(voxel_idx[d],res-2); // in case x[d]=1 -> voxel_idx=res-1导致越界
        voxel[d]=x[d]*(res-1)-voxel_idx[d];
        
        // smooth voxel x: 加高次项使二阶导数连续(比如如果要求sdf的hessian矩阵：不升维一定全0[一阶导已与坐标无关])
        voxel_smooth[d]=smooth(voxel[d]);
        voxel_smooth_derivative[d]=smooth_derivative(voxel[d]);
        if(x[d]<0||x[d]>1){
            /* [ERROR] out of range, fill y with 0 */
            #pragma unroll
            for(uint32_t c=0;c<N_C;c++)y[c]=0;
            if(if_cal_grad_x){
                #pragma unroll
                for(uint32_t d=0;d<D;d++){
                    #pragma unroll
                    for(uint32_t c=0;c<N_C;c++){
                        dy_dx[d*C+N_C*Ci+c]=0;
                    }
                }
            }
            return ; //return
        }
    }
    // 1. interpolation
    uint32_t cur_grid_idx[D];
    uint32_t hash_idx;
    scalar_t local_y[N_C]={0};
    #pragma unroll
    for(uint32_t i=0;i<(1<<D);i++){
        float w=1;
        #pragma unroll
        for(uint32_t d=0;d<D;d++){
            if(((1<<d)&i)==0){ // left side
                w*=(1-voxel_smooth[d]);
                cur_grid_idx[d]=voxel_idx[d];
            }else{ //right side
                w*=voxel_smooth[d];
                cur_grid_idx[d]=voxel_idx[d]+1;
            }
        }
        // std::printf("%d %f\n",i,w);
        // hash cur grid idx
        hash_idx = get_hash_idx<D>(cur_grid_idx, res, offset);
        #pragma unroll
        for(uint32_t c=0;c<N_C;c++){
            local_y[c]+=embeddings[hash_idx*C+Ci*N_C+c]*w; // write to y
        }
    }
    // 2. write local_y to y
    #pragma unroll
    for(uint32_t c=0;c<N_C;c++) y[c]=local_y[c];

    /* 3. precompute dy_dx for grad_x's backward */ 
    // 只考虑level层的local voxel feature
    // known: voxel_smooth, voxel_idx
    // dy_dx: (D, C)
    if(!if_cal_grad_x) return ;
    scalar_t local_dy_dx[N_C]={0};
    #pragma unroll
    for(uint32_t gd=0;gd<D;gd++){ // gd: cur grad computing dim in D.
        #pragma unroll
        for(uint32_t i=0;i<(1<<(D-1));i++){ // gd维度值pair
            float w=(float)(res-1);
            #pragma unroll
            for(uint32_t j=0;j<D-1;j++){
                uint32_t d=j<gd?j:(j+1); // 索引为d
                if((i&(1<<j))==0){ //left side
                    w*=(1-voxel_smooth[d]);
                    cur_grid_idx[d]=voxel_idx[d];
                }else{
                    w*=voxel_smooth[d];
                    cur_grid_idx[d]=voxel_idx[d]+1;
                }
            }
            // std::printf("[DEBUG]-[forward,dy_dx] w:%.4f\n",w);
            cur_grid_idx[gd]=voxel_idx[gd];
            uint32_t hash_idx_left=get_hash_idx<D>(cur_grid_idx,res,offset)*C;
            cur_grid_idx[gd]=voxel_idx[gd]+1;
            uint32_t hash_idx_right=get_hash_idx<D>(cur_grid_idx,res,offset)*C;
            // std::printf("[DEBUG]-[forward,dy_dx] hash_idx_left:%d, hash_idx_right:%d\n",hash_idx_left,hash_idx_right);
            #pragma unroll
            for(uint32_t c=0;c<N_C;c++){
                // dy/dx=dy/dsmooth_x * dsmooth_x/dcx * dcx/dx' * dx'/dx 
                local_dy_dx[c]+=w*(embeddings[hash_idx_right+Ci*N_C+c]-embeddings[hash_idx_left+Ci*N_C+c])*voxel_smooth_derivative[gd];// *0.5f; 不用乘0.5f, 在Function中的缩放计算会被torch自动检测到并添加到计算图。
            }
        }
        // write to global memery dy_dx: torch.empty(B*L*D*C)
        #pragma unroll
        for(uint32_t c=0;c<N_C;c++){
            // std::printf("[DEBUG]-[forward,dy_dx] local_dy_dx:%.4f\n",local_dy_dx[c]);
            dy_dx[gd*C+Ci*N_C+c]=local_dy_dx[c];
            local_dy_dx[c]=0;
        }
    }
}
//backward-x kernel
template<typename scalar_t,uint32_t D, uint32_t C,uint32_t N_C>
__global__ void kernel_x_backward(const scalar_t * __restrict__ grad_y,
const scalar_t * __restrict__ dy_dx,
const uint32_t B,
const uint32_t L,
const uint32_t AL,
scalar_t * __restrict__ grad_x){
    /** parallel: b,d
     * grad_y: B,L,C
     * dy_dx: B,L,D,C
     * grad_x: B,D
    */
    // split
    const uint32_t b= blockDim.x*blockIdx.x+threadIdx.x;
    if(b>=B) return ;
    const uint32_t Ci=blockIdx.z;
    const uint32_t d= blockIdx.y;
    
    // locate
    grad_y+=b*L*C;
    dy_dx+=b*L*D*C;
    grad_x+=b*D;

    // accumulate grad_x[b,d]
    // TODO: 这里访问dy_dx没能最大化利用cache, 仅针对这里的dy_dx访问模式，最佳dim顺序: B,D,L,C, 但forward的dy_dx是B,L,D,C
    #pragma unroll
    for(uint32_t l=0;l<MIN(L,AL);l++){
        #pragma unroll
        for(uint32_t c=0;c<N_C;c++){
            atomicAdd(&grad_x[d], (dy_dx[l*D*C+d*C+Ci*N_C+c]*grad_y[l*C+Ci*N_C+c]));
            // grad_x[d]+=dy_dx[l*D*C+d*C+Ci*N_C+c]*grad_y[l*C+Ci*N_C+c];
        }
    }
}
// backward-embedding kernel
template<typename scalar_t,uint32_t D, uint32_t C, uint32_t N_C>
__global__ void kernel_embedding_backward(const scalar_t * __restrict__ grad_y,
const float * __restrict__ x,
const uint32_t * __restrict__ offsets,
const uint32_t * __restrict__ resolution_list,
const uint32_t  __restrict__ B,
const uint32_t  __restrict__ L,
scalar_t * __restrict__ grad_embeddings){
    /**
     * parellel: b,level,C/N_C
     * grad_y: B,L,C
     * x: B,D
     * grad_embeddings: offsets[-1],C
    */
    // split
    const uint32_t b=blockIdx.x*blockDim.x+threadIdx.x;
    // ATTENTION: thread-b must < B
    if(b>=B)return ;
    const uint32_t Ci=blockIdx.z;
    const uint32_t level=blockIdx.y;
    const uint32_t res=resolution_list[level];
    const uint32_t offset=offsets[level+1]-offsets[level];
    
    // locate
    grad_y+=b*L*C+level*C+Ci*N_C;
    x+=b*D;
    grad_embeddings+=offsets[level]*C;

    // find voxel and voxel idx
    uint32_t voxel_idx[D]; // local voxel's grid idx(bottom-left corner)
    float voxel[D]; // coordinate in local voxel, \in [0,1) 
    float voxel_smooth[D]; // smooth
    float voxel_smooth_derivative[D]; // smooth derivative
    #pragma unroll
    for(uint32_t d=0;d<D;d++){
        voxel_idx[d] = floorf(x[d]*(res-1));
        voxel_idx[d] = MIN(voxel_idx[d],res-2);
        voxel[d]=x[d]*(res-1)-voxel_idx[d];
        
        voxel_smooth[d]=smooth(voxel[d]);
        voxel_smooth_derivative[d]=smooth_derivative(voxel[d]);
        if(x[d]<0||x[d]>1) return ;/* [ERROR] out of range */
    }

    // grad_y to local_grad_y
    scalar_t local_grad_y[N_C]={0}; // 尽量少访问全局内存
    #pragma unroll
    for(uint32_t i=0;i<N_C;i++) local_grad_y[i]=grad_y[i];

    // accumulate grad
    uint32_t cur_grid_idx[D];
    uint32_t hash_idx;
    #pragma unroll
    for(uint32_t i=0;i<(1<<D);i++){
        float w=1;
        #pragma unroll
        for(uint32_t d=0;d<D;d++){
            if(((1<<d)&i)==0){ // left side
                w*=(1-voxel_smooth[d]);
                cur_grid_idx[d]=voxel_idx[d];
            }else{ //right side
                w*=voxel_smooth[d];
                cur_grid_idx[d]=voxel_idx[d]+1;
            }
        }
        hash_idx = get_hash_idx<D>(cur_grid_idx, res, offset);
        
        // atomicAdd for __half is slow, use __half2 to accelerate (3/4 times faster)
        if(std::is_same<scalar_t, c10::Half>::value){
            #pragma unroll
            for(uint32_t c=0;c<N_C;c+=2){
                __half2 v={(__half)(w*grad_y[c]),(__half)(w*grad_y[c+1])}; // 向量化
                atomicAdd((__half2 *)(&grad_embeddings[hash_idx*C+Ci*N_C+c]), v);
            }if(N_C%2!=0) atomicAdd((c10::Half *)&grad_embeddings[hash_idx*C+Ci*N_C+N_C-1],(c10::Half)(w*local_grad_y[N_C-1])); // C为奇数的情况
        }else{
            #pragma unroll
            for(uint32_t c=0;c<N_C;c++){
                atomicAdd(&grad_embeddings[hash_idx*C+Ci*N_C+c], (w*local_grad_y[c]));
            }
        }
    }
    
}
// second backward-embedding kernel
template<typename scalar_t,uint32_t D, uint32_t C, uint32_t N_C>
__global__ void kernel_embedding_second_backward(const scalar_t * __restrict__ grad2_grad_x,
const float * __restrict__ x,
const scalar_t * __restrict__ grad_y,
const uint32_t * __restrict__ offsets,
const uint32_t * __restrict__ resolution_list,
const uint32_t  __restrict__ B,
const uint32_t  __restrict__ L,
scalar_t * __restrict__ grad2_embeddings){
    /**
     * parellel: b,level,C/N_C
     * grad2_grad_x: B,D
     * x: B,D
     * grad_y: B,L,C
     * grad2_embeddings: offsets[-1],C
    */
    //split
    const uint32_t b=blockIdx.x*blockDim.x+threadIdx.x;
    // ATTENTION: thread-b must < B
    if(b>=B)return ;
    const uint32_t Ci=blockIdx.z;
    const uint32_t level=blockIdx.y;
    const uint32_t res=resolution_list[level];
    const uint32_t offset=offsets[level+1]-offsets[level];

    // locate
    grad2_grad_x+=b*D;
    x+=b*D;
    grad_y+=b*L*C+level*C+Ci*N_C;
    grad2_embeddings+=offsets[level]*C;

    // find voxel and voxel idx
    uint32_t voxel_idx[D]; // local voxel's grid idx(bottom-left corner)
    float voxel[D]; // coordinate in local voxel, \in [0,1) 
    float voxel_smooth[D]; // smooth
    float voxel_smooth_derivative[D]; // smooth derivative
    #pragma unroll
    for(uint32_t d=0;d<D;d++){
        voxel_idx[d] = floorf(x[d]*(res-1));
        voxel_idx[d] = MIN(voxel_idx[d],res-2);
        voxel[d]=x[d]*(res-1)-voxel_idx[d];
        
        voxel_smooth[d]=smooth(voxel[d]);
        voxel_smooth_derivative[d]=smooth_derivative(voxel[d]);
        if(x[d]<0||x[d]>1) return ;/* [ERROR] out of range */
    }
    
    // grad2_grad_x to local_grad2_grad_x
    scalar_t local_grad2_grad_x[D]={0}; // 尽量少访问全局内存
    #pragma unroll
    for(uint32_t i=0;i<D;i++) local_grad2_grad_x[i]=grad2_grad_x[i];

    // accumulate local voxel grad2_embeddings
    scalar_t grad2_embeddings_wo_grad_y[1<<D]={0}; // 可提为公因式的除grad_y的项, grad_y
    #pragma unroll
    for(uint32_t gd=0;gd<D;gd++){ // dy_dx's grad dim
        #pragma unroll
        for(uint32_t i=0;i<(1<<(D-1));i++){ // 枚举坐标另外两个维度的所有取值
            float w=1;
            uint32_t cur=0; // voxel index in binary representation, eg: D=3, 000-111
            #pragma unroll
            for(uint32_t j=0;j<D-1;j++){
                uint32_t d=j<gd?j:(j+1);
                if((i&(1<<j))==0){ // leftside
                    w*=(1-voxel_smooth[d]);
                }else{
                    w*=voxel_smooth[d];
                    cur+=(1<<d);
                }
            }
            // accumulate grad term w/o grad_y
            grad2_embeddings_wo_grad_y[cur]+=-local_grad2_grad_x[gd]*w*voxel_smooth_derivative[gd]*(res-1);//*0.5f; // left
            grad2_embeddings_wo_grad_y[cur+(1<<gd)]+=local_grad2_grad_x[gd]*w*voxel_smooth_derivative[gd]*(res-1);//*0.5f; //right
        }
    }

    // grad_y to local_grad_y
    scalar_t local_grad_y[N_C]={0}; // 尽量少访问全局内存
    #pragma unroll
    for(uint32_t i=0;i<N_C;i++) local_grad_y[i]=grad_y[i];

    // accumulate grad2_embeddings
    uint32_t cur_grid_idx[D];
    #pragma unroll
    for(uint32_t i=0;i<(1<<D);i++){
        #pragma unroll
        for(uint32_t d=0;d<D;d++){
            if((i&(1<<d))==0){ // leftside
                cur_grid_idx[d]=voxel_idx[d];
            }else{
                cur_grid_idx[d]=voxel_idx[d]+1;
            }
        }
        uint32_t hash_idx=get_hash_idx<D>(cur_grid_idx,res,offset)*C;
        // TODO: 这里需要原子操作。[√]
        if(std::is_same<scalar_t, c10::Half>::value){ // __half2 to accelerate half atomicAdd
            #pragma unroll
            for(uint32_t c=0;c<N_C;c+=2){
                __half2 v={(__half)(1.0f*local_grad_y[c]*grad2_embeddings_wo_grad_y[i]),(__half)(1.0f*local_grad_y[c+1]*grad2_embeddings_wo_grad_y[i])}; // 这里的1.0f是为了转换为float，不能直接c10::Half->__half; 据说也能使用__ushort_as_half来转换
                atomicAdd((__half2*)&grad2_embeddings[hash_idx+Ci*N_C+c], v);
            }if(N_C%2!=0) atomicAdd((c10::Half*)&grad2_embeddings[hash_idx+Ci*N_C+N_C-1], (c10::Half)(local_grad_y[N_C-1]*grad2_embeddings_wo_grad_y[i]));
        }else{
            #pragma unroll
            for(uint32_t c=0;c<N_C;c++){
                atomicAdd(&grad2_embeddings[hash_idx+Ci*N_C+c], (local_grad_y[c]*grad2_embeddings_wo_grad_y[i]));
            }
        }
    }
}
// second backward-grad_y kernel
template<typename scalar_t, uint32_t D, uint32_t C, uint32_t N_C>
__global__ void kernel_grad_y_second_backward(const scalar_t * __restrict__ grad2_grad_x,
const scalar_t * __restrict__ dy_dx,
const uint32_t  __restrict__ B,
const uint32_t  __restrict__ L,
scalar_t * __restrict__ grad2_grad_y){
    /**
     * parallel: b,level,C/N_C
     * grad2_grad_x: B,D
     * dy_dx: B,L,D,C
     * grad2_grad_y: B,L,C
    */
    //split
    const uint32_t b=blockIdx.x*blockDim.x+threadIdx.x;
    // ATTENTION: thread-b must < B
    if(b>=B)return ;
    const uint32_t Ci=blockIdx.z;
    const uint32_t level=blockIdx.y;

    // locate
    grad2_grad_x+=b*D;
    dy_dx+=b*L*D*C+level*D*C;
    grad2_grad_y+=b*L*C+level*C+Ci*N_C;
    
    // local_grad2_grad_y and local_grad2_grad_x
    scalar_t local_grad2_grad_y[N_C]={0};
    scalar_t local_grad2_grad_x[D]={0};
    #pragma unroll
    for(uint32_t i=0;i<D;i++) local_grad2_grad_x[i]=grad2_grad_x[i];

    // accumulate grad2_grad_y
    #pragma unroll
    for(uint32_t d=0;d<D;d++){
        #pragma unroll
        for(uint32_t c=0;c<N_C;c++){
            local_grad2_grad_y[c]+=local_grad2_grad_x[d]*dy_dx[d*C+Ci*N_C+c];
        }
    }

    // write to global memery
    #pragma unroll
    for(uint32_t c=0;c<N_C;c++) grad2_grad_y[c]=local_grad2_grad_y[c];
}

// TODO: grad2_x
// second backward-grad2_x kernel
template<typename scalar_t, uint32_t D, uint32_t C>
__global__ void kernel_grad2_x_second_backward(const scalar_t * __restrict__ grad2_grad_x,
const float * __restrict__ x,
const scalar_t * __restrict__ embeddings,
const scalar_t * __restrict__ grad_y,
const uint32_t * __restrict__ offsets,
const uint32_t * __restrict__ resolution_list,
const uint32_t  __restrict__ B,
const uint32_t  __restrict__ L,
scalar_t * __restrict__ grad2_x){
    /**
    * parallel: b,level
    * grad2_grad_x: B,D
    * x: B,D
    * embeddings: offsets[-1]*C
    * grad_y: B,L,C
    * grad2_x: B,D
    */
    //split
    const uint32_t b=blockIdx.x*blockDim.x+threadIdx.x;
    // ATTENTION: thread-b must < B
    if(b>=B)return ;
    const uint32_t level=blockIdx.y;
    const uint32_t res=resolution_list[level];
    const uint32_t offset=offsets[level+1]-offsets[level];

    // locate
    grad2_grad_x+=b*D;
    x+=b*D;
    embeddings+=offsets[level]*C;
    grad_y+=b*L*C+level*C;
    grad2_x+=b*D;

    // find voxel and voxel idx
    uint32_t voxel_idx[D]; // local voxel's grid idx(bottom-left corner)
    float voxel[D]; // coordinate in local voxel, \in [0,1)
    float voxel_smooth[D]; // smooth
    float voxel_smooth_derivative[D]; // smooth derivative
    float voxel_smooth_second_derivative[D]; // smooth second derivative
    #pragma unroll
    for(uint32_t d=0;d<D;d++){
        voxel_idx[d] = floorf(x[d]*(res-1));
        voxel_idx[d] = MIN(voxel_idx[d],res-2);
        voxel[d]=x[d]*(res-1)-voxel_idx[d];

        voxel_smooth[d]=smooth(voxel[d]);
        voxel_smooth_derivative[d]=smooth_derivative(voxel[d]);
        voxel_smooth_second_derivative[d]=smooth_second_derivative(voxel[d]);
        if(x[d]<0||x[d]>1) return ;/* [ERROR] out of range */
    }
    // 先查询cube顶点的hash idx
    uint32_t cur_grid_idx[D], hash_idx[1<<D];
    #pragma unroll
    for(uint32_t i=0;i<(1<<D);i++){
        #pragma unroll
        for(uint32_t j=0;j<D;j++){
            if(((1<<j)&i)==0){
                cur_grid_idx[j]=voxel_idx[j];
            }else cur_grid_idx[j]=voxel_idx[j]+1;
        }
        hash_idx[i]=get_hash_idx<D>(cur_grid_idx, res, offset);
    }

    // level下的local hessian
    // 实际上这里就能求出level特征贡献的hessian矩阵，但torch的规定只求标量梯度，我这里还是把hessian累加上去。
    scalar_t hessian[D][D]={0};
    #pragma unroll
    for(uint32_t gd=0;gd<D;gd++){ // gd表示一阶grad_x的第gd维
        #pragma unroll
        for(uint32_t d=0;d<D;d++){ // d表示对第d维求导。
            if(gd==d){ // 对角线, 二阶导数
                cur_grid_idx[d]=voxel_idx[d];
                #pragma unroll
                for(uint32_t i=0;i<(1<<(D-1));i++){ 
                    float w=1;
                    uint32_t cur=0; // voxel index in binary representation, eg: D=3, 000-111
                    #pragma unroll
                    for(uint32_t j=0;j<D-1;j++){
                        uint32_t tj=j<gd?j:(j+1);
                        if((i&(1<<j))==0){ // leftside
                            w*=(1-voxel_smooth[tj]);
                        }else{
                            w*=voxel_smooth[tj];
                            cur+=(1<<tj);
                        }
                    }
                    uint32_t idx_l=hash_idx[cur];
                    uint32_t idx_r=hash_idx[cur+(1<<d)];
                    // 累加对角线hessian
                    #pragma unroll
                    for(uint32_t c=0;c<C;c++){
                        hessian[gd][d]+=-(res-1)*(res-1)*voxel_smooth_second_derivative[d]*w*embeddings[idx_l*C+c]*grad_y[c];
                        hessian[gd][d]+= (res-1)*(res-1)*voxel_smooth_second_derivative[d]*w*embeddings[idx_r*C+c]*grad_y[c];
                    }
                }
            }else{
                uint32_t td; // 剩下一维。
                #pragma unroll
                for(uint32_t i=0;i<D;i++){
                    if(i!=gd&&i!=d){
                        td=i;
                        break;
                    }
                }
                // 枚举gd和d对应坐标的选择, 两个在一侧为正, 反之为负
                #pragma unroll
                for(uint32_t i=0;i<(1<<(D-1));i++){
                    bool flag=1; // gd、d是否同侧
                    uint32_t cur=0;
                    #pragma unroll
                    for(uint32_t j=0;j<D-1;j++){
                        uint32_t corr_d=j==0?gd:d; // j对应坐标
                        if(((1<<j)&i)==0){ // leftside
                            cur+=0;
                            flag^=0;
                        }else{
                            cur+=1<<corr_d;
                            flag^=1;
                        }
                    }
                    // 累加hessian非对角线
                    uint32_t idx_l=hash_idx[cur];
                    uint32_t idx_r=hash_idx[cur+(1<<td)];
                    #pragma unroll
                    for(uint32_t c=0;c<C;c++){
                        // 这里两个res-1分别表示gd、d坐标网格分辨率，cube所以相同
                        hessian[gd][d]+= (flag?1.0f:-1.0f)*(res-1)*(res-1)*voxel_smooth_derivative[gd]*voxel_smooth_derivative[d]*(1-voxel_smooth[td])*embeddings[idx_l*C+c]*grad_y[c];
                        hessian[gd][d]+= (flag?1.0f:-1.0f)*(res-1)*(res-1)*voxel_smooth_derivative[gd]*voxel_smooth_derivative[d]*  voxel_smooth[td]  *embeddings[idx_r*C+c]*grad_y[c];
                    }
                }
            }
        }
    }

    // 将local hessian累加到grad2_x上
    #pragma unroll
    for(uint32_t d=0;d<D;d++){
        #pragma unroll
        for(uint32_t gd=0;gd<D;gd++){
            // TODO: 这里没考虑scalar_t是half的情况，half原子操作很慢，但转__half2一起加需要pair，先懒得弄了。
            atomicAdd((scalar_t *)&grad2_x[d], (scalar_t)(grad2_grad_x[gd]*hessian[gd][d]));
        }
    }
}

/* ------------------------------------------- Channel Dispatch ------------------------------------------- */
// forward call kernel
template<typename scalar_t, uint32_t D>
void hash_encoder_forward_channelDispatch(const float *x, const scalar_t *embeddings, const uint32_t *offsets, const uint32_t *resolution_list, const uint32_t B, const uint32_t C, const uint32_t L, const bool if_cal_grad_x, const uint32_t AL, scalar_t *y, scalar_t *dy_dx){
    constexpr uint32_t num_threads=512;
    // 但forward好像没必要per C=2并行化，因为dy_dx求解本身很复杂，过度并行化会增加复杂度。所以下面N_C=C了
    const uint32_t N_C = MIN(2u, C); // 以N_C=2为单位切分特征维度C来进一步增加并行化，根据instant-ngp原论文,per_dim=2在3090上适配性能最高。
    const dim3 block_grid={divide_round_up<uint32_t>(B,num_threads),MIN(AL,L),C/C};
    switch(C){
        // call kernel to forward
        // x_thread: per input in batch
        // y_block: per level
        case 1: kernel_forward<scalar_t, D, 1, 1><<<block_grid, num_threads>>>(x, embeddings, offsets, resolution_list, B, L, if_cal_grad_x, y, dy_dx);break;
        case 2: kernel_forward<scalar_t, D, 2, 2><<<block_grid, num_threads>>>(x, embeddings, offsets, resolution_list, B, L, if_cal_grad_x, y, dy_dx);break;
        case 4: kernel_forward<scalar_t, D, 4, 4><<<block_grid, num_threads>>>(x, embeddings, offsets, resolution_list, B, L, if_cal_grad_x, y, dy_dx);break;
        case 8: kernel_forward<scalar_t, D, 8, 8><<<block_grid, num_threads>>>(x, embeddings, offsets, resolution_list, B, L, if_cal_grad_x, y, dy_dx);break;
        case 16: kernel_forward<scalar_t, D, 16, 16><<<block_grid, num_threads>>>(x, embeddings, offsets, resolution_list, B, L, if_cal_grad_x, y, dy_dx);break;
        default : throw std::runtime_error("[ERROR] per level feature dim only supports 1 2 4 8 16. ");
    }
}
// backward call kernel
template<typename scalar_t, uint32_t D>
void hash_encoder_backward_channelDispatch(const scalar_t *grad_y, const float *x, const scalar_t *dy_dx, const uint32_t *offsets, const uint32_t *resolution_list, const uint32_t B, const uint32_t C, const uint32_t L, const bool if_cal_grad_x, const uint32_t AL, scalar_t *grad_x, scalar_t *grad_embeddings){
    constexpr uint32_t num_threads=512;
    const uint32_t N_C=MIN(2u,C);
    const dim3 block_grid_embedding={divide_round_up<uint32_t>(B,num_threads),MIN(L,AL),C/C}; // embedding一阶backward计算复杂不宜过度并行化
    const dim3 block_grid_x={divide_round_up<uint32_t>(B,num_threads),D,C/C}; // 对x的一阶导引入per C并行化加速，会引入原子操作。
    switch(C){
        case 1: kernel_embedding_backward<scalar_t, D, 1, 1><<<block_grid_embedding, num_threads>>>(grad_y, x, offsets, resolution_list, B, L, grad_embeddings);
                if(if_cal_grad_x) kernel_x_backward<scalar_t, D, 1, 1><<<block_grid_x, num_threads>>>(grad_y, dy_dx, B, L, AL, grad_x);break;
        case 2: kernel_embedding_backward<scalar_t, D, 2, 2><<<block_grid_embedding, num_threads>>>(grad_y, x, offsets, resolution_list, B, L, grad_embeddings);
                if(if_cal_grad_x) kernel_x_backward<scalar_t, D, 2, 2><<<block_grid_x, num_threads>>>(grad_y, dy_dx, B, L, AL, grad_x);break;
        case 4: kernel_embedding_backward<scalar_t, D, 4, 4><<<block_grid_embedding, num_threads>>>(grad_y, x, offsets, resolution_list, B, L, grad_embeddings);
                if(if_cal_grad_x) kernel_x_backward<scalar_t, D, 4, 4><<<block_grid_x, num_threads>>>(grad_y, dy_dx, B, L, AL, grad_x);break;
        case 8: kernel_embedding_backward<scalar_t, D, 8, 8><<<block_grid_embedding, num_threads>>>(grad_y, x, offsets, resolution_list, B, L, grad_embeddings);
                if(if_cal_grad_x) kernel_x_backward<scalar_t, D, 8, 8><<<block_grid_x, num_threads>>>(grad_y, dy_dx, B, L, AL, grad_x);break;
        case 16: kernel_embedding_backward<scalar_t, D, 16, 16><<<block_grid_embedding, num_threads>>>(grad_y, x, offsets, resolution_list, B, L, grad_embeddings);
                if(if_cal_grad_x) kernel_x_backward<scalar_t, D, 16, 16><<<block_grid_x, num_threads>>>(grad_y, dy_dx, B, L, AL, grad_x);break;
        default : throw std::runtime_error("[ERROR] per level feature dim only supports 1 2 4 8 16. ");
    }
}
// second backward call kernel
template<typename scalar_t, uint32_t D>
void hash_encoder_second_backward_channelDispatch(const scalar_t *grad2_grad_x, const float *x, const scalar_t *embeddings, const scalar_t *grad_y, const scalar_t *dy_dx, const uint32_t *offsets, const uint32_t *resolution_list, const uint32_t B, const uint32_t C, const uint32_t L, const bool if_cal_hessian_x, const uint32_t AL, scalar_t *grad2_embeddings, scalar_t *grad2_grad_y, scalar_t *grad2_x){
    constexpr uint32_t num_threads=512;
    const uint32_t N_C=MIN(2u,C);
    const dim3 block_grid={divide_round_up<uint32_t>(B,num_threads),MIN(L,AL),C/C};
    switch (C){
        case 1: kernel_embedding_second_backward<scalar_t, D, 1, 1><<<block_grid, num_threads>>>(grad2_grad_x, x, grad_y, offsets, resolution_list, B, L, grad2_embeddings);
                kernel_grad_y_second_backward<scalar_t, D, 1, 1><<<block_grid, num_threads>>>(grad2_grad_x, dy_dx, B, L, grad2_grad_y);
                if(if_cal_hessian_x) kernel_grad2_x_second_backward<scalar_t, D, 1><<<block_grid, num_threads>>>(grad2_grad_x, x, embeddings, grad_y, offsets, resolution_list, B, L, grad2_x);
                break;
        case 2: kernel_embedding_second_backward<scalar_t, D, 2, 2><<<block_grid, num_threads>>>(grad2_grad_x, x, grad_y, offsets, resolution_list, B, L, grad2_embeddings);
                kernel_grad_y_second_backward<scalar_t, D, 2, 2><<<block_grid, num_threads>>>(grad2_grad_x, dy_dx, B, L, grad2_grad_y);
                if(if_cal_hessian_x) kernel_grad2_x_second_backward<scalar_t, D, 2><<<block_grid, num_threads>>>(grad2_grad_x, x, embeddings, grad_y, offsets, resolution_list, B, L, grad2_x);
                break;
        case 4: kernel_embedding_second_backward<scalar_t, D, 4, 4><<<block_grid, num_threads>>>(grad2_grad_x, x, grad_y, offsets, resolution_list, B, L, grad2_embeddings);
                kernel_grad_y_second_backward<scalar_t, D, 4, 4><<<block_grid, num_threads>>>(grad2_grad_x, dy_dx, B, L, grad2_grad_y);
                if(if_cal_hessian_x) kernel_grad2_x_second_backward<scalar_t, D, 4><<<block_grid, num_threads>>>(grad2_grad_x, x, embeddings, grad_y, offsets, resolution_list, B, L, grad2_x);
                break;
        case 8: kernel_embedding_second_backward<scalar_t, D, 8, 8><<<block_grid, num_threads>>>(grad2_grad_x, x, grad_y, offsets, resolution_list, B, L, grad2_embeddings);
                kernel_grad_y_second_backward<scalar_t, D, 8, 8><<<block_grid, num_threads>>>(grad2_grad_x, dy_dx, B, L, grad2_grad_y);
                if(if_cal_hessian_x) kernel_grad2_x_second_backward<scalar_t, D, 8><<<block_grid, num_threads>>>(grad2_grad_x, x, embeddings, grad_y, offsets, resolution_list, B, L, grad2_x);
                break;
        case 16: kernel_embedding_second_backward<scalar_t, D, 16, 16><<<block_grid, num_threads>>>(grad2_grad_x, x, grad_y, offsets, resolution_list, B, L, grad2_embeddings);
                kernel_grad_y_second_backward<scalar_t, D, 16, 16><<<block_grid, num_threads>>>(grad2_grad_x, dy_dx, B, L, grad2_grad_y);
                if(if_cal_hessian_x) kernel_grad2_x_second_backward<scalar_t, D, 16><<<block_grid, num_threads>>>(grad2_grad_x, x, embeddings, grad_y, offsets, resolution_list, B, L, grad2_x);
                break;
        default : throw std::runtime_error("[ERROR] per level feature dim only supports 1 2 4 8 16. ");
    }
}
/* ------------------------------------------- Dim Dispatch ------------------------------------------- */
// 暂时规定只能插值2/3维grid/cube
template<typename scalar_t>
void hash_encoder_forward_dimDispatch(const float *x, const scalar_t *embeddings, const uint32_t *offsets, const uint32_t *resolution_list, const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L, const bool if_cal_grad_x, const uint32_t AL, scalar_t *y, scalar_t *dy_dx){
    switch (D)
    {  
        case 2: hash_encoder_forward_channelDispatch<scalar_t,2>(x, embeddings, offsets, resolution_list, B, C, L, if_cal_grad_x, AL, y, dy_dx);break;
        case 3: hash_encoder_forward_channelDispatch<scalar_t,3>(x, embeddings, offsets, resolution_list, B, C, L, if_cal_grad_x, AL, y, dy_dx);break;
        default: throw std::runtime_error("[ERROR] doesn't support %ddim input coordinate. ");
    }
}
template<typename scalar_t>
void hash_encoder_backward_dimDispatch(const scalar_t *grad_y, const float *x, const scalar_t *dy_dx, const uint32_t *offsets, const uint32_t *resolution_list, const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L, const bool if_cal_grad_x, const uint32_t AL, scalar_t *grad_x, scalar_t *grad_embeddings){
    switch (D)
    {
        case 2: hash_encoder_backward_channelDispatch<scalar_t,2>(grad_y, x, dy_dx, offsets, resolution_list, B, C, L, if_cal_grad_x, AL, grad_x, grad_embeddings);break;
        case 3: hash_encoder_backward_channelDispatch<scalar_t,3>(grad_y, x, dy_dx, offsets, resolution_list, B, C, L, if_cal_grad_x, AL, grad_x, grad_embeddings);break;
        default: throw std::runtime_error("[ERROR] doesn't support %ddim input coordinate. ");
    }
}
template<typename scalar_t>
void hash_encoder_second_backward_dimDispatch(const scalar_t *grad2_grad_x, const float *x, const scalar_t *embeddings, const scalar_t *grad_y, const scalar_t *dy_dx, const uint32_t *offsets, const uint32_t *resolution_list, const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L, const bool if_cal_hessian_x, const uint32_t AL, scalar_t *grad2_embeddings, scalar_t *grad2_grad_y, scalar_t *grad2_x){
    switch (D)
    {
        case 2: hash_encoder_second_backward_channelDispatch<scalar_t,2>(grad2_grad_x, x, embeddings, grad_y, dy_dx, offsets, resolution_list, B, C, L, if_cal_hessian_x, AL, grad2_embeddings, grad2_grad_y, grad2_x);break;
        case 3: hash_encoder_second_backward_channelDispatch<scalar_t,3>(grad2_grad_x, x, embeddings, grad_y, dy_dx, offsets, resolution_list, B, C, L, if_cal_hessian_x, AL, grad2_embeddings, grad2_grad_y, grad2_x);break;
        default: throw std::runtime_error("[ERROR] doesn't support %ddim input coordinate. ");
    }
}

/* ------------------------------------------- Launcher(Type Dispatch) ------------------------------------------- */
void hash_encoder_forward(const at::Tensor x, const at::Tensor embeddings, const at::Tensor offsets, const at::Tensor resolution_list ,const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L, const bool if_cal_grad_x, const uint32_t AL, at::Tensor y, at::Tensor dy_dx){
    CHECK_CUDA(x);CHECK_CUDA(embeddings);CHECK_CUDA(offsets);CHECK_CUDA(resolution_list);CHECK_CUDA(y);
    CHECK_CONTIGUOUS(x);CHECK_CONTIGUOUS(embeddings);CHECK_CONTIGUOUS(offsets);CHECK_CONTIGUOUS(resolution_list);CHECK_CONTIGUOUS(y);
    CHECK_FLOAT(x);CHECK_FLOAT(embeddings);CHECK_FLOAT(y);
    CHECK_INT(offsets);CHECK_INT(resolution_list);

    // type dispatch: using torch micro
    // scalar_t: c10::Half, float, double
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        embeddings.scalar_type(),"hash_encoder_forward",[&]{
            hash_encoder_forward_dimDispatch<scalar_t>(x.data_ptr<float>(),embeddings.data_ptr<scalar_t>(),(uint32_t*)offsets.data_ptr<int>(),(uint32_t*)resolution_list.data_ptr<int>(),B,D,C,L,if_cal_grad_x, AL, y.data_ptr<scalar_t>(), dy_dx.data_ptr<scalar_t>());
        }
    );

    // // type dispatch: switch
    // switch (embeddings.scalar_type())
    // {
    //     case at::ScalarType::Half: hash_encoder_forward_dimDispatch<c10::Half>(x.data_ptr<float>(),embeddings.data_ptr<c10::Half>(),offsets.data_ptr<uint32_t>(),resolution_list.data_ptr<uint32_t>(),B,D,C,L,if_cal_grad_x, y.data_ptr<c10::Half>(), dy_dx.data_ptr<c10::Half>());break;
    //     case at::ScalarType::Float: hash_encoder_forward_dimDispatch<float>(x.data_ptr<float>(),embeddings.data_ptr<float>(),offsets.data_ptr<uint32_t>(),resolution_list.data_ptr<uint32_t>(),B,D,C,L,y.data_ptr<float>());break;
    //     case at::ScalarType::Double: hash_encoder_forward_dimDispatch<double>(x.data_ptr<float>(),embeddings.data_ptr<double>(),offsets.data_ptr<uint32_t>(),resolution_list.data_ptr<uint32_t>(),B,D,C,L,y.data_ptr<double>());break;
    //     default: throw std::runtime_error("[ERROR] only support float16, float32, float64. ");
    // };
}

void hash_encoder_backward(const at::Tensor grad_y, const at::Tensor x, const at::Tensor dy_dx, const at::Tensor offsets, const at::Tensor resolution_list,  const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L, const bool if_cal_grad_x, const uint32_t AL, at::Tensor grad_x, at::Tensor grad_embeddings){
    CHECK_CUDA(grad_y);CHECK_CUDA(x);CHECK_CUDA(dy_dx);CHECK_CUDA(offsets);CHECK_CUDA(resolution_list);CHECK_CUDA(grad_x);CHECK_CUDA(grad_embeddings);
    CHECK_CONTIGUOUS(grad_y);CHECK_CONTIGUOUS(x);CHECK_CONTIGUOUS(dy_dx);CHECK_CONTIGUOUS(offsets);CHECK_CONTIGUOUS(resolution_list);CHECK_CONTIGUOUS(grad_x);CHECK_CONTIGUOUS(grad_embeddings);
    CHECK_FLOAT(grad_y);CHECK_FLOAT(x);CHECK_FLOAT(dy_dx);CHECK_FLOAT(grad_x);CHECK_FLOAT(grad_embeddings);
    CHECK_INT(offsets);CHECK_INT(resolution_list);

    // type dispatch: using torch micro
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        grad_y.scalar_type(),"hash_encoder_backward",[&]{
            hash_encoder_backward_dimDispatch<scalar_t>(grad_y.data_ptr<scalar_t>(),x.data_ptr<float>(),dy_dx.data_ptr<scalar_t>(),(uint32_t*)offsets.data_ptr<int>(),(uint32_t*)resolution_list.data_ptr<int>(),B,D,C,L,if_cal_grad_x,AL,grad_x.data_ptr<scalar_t>(), grad_embeddings.data_ptr<scalar_t>());
        }
    );
}

void hash_encoder_second_backward(const at::Tensor grad2_grad_x, const at::Tensor x, const at::Tensor embeddings, const at::Tensor grad_y, const at::Tensor dy_dx,  const at::Tensor offsets, const at::Tensor resolution_list,  const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L, const bool if_cal_hessian_x, const uint32_t AL, at::Tensor grad2_embeddings, at::Tensor grad2_grad_y, at::Tensor grad2_x){
    CHECK_CUDA(grad2_grad_x);CHECK_CUDA(x);CHECK_CUDA(embeddings);CHECK_CUDA(grad_y);CHECK_CUDA(dy_dx);CHECK_CUDA(offsets);CHECK_CUDA(resolution_list);CHECK_CUDA(grad2_embeddings);CHECK_CUDA(grad2_grad_y);CHECK_CUDA(grad2_x);
    CHECK_CONTIGUOUS(grad2_grad_x);CHECK_CONTIGUOUS(x);CHECK_CONTIGUOUS(embeddings);CHECK_CONTIGUOUS(grad_y);CHECK_CONTIGUOUS(dy_dx);CHECK_CONTIGUOUS(offsets);CHECK_CONTIGUOUS(resolution_list);CHECK_CONTIGUOUS(grad2_embeddings);CHECK_CONTIGUOUS(grad2_grad_y);CHECK_CONTIGUOUS(grad2_x);
    CHECK_FLOAT(grad2_grad_x);CHECK_FLOAT(x);CHECK_FLOAT(embeddings);CHECK_FLOAT(grad_y);CHECK_FLOAT(dy_dx);CHECK_FLOAT(grad2_embeddings);CHECK_FLOAT(grad2_grad_y);CHECK_FLOAT(grad2_x);
    CHECK_INT(offsets);CHECK_INT(resolution_list);

    // type dispatch: using torch micro
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        grad2_grad_x.scalar_type(),"hash_encoder_second_backward",[&]{
            hash_encoder_second_backward_dimDispatch<scalar_t>(grad2_grad_x.data_ptr<scalar_t>(),x.data_ptr<float>(),embeddings.data_ptr<scalar_t>(),grad_y.data_ptr<scalar_t>(),dy_dx.data_ptr<scalar_t>(),(uint32_t*)offsets.data_ptr<int>(),(uint32_t*)resolution_list.data_ptr<int>(),B,D,C,L,if_cal_hessian_x,AL,grad2_embeddings.data_ptr<scalar_t>(),grad2_grad_y.data_ptr<scalar_t>(),grad2_x.data_ptr<scalar_t>());
        }
    );
}