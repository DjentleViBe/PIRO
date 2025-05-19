__kernel void prefixsum(__global const int* input,
                        __global int* output,
                        const int N,
                        const int row){
    
    int index = get_global_id(0);  
    if (index >= N || index < row) return;
    if(index == 0){
        output[index + 1] = input[index];
    }
    else{
        output[index + 1] = input[index] + output[index];  
    }
}