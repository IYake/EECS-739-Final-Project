/*
  * mutlShare.cu
  * 
  * Robert Hochberg
  * January 24, 2012
  * 
  * Based nearly entirely on the code from the CUDA C Programming Guide
  */

#include "multShare.h"

// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatMul(const Matrix A, const Matrix B, Matrix C) {
// Load A and B to device memory
  Matrix d_A;
  d_A.width = d_A.stride = A.width;
  d_A.height = A.height;
  size_t size = A.width * A.height * sizeof(float);
  cudaError_t err = cudaMalloc(&d_A.elements, size);
  //printf("CUDA malloc A: %s\n",cudaGetErrorString(err));
  cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);

  Matrix d_B;
  d_B.width = d_B.stride = B.width;
  d_B.height = B.height;
  size = B.width * B.height * sizeof(float);
  err = cudaMalloc(&d_B.elements, size);
  //printf("CUDA malloc B: %s\n",cudaGetErrorString(err));
  cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);

  // Allocate C in device memory
  Matrix d_C;
  d_C.width = d_C.stride = C.width;
  d_C.height = C.height;
  size = C.width * C.height * sizeof(float);
  err = cudaMalloc(&d_C.elements, size);
  //printf("CUDA malloc C: %s\n",cudaGetErrorString(err));

  // Invoke kernel
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
  MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
  err = cudaThreadSynchronize();
  //printf("Run kernel: %s\n", cudaGetErrorString(err));

  // Read C from device memory
  err = cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost);
  //printf("Copy C off of device: %s\n",cudaGetErrorString(err));

  // Free device memory
  cudaFree(d_A.elements);
  cudaFree(d_B.elements);
  cudaFree(d_C.elements);
}

// Get a matrix element
__device__ float GetElement(const Matrix A, int row, int col) {
  return A.elements[row * A.stride + col];
}

// Set a matrix element
__device__ void SetElement(Matrix A, int row, int col, float value) {
  A.elements[row * A.stride + col] = value;
}

// Get the BLOCK_SIZExBLOCK_SIZE sub-matrix Asub of A that is
// located col sub-matrices to the right and row sub-matrices down
// from the upper-left corner of A
__device__ Matrix GetSubMatrix(Matrix A, int row, int col) {
  Matrix Asub;
  Asub.width = BLOCK_SIZE;
  Asub.height = BLOCK_SIZE;
  Asub.stride = A.stride;
  Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row + BLOCK_SIZE * col];
  return Asub;
}

// Matrix multiplication kernel called by MatMul()
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C) {
  // Block row and column
  int blockRow = blockIdx.y;
  int blockCol = blockIdx.x;

  // Each thread block computes one sub-matrix Csub of C
  Matrix Csub = GetSubMatrix(C, blockRow, blockCol);

  // Each thread computes one element of Csub
  // by accumulating results into Cvalue
  float Cvalue = 0.0;

  // Thread row and column within Csub
  int row = threadIdx.y;
  int col = threadIdx.x;

  // Loop over all the sub-matrices of A and B that are
  // required to compute Csub
  // Multiply each pair of sub-matrices together
  // and accumulate the results
  for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) {
    // Get sub-matrix Asub of A
    Matrix Asub = GetSubMatrix(A, blockRow, m);

    // Get sub-matrix Bsub of B
    Matrix Bsub = GetSubMatrix(B, m, blockCol);

    // Shared memory used to store Asub and Bsub respectively
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];  
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    // Load Asub and Bsub from device memory to shared memory
    // Each thread loads one element of each sub-matrix
    As[row][col] = GetElement(Asub, row, col);
    Bs[row][col] = GetElement(Bsub, row, col);

    // Synchronize to make sure the sub-matrices are loaded
    // before starting the computation
    __syncthreads();

    // Multiply Asub and Bsub together
    for (int e = 0; e < BLOCK_SIZE; ++e)
      Cvalue += As[row][e] * Bs[e][col];

    // Synchronize to make sure that the preceding 
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    __syncthreads();
  }

  // Write Csub to device memory
  // Each thread writes one element
  SetElement(Csub, row, col, Cvalue);
}

float funcOneTrue(float x){
  //function in this example is linear
  return 2*x;
}

float funcOneP(float x, Matrix b){
  return b.elements[0]*x+b.elements[1];
}

float funcOneD(float x, int param, Matrix b){
  //since this is linear: b_0*x+b_1
  if (param == 0)
	return b.elements[0];
  else
    return 0;
}

float funcTwoTrue(float x){
  //function in this example is quadratic
  return 4*x*x+6*x;
}

float funcTwoP(float x, Matrix b){
  return b.elements[0]*x*x+b.elements[1]*x;
}

float funcTwoD(float x, int param, Matrix b){
  //since this is linear: b_0*x+b_1
  if (param == 0)
	return 2*b.elements[0]*x;
  else
    return b.elements[1];
}

float funcThreeTrue(float t){
  //exponential example, x_1*e^(x_2)*t
  double x_1 = 2.5411;
  double x_2 = 0.2595;
  double temp = t;
  
  return x_1*exp(x_2*temp);
}

float funcThreeP(float t, Matrix b){
  return b.elements[0]*exp(b.elements[1]*t);
}

float funcThreeD(float t, int param, Matrix b){
  if (param == 0)
    return exp(b.elements[1]*t);
  else //param == 1
  {
    double x_1 = b.elements[0];
  double x_2 = b.elements[1];
  double temp = t;
  
  return temp*x_1*exp(x_2*temp);
	}
}

void print(Matrix A){
  for(int i = 0; i < min(10, A.height); i++){
    for(int j = 0; j < min(10, A.width); j++)
      printf("%f ", A.elements[i*A.width + j]);
    printf("\n");
  }
  printf("\n");
}

void init(Matrix* A, int height, int width){
  A->height = height;
  A->width = width;
  A->elements = (float*)malloc(A->width * A->height * sizeof(float));
}

void copy(Matrix* A, Matrix* B){
//must be the same size
  for(int i = 0; i < A->height; i++)
    for(int j = 0; j < A->width; j++)
      A->elements[i*A->width + j] = B->elements[i*A->width + j];
}

Matrix T(Matrix A){
	Matrix transpose;
	init(&transpose,A.width,A.height);
	for(int i = 0; i < A.width; i++)
      for(int j = 0; j < A.height; j++)
	    transpose.elements[i*transpose.width + j] = A.elements[j*A.width + i];
	return transpose;
}


//make this efficient with cuda later
//Matrix pseudoinv(Matrix A){

//}

//Matrix inverse

void getCofactor(Matrix* A, Matrix* temp, int p, int q, int n) 
{
	int i = 0;
	int j = 0;
    for(int row = 0; row < n; row++)
	{
		for(int col = 0; col < n; col++)
        { 
            //  Copying into temporary matrix only those element 
            //  which are not in given row and column 
            if (row != p && col != q) 
            { 
                temp->elements[i*temp->width + j++] = A->elements[row*A->width+col]; 
  
                // Row is filled, so increase row index and 
                // reset col index 
                if (j == n - 1) 
                { 
                    j = 0; 
                    i++; 
                } 
            } 
        } 
    } 
}

/* Recursive function for finding determinant of matrix. 
   n is current dimension of A[][]. */
int determinant(Matrix* A, int n) 
{ 
    int D = 0; // Initialize result 
  
    //  Base case : if matrix contains single element 
	//printf("This is A: ");
	//print(*A);
    if (n == 1) 
        return A->elements[0]; 
	//printf("A->elements[0]: %f\n",A->elements[0]);
	Matrix temp;
	init(&temp,n,n);// To store cofactors 
  
    int sign = 1;  // To store sign multiplier 
  
     // Iterate for each element of first row 
    for (int f = 0; f < n; f++) 
    { 
        // Getting Cofactor of A[0][f] 
        getCofactor(A, &temp, 0, f, n); 
		//if (n == A->width)
			//print(temp);
        D += sign * A->elements[0*A->width+f] * determinant(&temp, n - 1); 
  
        // terms are to be added with alternate sign 
        sign = -sign;
		//printf("D: %d\n",D);
		//printf("A->elements[0*A->width+f]: %f\n",A->elements[0*A->width+f]);
    } 
  
    return D; 
}

void adjoint(Matrix* A, Matrix* Adj) 
{ 
	int N = A->width;
    if (N == 1) 
    { 
        Adj->elements[0] = 1; 
        return; 
    } 
  
    // temp is used to store cofactors of A[][] 
	Matrix temp;
	init(&temp, N,N);
    int sign = 1; 
  
    for (int i=0; i<N; i++) 
    { 
        for (int j=0; j<N; j++) 
        { 
            // Get cofactor of A[i][j] 
            getCofactor(A, &temp, i, j, N); 
  
            // sign of adj[j][i] positive if sum of row 
            // and column indexes is even. 
            sign = ((i+j)%2==0)? 1: -1; 
  
            // Interchanging rows and columns to get the 
            // transpose of the cofactor matrix 
            Adj->elements[j*Adj->width+i] = (sign)*(determinant(&temp, N-1)); 
        } 
    } 
}

// Function to calculate and store inverse, returns false if 
// matrix is singular 
int inverse(Matrix* A, Matrix* Inv) 
{ 
	//printf("A in inverse: ");
	//print(*A);
	int N = A->width;
    // Find determinant of A[][] 
    int det = determinant(A, N); 
    if (det == 0) 
    { 
        printf("Singular matrix, can't find its inverse\n");
        return 0; 
    } 
  
    // Find adjoint 
	Matrix Adj;
	init(&Adj,N,N); 
    adjoint(A, &Adj); 
  
    // Find Inverse using formula "inverse(A) = adj(A)/det(A)" 
    for (int i=0; i<N; i++) 
        for (int j=0; j<N; j++) 
            Inv->elements[i*N+j] = Adj.elements[i*N+j]/float(det); 
    return 1; 
}



Matrix subtract(Matrix A, Matrix B){
	Matrix C;
	if (A.height != B.height || A.width != B.width)
	  printf("Subtract error: dimensions of matrices don't match\n");
	init(&C,A.height,A.width);
	double alpha = 0.1;
	for (int i = 0; i < C.height; i++)
	  for (int j = 0; j < C.width; j++)
		C.elements[i*C.width+j] = A.elements[i*C.width+j] - alpha*B.elements[i*C.width+j];
	
	return C;
}

void MatMul_(Matrix* A, Matrix* B, Matrix* C){
  for (int i = 0; i < A->height; i++)
    for (int j = 0; j < B->width; j++){
	  for (int k = 0; k < A->width; k++)
	    C->elements[i*C->width+j]+=A->elements[i*A->width+k]*B->elements[k*B->width+j];
	  //printf("C: ");
	  //print(*C);
	 }
}

//INPUT MATRIX MUST HAVE DIMENSIONS THAT ARE MULTIPLES OF THE BLOCK SIZE (= 2)
Matrix pseudoInv(Matrix A){
	Matrix pseudo, inv,temp,temp2;
	init(&pseudo,A.width,A.height);
	init(&inv,A.width,A.width);
	init(&temp,A.width,A.width);
	init(&temp2,A.width,A.height);
	MatMul(T(A),A,temp);
	//printf("T(A): \n");
	//print(T(A));
	
	//printf("\nA: ");
	//print(A);
	printf("T(A)*A: \n");
	print(temp);
	inverse(&temp,&inv);
	printf("inverse(temp): \n");
	//print(inv);
	
	MatMul(inv,T(A),pseudo);
	
	//printf("sending pseudo: \n");
	//print(pseudo);
	return pseudo;
	
}

int main(int argc, char* argv[]){
  Matrix x, y, b, J, b_prev, r, temp, inv,temp2, temp3, temp4, temp_A, temp_B,identity, temp5;
  int N = 5;
  int num_params = 2;
  
  init(&x,N,1);
  init(&y,N,1);
  init(&r,N,1);
  init(&b,1,num_params);
  init(&b_prev,1,num_params);
  init(&J,N,num_params);
  init(&temp,N,1);
  init(&inv,N,N);
  init(&temp2,J.width,r.width);
  //init(&temp3,J.width,J.height);
  init(&temp4,b.height,b.width);
  init(&identity,J.width,J.width);
  init(&temp_A,num_params,num_params);
  init(&temp_B,num_params,1);
  init(&temp5,num_params,num_params);
  
  temp.elements[0] = 1;
  temp.elements[1] = 2;
  temp.elements[2] = 4;
  temp.elements[3] = 5;
  temp.elements[4] = 8;
  

  for(int i = 0; i < x.height; i++)
    for(int j = 0; j < x.width; j++){
      x.elements[i*x.width + j] = temp.elements[i];
      y.elements[i*y.width + j] = funcThreeTrue(x.elements[i*x.width+j]);
	  printf("f(%f) = %f\n", x.elements[i*x.width + j],funcThreeTrue(x.elements[i*x.width+j]));
	}

//  for(int i = 0; i < b.height; i++)
//    for(int j = 0; j < b.width; j++){
//      b.elements[i*b.width + j] = 1;
//      b_prev.elements[i*b_prev.width + j] = b.elements[i*b_prev.width + j];
//	}
	b.elements[0] = 2.5;
	b.elements[1] = 0.25;
	//b.elements[0] = 3.0;
	//b.elements[1] = 1.0;

  for(int i = 0; i < J.height; i++) //for each equation
    for(int j = 0; j < J.width; j++) //for each parameter
       J.elements[i*J.width + j] = funcThreeD(x.elements[i*x.width],j, b);
		

  printf("true x:\n");
  print(x);
  printf("true y:\n");
  print(y);
  printf("initial b:\n");
  print(b);
  printf("initial J:\n");
  print(J);
  //printf("Finished initializing...\n");
  ////convergence loop
  //copy(&b,&b_prev);
  for (int iter = 0; iter < 20; iter++){
	  for (int i = 0; i < N; i++)
		r.elements[i] = funcThreeP(x.elements[i],b) - y.elements[i];
		//printf("predict(%f) = %f\n",x.elements[i],funcThreeP(x.elements[i],b));
	
	  //printf("r:\n");
	  //print(r);
	  
	  //printf("J:\n");
	  //print(J);
	  
	  temp4 = T(J);
	  MatMul_(&temp4,&r,&temp_B);
	  //printf("temp_B:\n");
	  //print(temp_B);
	  
	  
	  init(&temp3,num_params,1);
	  MatMul_(&temp4,&J,&temp_A);
	  
	  //printf("temp_A:\n");
	  //print(temp_A);
	  
	  
	  
	  
	  
	  
	  inverse(&temp_A,&temp5);
	  //printf("temp5: \n");
	  //print(temp5);
	  MatMul_(&temp5,&temp_B,&temp3);
	  
	  
	  //printf("temp3: \n");
	  //print(temp3);
	  //temp3 = pseudoInv(J);
	  //MatMul_(&temp3,&r,&temp3);
	  
	  //printf("J^-1:\n");
	  //print(temp3);
	  
	  printf("pseudo * r:\n");
	  if (temp3.elements[0]+temp3.elements[1] > 1){
		temp3.elements[0] = temp3.elements[0]/(temp3.elements[0]+temp3.elements[1]);
		temp3.elements[1] = temp3.elements[1]/(temp3.elements[0]+temp3.elements[1]);
		}
	
	  //NORMALIZATION ISN"T WORKING
	  print(temp3);
	  
	  b = subtract(b,T(temp3));
	  //SEE IF THIS SUBTRACT IS WORKING
	  //WHY DOES B ALMOST ALWAYS INCREASE
	  //DIVERGES FOR LARGE ITERATIONS
	  
	  for(int i = 0; i < J.height; i++) //for each equation
	    for(int j = 0; j < J.width; j++) //for each parameter
          J.elements[i*J.width + j] = funcThreeD(x.elements[i*x.width],j, b);
	  printf("b updated:\n");
	  print(b);
  }
}














