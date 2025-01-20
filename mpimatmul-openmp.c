#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <mpi.h>

#include <omp.h>

typedef struct Matrix {
    uint64_t total_height, total_width;
    uint64_t height_start, height_stop;
    uint64_t width_start, width_stop;
    uint64_t block_size;
    double* mat;
} Matrix;

/**
 * read a sqare chunk of a sqare matrix depending on the rank of the process
 * params:
 *      mpi_line_size: the number of processes per line assuming a sqare grid
 *      mpi_rank: the rank of the current process
*/
Matrix read_mat(char* filename, int mpi_line_size, int mpi_rank, int mpi_proc_line, int mpi_proc_col) {
    FILE* f = fopen(filename, "rb");
    if (f == NULL) {
        printf("[%d] Could not open for read: %s\n", mpi_rank, filename);
        exit(1);
    }

    Matrix mat;
    fread(&(mat.total_height), sizeof(uint64_t), 1, f);
    fread(&(mat.total_width), sizeof(uint64_t), 1, f);

    if (mat.total_height * mat.total_width == 0 || mat.total_height != mat.total_width) {
        printf("[%d] The invalid size (%ld x %ld) for matrix %s\nNeeds to be a square matrix!\n", mpi_rank, mat.total_height, mat.total_width, filename);
        fclose(f);
        MPI_Finalize();
        exit(1);
    }

    if (mat.total_height % mpi_line_size != 0) {
        printf("[%d] The choosen number of processes doesn't divide '%s' exactly\n", mpi_rank, filename);
        fclose(f);
        MPI_Finalize();
        exit(1);
    }

    mat.block_size = mat.total_height / mpi_line_size;

    mat.height_start = mat.block_size * mpi_proc_line;
    mat.height_stop = mat.block_size * (mpi_proc_line + 1);

    mat.width_start = mat.block_size * mpi_proc_col;
    mat.width_stop = mat.block_size * (mpi_proc_col + 1);

    mat.mat = (double*) malloc (mat.block_size * mat.block_size * sizeof(double));
    if (mat.mat == NULL) {
        printf("[%d] Could not allocate memory for '%s'\n", mpi_rank, filename);
        fclose(f);
        exit(1);
    }

    for (uint64_t i = 0; i < mat.block_size; i++) {
        if (fseek(f, 8 * (2 + (i + mat.height_start) * mat.total_width + mat.width_start), SEEK_SET) != 0) {
            printf("[%d] Could not fseek in '%s' row %lu", mpi_rank, filename, i);
            fclose(f);
            free(mat.mat);
            exit(1);
        }

        if (fread(&(mat.mat[i * mat.block_size]), sizeof(double), mat.block_size, f) != mat.block_size) {
            printf("[%d] Couldn't read all elements from row %lu from '%s'!\n", mpi_rank, i, filename);
            if(feof(f)) printf("[%d] EOF\n", mpi_rank);
            if(ferror(f)) printf("[%d] File error\n", mpi_rank);
            fclose(f);
            free(mat.mat);
            exit(1);
        }
    }

    fclose(f);
    return mat;
}

void write_mat(char* filename, Matrix mat, int mpi_rank) {
    FILE* f = fopen(filename, "rb+");
    if (f == NULL) {
        printf("[%d] Could not open for write: %s\n", mpi_rank, filename);
        exit(1);
    }

    if (mpi_rank == 0) {
        fwrite(&(mat.total_height), sizeof(uint64_t), 1, f);
        fwrite(&(mat.total_width), sizeof(uint64_t), 1, f);
    }

    for (uint64_t i = 0; i < mat.block_size; i++) {
        if (fseek(f, 8 * (2 + (i + mat.height_start) * mat.total_width + mat.width_start), SEEK_SET) != 0) {
            printf("[%d] Could not fseek in '%s' row %lu", mpi_rank, filename, i);
            fclose(f);
            exit(1);
        }

        if (fwrite(&(mat.mat[i * mat.block_size]), sizeof(double), mat.block_size, f) != mat.block_size) {
            printf("[%d] Couldn't write all elements from row %lu, from '%s'!\n", mpi_rank, i, filename);
            if(feof(f)) printf("[%d] EOF\n", mpi_rank);
            if(ferror(f)) printf("[%d] File error\n", mpi_rank);
            fclose(f);
            exit(1);
        }
    }

    fclose(f);
}

int main(int argc, char *argv[])
{
    MPI_Status status;
    
    int rank, size;
    int left, right, up, down;
    
    double *A, *B, *C, *buf, *tmp;

    int sqrt_size, Nl, proc_coords[2], aux;
    
    MPI_Init(&argc, &argv);
    double before_read = MPI_Wtime();

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    sqrt_size = sqrt(size);
    if (sqrt_size * sqrt_size != size) {
        if (rank == 0)
            printf("The number of processes must be a perfect square!\n");
        MPI_Finalize();
        return 1;
    }

    if (argc < 5) {
        if (rank == 0) 
            printf("Usage: %s [mat1_file] [mat2_file] [out_file] [num_treads]\n", argv[0]);
        MPI_Finalize();
        return 1;
    }
    
    int num_threads = atoi(argv[4]);
    num_threads = num_threads > 0 ? num_threads : 1;
    
    int dims[2], periods[2];
    MPI_Comm cannon_comm;

    dims[0]=0;
    dims[1]=0;
    periods[0]=1;
    periods[1]=1;
    
    MPI_Dims_create(size, 2, dims);
    
    if (dims[0] != dims[1]) {
        if (rank == 0) 
        	printf("The number of processors must be a square.\n");
        MPI_Finalize();
        return 0;
    }

    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cannon_comm);

    MPI_Cart_coords(cannon_comm, rank, 2, proc_coords);
    MPI_Cart_shift(cannon_comm, 0, 1, &left, &right);
    MPI_Cart_shift(cannon_comm, 1, 1, &up, &down);

    // mpi is a lier, the process coords that it returns are row major, but if you reconstruct the matrix based on the neighbors the matrix is in fact column major!
    aux = proc_coords[0];
    proc_coords[0] = proc_coords[1];
    proc_coords[1] = aux;

    int A_row, A_col, B_row, B_col;
    A_row = proc_coords[0];
    A_col = (proc_coords[1] + proc_coords[0]) % sqrt_size;
    B_row = (proc_coords[1] + proc_coords[0]) % sqrt_size;
    B_col = proc_coords[1];

    // Reading can be performed completely independent, no need to use a barrier
    // MPI_Barrier(cannon_comm);

    Matrix mat1 = read_mat(argv[1], sqrt_size, rank, A_row, A_col);
    Matrix mat2 = read_mat(argv[2], sqrt_size, rank, B_row, B_col);

    if (mat1.total_height != mat2.total_height) {
        if (rank == 0)
            printf("Input matrices do not have the same dimensions");
        free(mat1.mat);
        free(mat2.mat);
        MPI_Finalize();
        return 1;
    }

    Nl = mat1.block_size;
    
    A = mat1.mat;
    B = mat2.mat;
    C = (double*) calloc(Nl * Nl, sizeof(double));
    buf = (double*) malloc(Nl * Nl * sizeof(double));

    // MPI_Barrier(MPI_COMM_WORLD);
    double before_dot = MPI_Wtime();
    
    for (int shift = 0; shift < sqrt_size; shift++) {
		// Matrix multiplication

        #pragma omp parallel for num_threads(num_threads)
        for (int i = 0; i < Nl; i++) {
            if (i == 0 && rank == 0) {
                printf("thr: %d\n", omp_get_num_threads());
            }
            for (int k = 0; k < Nl; k++)
                for (int j = 0; j < Nl; j++)
                    C[i * Nl + j] += A[i * Nl + k] * B[k * Nl + j];
        }
                    
        if (shift == sqrt_size - 1) 
        	break;

		// Communication
        MPI_Sendrecv(A, Nl * Nl, MPI_DOUBLE, left, 1, buf, Nl * Nl, MPI_DOUBLE, right, 1, cannon_comm, &status);
        tmp = buf;
        buf = A;
        A = tmp;
        
        MPI_Sendrecv(B, Nl * Nl, MPI_DOUBLE, up, 2, buf, Nl * Nl, MPI_DOUBLE, down, 2, cannon_comm, &status);
        tmp = buf;
        buf = B;
        B = tmp;
    }
    
    // MPI_Barrier(MPI_COMM_WORLD);
    double before_write = MPI_Wtime();

    Matrix out_mat = mat1;

    out_mat.height_start = out_mat.block_size * proc_coords[0];
    out_mat.height_stop = out_mat.block_size * (proc_coords[0] + 1);

    out_mat.width_start = out_mat.block_size * proc_coords[1];
    out_mat.width_stop = out_mat.block_size * (proc_coords[1] + 1);

    out_mat.mat = C;
    write_mat(argv[3], out_mat, rank);

    free(A);
    free(B);
    free(C);
    free(buf);
    
    MPI_Finalize();
    double after_write = MPI_Wtime();

    if (rank == 0) {
        double read_time = before_dot - before_read;
        double dot_time = before_write - before_dot;
        double write_time = after_write - before_write;
    	printf("%lf\n", read_time * 1000);
    	printf("%lf\n", dot_time * 1000);
    	printf("%lf\n", write_time * 1000);
    }

    return 0;
}
