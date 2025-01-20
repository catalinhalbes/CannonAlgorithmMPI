from subprocess import Popen, PIPE
import numpy
import struct

BUILD_PATH = "./build/"

def prepare_out(out):
    return out.decode('utf-8').replace("\\n", "\n").replace("\\r", "\r")

def compile_c_mpi(filename):
    proc = Popen(["mpicc", "-O3", "-Wall", "-march=native", "-mtune=native", "-flto", "-fuse-linker-plugin", "-pthread", "-fopenmp", "-lm", "-o", BUILD_PATH + filename, filename + ".c"], stdout=PIPE)
    (out, err) = proc.communicate()
    if out:
        print("Compilation output:\n", prepare_out(out))
    if err:
        print("Compilation error:\n", prepare_out(err))
    return proc.wait()

def loadmat(filename: str) -> numpy.array:
    with open(filename, 'rb') as f:
        h, w = struct.unpack('<QQ', f.read(16))
        data = numpy.frombuffer(f.read(), dtype=numpy.float64).reshape(h, w)
        return data

def writemat(filename: str, mat: numpy.array) -> None:
    with open(filename, 'wb') as f:
        f.write(struct.pack("<QQ", mat.shape[0], mat.shape[1]))
        f.write(mat.tobytes())

def mat_MAPE(mat1_file, mat2_file) -> float:
    mat1 = loadmat(mat1_file)
    mat2 = loadmat(mat2_file)

    mask = mat1 != 0
    return (numpy.fabs(mat1 - mat2)/mat1)[mask].mean()

def mat_MAE(mat1_file, mat2_file):
    mat1 = loadmat(mat1_file)
    mat2 = loadmat(mat2_file)
    return numpy.sum(numpy.abs(mat1 - mat2)) / numpy.prod(mat1.shape)
