# distutils: extra_compile_args = -fopenmp
# distutils: extra_link_args = -fopenmp

"""
This module exists to parallelize calls to PhaseRecord.obj with OpenMP
"""

cimport cython
from pycalphad.core.phase_rec cimport PhaseRecord
from cython.parallel cimport parallel, prange

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void calc_obj(PhaseRecord prx, double[::1] out, double[:,::1] dof) nogil:
    cdef int chunks = 2
    cdef int chunksize = out.shape[0] // chunks
    cdef int final_chunk_idx = (chunks - 1)*chunksize
    cdef int chunk_idx
    cdef int i

    for i in range(chunks-1):
        chunk_start_idx = i*chunksize
        chunk_end_idx = (i+1)*chunksize
        prx.obj(out[chunk_start_idx:chunk_end_idx], dof[chunk_start_idx:chunk_end_idx, :])
    prx.obj(out[chunk_end_idx:], dof[chunk_end_idx:, :])
