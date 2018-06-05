# distutils: extra_compile_args = -fopenmp
# distutils: extra_link_args = -fopenmp

"""
This module exists to parallelize calls to PhaseRecord.obj with OpenMP
"""

from libc.stdio cimport printf
cimport openmp
from cython cimport boundscheck, wraparound
from cython.parallel cimport prange

from pycalphad.core.phase_rec cimport PhaseRecord

@boundscheck(False)
@wraparound(False)
cpdef void parallel_obj(PhaseRecord prx, double[::1] out, double[:,::1] dof) nogil:
    """Wrapper around PhaseRecord.obj to compute the phase values in parallel"""
    # we define the total number of chunks to be one greater than what we will parallelize
    # this ensures that we can do a final pass and get any remaining rows left by integer division
    # the whole reason we are doing this, rather than letting prange sort it out,
    # is because prx.obj specifically expects that the inputs are vectorized
    # e.g. out must be 1d and dof must be 2d.
    cdef int nprocs = openmp.omp_get_num_procs()
    cdef int chunks = nprocs + 1
    cdef int chunksize = out.shape[0] // chunks
    cdef int final_chunk_idx = (chunks - 1)*chunksize
    cdef int chunk_idx
    cdef int i
    cdef int chunk_start_idx
    cdef int chunk_end_idx

    # TODO: num_threads should be num_procs, at least on a Mac, for optimal use.
    # check for other systems
    with nogil:
        for i in prange(chunks-1, num_threads=nprocs):
            chunk_start_idx = i*chunksize
            chunk_end_idx = (i+1)*chunksize
            prx.obj(out[chunk_start_idx:chunk_end_idx], dof[chunk_start_idx:chunk_end_idx, :])
        # go from the last parallel chunk through to the end
        prx.obj(out[final_chunk_idx:], dof[final_chunk_idx:, :])


@boundscheck(False)
@wraparound(False)
cpdef void check_threads() nogil:
    printf("num threads: %d\n", openmp.omp_get_num_threads())
    printf("max threads: %d\n", openmp.omp_get_max_threads())
    printf("num procs  : %d\n", openmp.omp_get_num_procs())
