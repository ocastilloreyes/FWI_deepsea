/*
 * =============================================================================
 * Copyright (c) 2016-2018, Barcelona Supercomputing Center (BSC)
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the <organization> nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * =============================================================================
 */

#include "fwi/fwi_kernel.h"
//CAFE
#include <string.h>
//CAFE

#if defined(HAVE_EXTRAE)
#include "extrae.h"
#endif

/*
 * Initializes an array of length "length" to a random number.
 */
void set_array_to_random_real( real* restrict array, const integer length)
{
    const real randvalue = rand() / (1.0 * RAND_MAX);

    print_debug("Array is being initialized to %f", randvalue);

    set_array_to_constant(array, randvalue, length);
}

/*
 * Initializes an array of length "length" to a constant floating point value.
 */
void set_array_to_constant( real* restrict array, const real value, const integer length)
{
#if defined(_OPENACC)
    #pragma acc kernels copyin(array[0:length])
#endif
    for( integer i = 0; i < length; i++ )
        array[i] = value;
}

void check_memory_shot( const integer dimmz,
                        const integer dimmx,
                        const integer dimmy,
                        coeff_t *c,
                        s_t     *s,
                        v_t     *v,
                        real    *rho)
{
#if defined(DEBUG)
    print_debug("Checking memory shot values");

    real UNUSED(value);
    const integer size = dimmz * dimmx * dimmy;
    for( int i=0; i < size; i++)
    {
        value = c->c11[i];
        value = c->c12[i];
        value = c->c13[i];
        value = c->c14[i];
        value = c->c15[i];
        value = c->c16[i];

        value = c->c22[i];
        value = c->c23[i];
        value = c->c24[i];
        value = c->c25[i];
        value = c->c26[i];

        value = c->c33[i];
        value = c->c34[i];
        value = c->c35[i];
        value = c->c36[i];

        value = c->c44[i];
        value = c->c45[i];
        value = c->c46[i];

        value = c->c55[i];
        value = c->c56[i];
        value = c->c66[i];

        value = v->tl.u[i];
        value = v->tl.v[i];
        value = v->tl.w[i];

        value = v->tr.u[i];
        value = v->tr.v[i];
        value = v->tr.w[i];

        value = v->bl.u[i];
        value = v->bl.v[i];
        value = v->bl.w[i];

        value = v->br.u[i];
        value = v->br.v[i];
        value = v->br.w[i];

        value = rho[i];
    }
#endif /* end of pragma DEBUG */
};


void alloc_memory_shot( const integer dimmz,
                        const integer dimmx,
                        const integer dimmy,
                        coeff_t *c,
                        s_t     *s,
                        v_t     *v,
                        real    **rho)
{
    PUSH_RANGE

    const integer ncells = dimmz * dimmx * dimmy;
    const size_t size    = ncells * sizeof(real);

    print_debug("ptr size = " I " bytes ("I" elements)", 
            size, (size_t) ncells);

    /* allocate coefficients */
    c->c11 = (real*) __malloc( ALIGN_REAL, size);
    c->c12 = (real*) __malloc( ALIGN_REAL, size);
    c->c13 = (real*) __malloc( ALIGN_REAL, size);
    c->c14 = (real*) __malloc( ALIGN_REAL, size);
    c->c15 = (real*) __malloc( ALIGN_REAL, size);
    c->c16 = (real*) __malloc( ALIGN_REAL, size);

    c->c22 = (real*) __malloc( ALIGN_REAL, size);
    c->c23 = (real*) __malloc( ALIGN_REAL, size);
    c->c24 = (real*) __malloc( ALIGN_REAL, size);
    c->c25 = (real*) __malloc( ALIGN_REAL, size);
    c->c26 = (real*) __malloc( ALIGN_REAL, size);

    c->c33 = (real*) __malloc( ALIGN_REAL, size);
    c->c34 = (real*) __malloc( ALIGN_REAL, size);
    c->c35 = (real*) __malloc( ALIGN_REAL, size);
    c->c36 = (real*) __malloc( ALIGN_REAL, size);

    c->c44 = (real*) __malloc( ALIGN_REAL, size);
    c->c45 = (real*) __malloc( ALIGN_REAL, size);
    c->c46 = (real*) __malloc( ALIGN_REAL, size);

    c->c55 = (real*) __malloc( ALIGN_REAL, size);
    c->c56 = (real*) __malloc( ALIGN_REAL, size);
    c->c66 = (real*) __malloc( ALIGN_REAL, size);

    /* allocate velocity components */
    v->tl.u = (real*) __malloc( ALIGN_REAL, size);
    v->tl.v = (real*) __malloc( ALIGN_REAL, size);
    v->tl.w = (real*) __malloc( ALIGN_REAL, size);

    v->tr.u = (real*) __malloc( ALIGN_REAL, size);
    v->tr.v = (real*) __malloc( ALIGN_REAL, size);
    v->tr.w = (real*) __malloc( ALIGN_REAL, size);

    v->bl.u = (real*) __malloc( ALIGN_REAL, size);
    v->bl.v = (real*) __malloc( ALIGN_REAL, size);
    v->bl.w = (real*) __malloc( ALIGN_REAL, size);

    v->br.u = (real*) __malloc( ALIGN_REAL, size);
    v->br.v = (real*) __malloc( ALIGN_REAL, size);
    v->br.w = (real*) __malloc( ALIGN_REAL, size);

    /* allocate stress components   */
    s->tl.zz = (real*) __malloc( ALIGN_REAL, size);
    s->tl.xz = (real*) __malloc( ALIGN_REAL, size);
    s->tl.yz = (real*) __malloc( ALIGN_REAL, size);
    s->tl.xx = (real*) __malloc( ALIGN_REAL, size);
    s->tl.xy = (real*) __malloc( ALIGN_REAL, size);
    s->tl.yy = (real*) __malloc( ALIGN_REAL, size);

    s->tr.zz = (real*) __malloc( ALIGN_REAL, size);
    s->tr.xz = (real*) __malloc( ALIGN_REAL, size);
    s->tr.yz = (real*) __malloc( ALIGN_REAL, size);
    s->tr.xx = (real*) __malloc( ALIGN_REAL, size);
    s->tr.xy = (real*) __malloc( ALIGN_REAL, size);
    s->tr.yy = (real*) __malloc( ALIGN_REAL, size);

    s->bl.zz = (real*) __malloc( ALIGN_REAL, size);
    s->bl.xz = (real*) __malloc( ALIGN_REAL, size);
    s->bl.yz = (real*) __malloc( ALIGN_REAL, size);
    s->bl.xx = (real*) __malloc( ALIGN_REAL, size);
    s->bl.xy = (real*) __malloc( ALIGN_REAL, size);
    s->bl.yy = (real*) __malloc( ALIGN_REAL, size);

    s->br.zz = (real*) __malloc( ALIGN_REAL, size);
    s->br.xz = (real*) __malloc( ALIGN_REAL, size);
    s->br.yz = (real*) __malloc( ALIGN_REAL, size);
    s->br.xx = (real*) __malloc( ALIGN_REAL, size);
    s->br.xy = (real*) __malloc( ALIGN_REAL, size);
    s->br.yy = (real*) __malloc( ALIGN_REAL, size);

    /* allocate density array       */
    *rho = (real*) __malloc( ALIGN_REAL, size);

#if defined(_OPENACC)
    const real* rrho  = *rho;

    coeff_t cc = *c;
    #pragma acc enter data create(cc)
    #pragma acc enter data create(cc.c11[:ncells])
    #pragma acc enter data create(cc.c12[:ncells])
    #pragma acc enter data create(cc.c13[:ncells])
    #pragma acc enter data create(cc.c14[:ncells])
    #pragma acc enter data create(cc.c15[:ncells])
    #pragma acc enter data create(cc.c16[:ncells])
    #pragma acc enter data create(cc.c22[:ncells])
    #pragma acc enter data create(cc.c23[:ncells])
    #pragma acc enter data create(cc.c24[:ncells])
    #pragma acc enter data create(cc.c25[:ncells])
    #pragma acc enter data create(cc.c26[:ncells])
    #pragma acc enter data create(cc.c33[:ncells])
    #pragma acc enter data create(cc.c34[:ncells])
    #pragma acc enter data create(cc.c35[:ncells])
    #pragma acc enter data create(cc.c36[:ncells])
    #pragma acc enter data create(cc.c44[:ncells])
    #pragma acc enter data create(cc.c45[:ncells])
    #pragma acc enter data create(cc.c46[:ncells])
    #pragma acc enter data create(cc.c55[:ncells])
    #pragma acc enter data create(cc.c56[:ncells])
    #pragma acc enter data create(cc.c66[:ncells])

    v_t vv = *v;

    #pragma acc enter data copyin(vv)
    #pragma acc enter data create(vv.tl.u[:ncells])
    #pragma acc enter data create(vv.tl.v[:ncells])
    #pragma acc enter data create(vv.tl.w[:ncells])
    #pragma acc enter data create(vv.tr.u[:ncells])
    #pragma acc enter data create(vv.tr.v[:ncells])
    #pragma acc enter data create(vv.tr.w[:ncells])
    #pragma acc enter data create(vv.bl.u[:ncells])
    #pragma acc enter data create(vv.bl.v[:ncells])
    #pragma acc enter data create(vv.bl.w[:ncells])
    #pragma acc enter data create(vv.br.u[:ncells])
    #pragma acc enter data create(vv.br.v[:ncells])
    #pragma acc enter data create(vv.br.w[:ncells])

    s_t ss = *s;
    #pragma acc enter data copyin(ss)
    #pragma acc enter data create(ss.tl.zz[:ncells])
    #pragma acc enter data create(ss.tl.xz[:ncells])
    #pragma acc enter data create(ss.tl.yz[:ncells])
    #pragma acc enter data create(ss.tl.xx[:ncells])
    #pragma acc enter data create(ss.tl.xy[:ncells])
    #pragma acc enter data create(ss.tl.yy[:ncells])
    #pragma acc enter data create(ss.tr.zz[:ncells])
    #pragma acc enter data create(ss.tr.xz[:ncells])
    #pragma acc enter data create(ss.tr.yz[:ncells])
    #pragma acc enter data create(ss.tr.xx[:ncells])
    #pragma acc enter data create(ss.tr.xy[:ncells])
    #pragma acc enter data create(ss.tr.yy[:ncells])
    #pragma acc enter data create(ss.bl.zz[:ncells])
    #pragma acc enter data create(ss.bl.xz[:ncells])
    #pragma acc enter data create(ss.bl.yz[:ncells])
    #pragma acc enter data create(ss.bl.xx[:ncells])
    #pragma acc enter data create(ss.bl.xy[:ncells])
    #pragma acc enter data create(ss.bl.yy[:ncells])
    #pragma acc enter data create(ss.br.zz[:ncells])
    #pragma acc enter data create(ss.br.xz[:ncells])
    #pragma acc enter data create(ss.br.yz[:ncells])
    #pragma acc enter data create(ss.br.xx[:ncells])
    #pragma acc enter data create(ss.br.xy[:ncells])
    #pragma acc enter data create(ss.br.yy[:ncells])

    #pragma acc enter data create(rrho[:ncells])

#endif /* end of pragma _OPENACC */

    POP_RANGE
};

void free_memory_shot( coeff_t *c,
                       s_t     *s,
                       v_t     *v,
                       real    **rho)
{
    PUSH_RANGE

#if defined(_OPENACC)
    #pragma acc wait

    #pragma acc exit data delete(c->c11)
    #pragma acc exit data delete(c->c12)
    #pragma acc exit data delete(c->c13)
    #pragma acc exit data delete(c->c14)
    #pragma acc exit data delete(c->c15)
    #pragma acc exit data delete(c->c16)
    #pragma acc exit data delete(c->c22)
    #pragma acc exit data delete(c->c23)
    #pragma acc exit data delete(c->c24)
    #pragma acc exit data delete(c->c25)
    #pragma acc exit data delete(c->c26)
    #pragma acc exit data delete(c->c33)
    #pragma acc exit data delete(c->c34)
    #pragma acc exit data delete(c->c35)
    #pragma acc exit data delete(c->c36)
    #pragma acc exit data delete(c->c44)
    #pragma acc exit data delete(c->c45)
    #pragma acc exit data delete(c->c46)
    #pragma acc exit data delete(c->c55)
    #pragma acc exit data delete(c->c56)
    #pragma acc exit data delete(c->c66)
    #pragma acc exit data delete(c)

    #pragma acc exit data delete(v->tl.u)
    #pragma acc exit data delete(v->tl.v)
    #pragma acc exit data delete(v->tl.w)
    #pragma acc exit data delete(v->tr.u)
    #pragma acc exit data delete(v->tr.v)
    #pragma acc exit data delete(v->tr.w)
    #pragma acc exit data delete(v->bl.u)
    #pragma acc exit data delete(v->bl.v)
    #pragma acc exit data delete(v->bl.w)
    #pragma acc exit data delete(v->br.u)
    #pragma acc exit data delete(v->br.v)
    #pragma acc exit data delete(v->br.w)


    #pragma acc exit data delete(s->tl.zz)
    #pragma acc exit data delete(s->tl.xz)
    #pragma acc exit data delete(s->tl.yz)
    #pragma acc exit data delete(s->tl.xx)
    #pragma acc exit data delete(s->tl.xy)
    #pragma acc exit data delete(s->tl.yy)
    #pragma acc exit data delete(s->tr.zz)
    #pragma acc exit data delete(s->tr.xz)
    #pragma acc exit data delete(s->tr.yz)
    #pragma acc exit data delete(s->tr.xx)
    #pragma acc exit data delete(s->tr.xy)
    #pragma acc exit data delete(s->tr.yy)
    #pragma acc exit data delete(s->bl.zz)
    #pragma acc exit data delete(s->bl.xz)
    #pragma acc exit data delete(s->bl.yz)
    #pragma acc exit data delete(s->bl.xx)
    #pragma acc exit data delete(s->bl.xy)
    #pragma acc exit data delete(s->bl.yy)
    #pragma acc exit data delete(s->br.zz)
    #pragma acc exit data delete(s->br.xz)
    #pragma acc exit data delete(s->br.yz)
    #pragma acc exit data delete(s->br.xx)
    #pragma acc exit data delete(s->br.xy)
    #pragma acc exit data delete(s->br.yy)
    #pragma acc exit data delete(s)

    const real* rrho  = *rho;
    #pragma acc exit data delete(rrho)

#endif /* end pragma _OPENACC */

    /* deallocate coefficients */
    __free( (void*) c->c11 );
    __free( (void*) c->c12 );
    __free( (void*) c->c13 );
    __free( (void*) c->c14 );
    __free( (void*) c->c15 );
    __free( (void*) c->c16 );

    __free( (void*) c->c22 );
    __free( (void*) c->c23 );
    __free( (void*) c->c24 );
    __free( (void*) c->c25 );
    __free( (void*) c->c26 );
    __free( (void*) c->c33 );

    __free( (void*) c->c34 );
    __free( (void*) c->c35 );
    __free( (void*) c->c36 );

    __free( (void*) c->c44 );
    __free( (void*) c->c45 );
    __free( (void*) c->c46 );

    __free( (void*) c->c55 );
    __free( (void*) c->c56 );

    __free( (void*) c->c66 );

    /* deallocate velocity components */
    __free( (void*) v->tl.u );
    __free( (void*) v->tl.v );
    __free( (void*) v->tl.w );

    __free( (void*) v->tr.u );
    __free( (void*) v->tr.v );
    __free( (void*) v->tr.w );

    __free( (void*) v->bl.u );
    __free( (void*) v->bl.v );
    __free( (void*) v->bl.w );

    __free( (void*) v->br.u );
    __free( (void*) v->br.v );
    __free( (void*) v->br.w );

    /* deallocate stres components   */
    __free( (void*) s->tl.zz );
    __free( (void*) s->tl.xz );
    __free( (void*) s->tl.yz );
    __free( (void*) s->tl.xx );
    __free( (void*) s->tl.xy );
    __free( (void*) s->tl.yy );

    __free( (void*) s->tr.zz );
    __free( (void*) s->tr.xz );
    __free( (void*) s->tr.yz );
    __free( (void*) s->tr.xx );
    __free( (void*) s->tr.xy );
    __free( (void*) s->tr.yy );

    __free( (void*) s->bl.zz );
    __free( (void*) s->bl.xz );
    __free( (void*) s->bl.yz );
    __free( (void*) s->bl.xx );
    __free( (void*) s->bl.xy );
    __free( (void*) s->bl.yy );

    __free( (void*) s->br.zz );
    __free( (void*) s->br.xz );
    __free( (void*) s->br.yz );
    __free( (void*) s->br.xx );
    __free( (void*) s->br.xy );
    __free( (void*) s->br.yy );


    /* deallocate density array       */
    __free( (void*) *rho );

    POP_RANGE
};

/*
 * Loads initial values from coeffs, stress and velocity.
 *
 * dimmz: number of z planes
 * dimmx: number of x planes
 * FirstYPlane: first Y plane of my local domain (includes HALO)
 * LastYPlane: last Y plane of my local domain (includes HALO)
 */
void load_local_velocity_model ( const real    waveletFreq,
                                 const integer dimmz,
                                 const integer dimmx,
                                 const integer FirstYPlane,
                                 const integer LastYPlane,
                                 coeff_t *c,
                                 s_t     *s,
                                 v_t     *v,
                                 real    *rho)
{
    PUSH_RANGE

    const integer cellsInVolume = dimmz * dimmx * (LastYPlane - FirstYPlane);

    /*
     * Material, velocities and stresses are initizalized
     * accorting to the compilation flags, either randomly
     * or by reading an input velocity model.
     */

    /* initialize stress arrays */
    set_array_to_constant( s->tl.zz, 0, cellsInVolume);
    set_array_to_constant( s->tl.xz, 0, cellsInVolume);
    set_array_to_constant( s->tl.yz, 0, cellsInVolume);
    set_array_to_constant( s->tl.xx, 0, cellsInVolume);
    set_array_to_constant( s->tl.xy, 0, cellsInVolume);
    set_array_to_constant( s->tl.yy, 0, cellsInVolume);
    set_array_to_constant( s->tr.zz, 0, cellsInVolume);
    set_array_to_constant( s->tr.xz, 0, cellsInVolume);
    set_array_to_constant( s->tr.yz, 0, cellsInVolume);
    set_array_to_constant( s->tr.xx, 0, cellsInVolume);
    set_array_to_constant( s->tr.xy, 0, cellsInVolume);
    set_array_to_constant( s->tr.yy, 0, cellsInVolume);
    set_array_to_constant( s->bl.zz, 0, cellsInVolume);
    set_array_to_constant( s->bl.xz, 0, cellsInVolume);
    set_array_to_constant( s->bl.yz, 0, cellsInVolume);
    set_array_to_constant( s->bl.xx, 0, cellsInVolume);
    set_array_to_constant( s->bl.xy, 0, cellsInVolume);
    set_array_to_constant( s->bl.yy, 0, cellsInVolume);
    set_array_to_constant( s->br.zz, 0, cellsInVolume);
    set_array_to_constant( s->br.xz, 0, cellsInVolume);
    set_array_to_constant( s->br.yz, 0, cellsInVolume);
    set_array_to_constant( s->br.xx, 0, cellsInVolume);
    set_array_to_constant( s->br.xy, 0, cellsInVolume);
    set_array_to_constant( s->br.yy, 0, cellsInVolume);

#if defined(DO_NOT_PERFORM_IO)

    /* initialize material coefficients */
    set_array_to_random_real( c->c11, cellsInVolume);
    set_array_to_random_real( c->c12, cellsInVolume);
    set_array_to_random_real( c->c13, cellsInVolume);
    set_array_to_random_real( c->c14, cellsInVolume);
    set_array_to_random_real( c->c15, cellsInVolume);
    set_array_to_random_real( c->c16, cellsInVolume);
    set_array_to_random_real( c->c22, cellsInVolume);
    set_array_to_random_real( c->c23, cellsInVolume);
    set_array_to_random_real( c->c24, cellsInVolume);
    set_array_to_random_real( c->c25, cellsInVolume);
    set_array_to_random_real( c->c26, cellsInVolume);
    set_array_to_random_real( c->c33, cellsInVolume);
    set_array_to_random_real( c->c34, cellsInVolume);
    set_array_to_random_real( c->c35, cellsInVolume);
    set_array_to_random_real( c->c36, cellsInVolume);
    set_array_to_random_real( c->c44, cellsInVolume);
    set_array_to_random_real( c->c45, cellsInVolume);
    set_array_to_random_real( c->c46, cellsInVolume);
    set_array_to_random_real( c->c55, cellsInVolume);
    set_array_to_random_real( c->c56, cellsInVolume);
    set_array_to_random_real( c->c66, cellsInVolume);

    /* initalize velocity components */
    set_array_to_random_real( v->tl.u, cellsInVolume );
    set_array_to_random_real( v->tl.v, cellsInVolume );
    set_array_to_random_real( v->tl.w, cellsInVolume );
    set_array_to_random_real( v->tr.u, cellsInVolume );
    set_array_to_random_real( v->tr.v, cellsInVolume );
    set_array_to_random_real( v->tr.w, cellsInVolume );
    set_array_to_random_real( v->bl.u, cellsInVolume );
    set_array_to_random_real( v->bl.v, cellsInVolume );
    set_array_to_random_real( v->bl.w, cellsInVolume );
    set_array_to_random_real( v->br.u, cellsInVolume );
    set_array_to_random_real( v->br.v, cellsInVolume );
    set_array_to_random_real( v->br.w, cellsInVolume );

    /* initialize density (rho) */
    set_array_to_random_real( rho, cellsInVolume );

#else /* load velocity model from external file */

    /* initialize material coefficients */
    set_array_to_constant( c->c11, 1.0, cellsInVolume);
    set_array_to_constant( c->c12, 1.0, cellsInVolume);
    set_array_to_constant( c->c13, 1.0, cellsInVolume);
    set_array_to_constant( c->c14, 1.0, cellsInVolume);
    set_array_to_constant( c->c15, 1.0, cellsInVolume);
    set_array_to_constant( c->c16, 1.0, cellsInVolume);
    set_array_to_constant( c->c22, 1.0, cellsInVolume);
    set_array_to_constant( c->c23, 1.0, cellsInVolume);
    set_array_to_constant( c->c24, 1.0, cellsInVolume);
    set_array_to_constant( c->c25, 1.0, cellsInVolume);
    set_array_to_constant( c->c26, 1.0, cellsInVolume);
    set_array_to_constant( c->c33, 1.0, cellsInVolume);
    set_array_to_constant( c->c34, 1.0, cellsInVolume);
    set_array_to_constant( c->c35, 1.0, cellsInVolume);
    set_array_to_constant( c->c36, 1.0, cellsInVolume);
    set_array_to_constant( c->c44, 1.0, cellsInVolume);
    set_array_to_constant( c->c45, 1.0, cellsInVolume);
    set_array_to_constant( c->c46, 1.0, cellsInVolume);
    set_array_to_constant( c->c55, 1.0, cellsInVolume);
    set_array_to_constant( c->c56, 1.0, cellsInVolume);
    set_array_to_constant( c->c66, 1.0, cellsInVolume);

    /* initialize density (rho) */
    set_array_to_constant( rho, 1.0, cellsInVolume );

    /* local variables */
    double tstart_outer, tstart_inner;
    double tend_outer, tend_inner;
    double iospeed_inner, iospeed_outer;
    char modelname[300];

     /* open initial model, binary file */
    sprintf( modelname, "../data/inputmodels/velocitymodel_%.2f.bin", waveletFreq );
    print_info("Loading input model %s from disk (this could take a while)", modelname);

    /* start clock, take into account file opening */
    tstart_outer = dtime();
    FILE* model = safe_fopen( modelname, "rb", __FILE__, __LINE__ );

    /* start clock, do not take into account file opening */
    tstart_inner = dtime();

    /* seek to the correct position corresponding to mpi_rank */
    if (fseek ( model, sizeof(real) * WRITTEN_FIELDS * dimmz * dimmx * FirstYPlane, SEEK_SET) != 0)
        print_error("fseek() failed to set the correct position");

    /* initalize velocity components */
    safe_fread( v->tl.u, sizeof(real), cellsInVolume, model, __FILE__, __LINE__ );
    safe_fread( v->tl.v, sizeof(real), cellsInVolume, model, __FILE__, __LINE__ );
    safe_fread( v->tl.w, sizeof(real), cellsInVolume, model, __FILE__, __LINE__ );
    safe_fread( v->tr.u, sizeof(real), cellsInVolume, model, __FILE__, __LINE__ );
    safe_fread( v->tr.v, sizeof(real), cellsInVolume, model, __FILE__, __LINE__ );
    safe_fread( v->tr.w, sizeof(real), cellsInVolume, model, __FILE__, __LINE__ );
    safe_fread( v->bl.u, sizeof(real), cellsInVolume, model, __FILE__, __LINE__ );
    safe_fread( v->bl.v, sizeof(real), cellsInVolume, model, __FILE__, __LINE__ );
    safe_fread( v->bl.w, sizeof(real), cellsInVolume, model, __FILE__, __LINE__ );
    safe_fread( v->br.u, sizeof(real), cellsInVolume, model, __FILE__, __LINE__ );
    safe_fread( v->br.v, sizeof(real), cellsInVolume, model, __FILE__, __LINE__ );
    safe_fread( v->br.w, sizeof(real), cellsInVolume, model, __FILE__, __LINE__ );

    /* stop inner timer */
    tend_inner = dtime() - tstart_inner;

    /* stop timer and compute statistics */
    safe_fclose ( modelname, model, __FILE__, __LINE__ );
    tend_outer = dtime() - tstart_outer;

    const integer bytesForVolume = WRITTEN_FIELDS * cellsInVolume * sizeof(real);

    iospeed_inner = (bytesForVolume / (1000.f * 1000.f)) / tend_inner;
    iospeed_outer = (bytesForVolume / (1000.f * 1000.f)) / tend_outer;

    print_stats("Initial velocity model loaded (%lf GB)", TOGB(1.f * bytesForVolume));
    print_stats("\tInner time %lf seconds (%lf MiB/s)", tend_inner, iospeed_inner);
    print_stats("\tOuter time %lf seconds (%lf MiB/s)", tend_outer, iospeed_outer);
    print_stats("\tDifference %lf seconds", tend_outer - tend_inner);

#if defined(_OPENACC)
    const real* vtlu = v->tl.u;
    const real* vtlv = v->tl.v;
    const real* vtlw = v->tl.w;

    const real* vtru = v->tr.u;
    const real* vtrv = v->tr.v;
    const real* vtrw = v->tr.w;

    const real* vblu = v->bl.u;
    const real* vblv = v->bl.v;
    const real* vblw = v->bl.w;

    const real* vbru = v->br.u;
    const real* vbrv = v->br.v;
    const real* vbrw = v->br.w;

    #pragma acc update device(vtlu[0:cellsInVolume], vtlv[0:cellsInVolume], vtlw[0:cellsInVolume]) \
                       device(vtru[0:cellsInVolume], vtrv[0:cellsInVolume], vtrw[0:cellsInVolume]) \
                       device(vblu[0:cellsInVolume], vblv[0:cellsInVolume], vblw[0:cellsInVolume]) \
                       device(vbru[0:cellsInVolume], vbrv[0:cellsInVolume], vbrw[0:cellsInVolume]) \
                       async(H2D)
#endif /* end of pragma _OPENACC */
#endif /* end of pragma DDO_NOT_PERFORM_IO clause */

    POP_RANGE
};


/*
 * Saves the complete velocity field to disk.
 */
void write_snapshot(char *folder,
                    int suffix,
                    v_t *v,
                    const integer dimmz,
                    const integer dimmx,
                    const integer dimmy,
		    real **array_mallocs,
		    int   i_mem)
{
    PUSH_RANGE

#if defined(DO_NOT_PERFORM_IO)
    print_info("We are not writing the snapshot here cause IO is not enabled!");
#else

    int rank = 0;
#if defined(USE_MPI)
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

    const integer cellsInVolume  = dimmz * dimmx * dimmy;

#if defined(_OPENACC)
    #pragma acc update self(v->tr.u[0:cellsInVolume], v->tr.v[0:cellsInVolume], v->tr.w[0:cellsInVolume]) \
                       self(v->tl.u[0:cellsInVolume], v->tl.v[0:cellsInVolume], v->tl.w[0:cellsInVolume]) \
                       self(v->br.u[0:cellsInVolume], v->br.v[0:cellsInVolume], v->br.w[0:cellsInVolume]) \
                       self(v->bl.u[0:cellsInVolume], v->bl.v[0:cellsInVolume], v->bl.w[0:cellsInVolume])
#endif /* end pragma _OPENACC*/

    /* local variables */
    char fname[300];

    /* open snapshot file and write results */
    sprintf(fname,"%s/snapshot.%03d.%05d", folder, rank, suffix);

#if defined(LOG_IO_STATS)
    double tstart_outer = dtime();
#endif
    FILE *snapshot = safe_fopen(fname,"wb", __FILE__, __LINE__ );
#if defined(LOG_IO_STATS)
    double tstart_inner = dtime();
#endif

    //CAFE
    //printf("Just before array_mallocs-- in write_snapshot Fwd phase \n");
    int wf = 0;
    //printf(" i_mem = %d\n", i_mem);
    //printf(" array_mallocs = %X\n", array_mallocs);
    memcpy(*(array_mallocs+i_mem)+wf, v->tr.u, cellsInVolume); wf++;
    memcpy(*(array_mallocs+i_mem)+wf, v->tr.v, cellsInVolume); wf++;
    memcpy(*(array_mallocs+i_mem)+wf, v->tr.w, cellsInVolume); wf++;
    memcpy(*(array_mallocs+i_mem)+wf, v->tl.u, cellsInVolume); wf++;
    memcpy(*(array_mallocs+i_mem)+wf, v->tl.v, cellsInVolume); wf++;
    memcpy(*(array_mallocs+i_mem)+wf, v->tl.w, cellsInVolume); wf++;
    memcpy(*(array_mallocs+i_mem)+wf, v->br.u, cellsInVolume); wf++;
    memcpy(*(array_mallocs+i_mem)+wf, v->br.v, cellsInVolume); wf++;
    memcpy(*(array_mallocs+i_mem)+wf, v->br.w, cellsInVolume); wf++;
    memcpy(*(array_mallocs+i_mem)+wf, v->bl.u, cellsInVolume); wf++;
    memcpy(*(array_mallocs+i_mem)+wf, v->bl.v, cellsInVolume); wf++;
    memcpy(*(array_mallocs+i_mem)+wf, v->bl.w, cellsInVolume);
    //CAFE
/*
    safe_fwrite( v->tr.u, sizeof(real), cellsInVolume, snapshot, __FILE__, __LINE__ );
    safe_fwrite( v->tr.v, sizeof(real), cellsInVolume, snapshot, __FILE__, __LINE__ );
    safe_fwrite( v->tr.w, sizeof(real), cellsInVolume, snapshot, __FILE__, __LINE__ );

    safe_fwrite( v->tl.u, sizeof(real), cellsInVolume, snapshot, __FILE__, __LINE__ );
    safe_fwrite( v->tl.v, sizeof(real), cellsInVolume, snapshot, __FILE__, __LINE__ );
    safe_fwrite( v->tl.w, sizeof(real), cellsInVolume, snapshot, __FILE__, __LINE__ );

    safe_fwrite( v->br.u, sizeof(real), cellsInVolume, snapshot, __FILE__, __LINE__ );
    safe_fwrite( v->br.v, sizeof(real), cellsInVolume, snapshot, __FILE__, __LINE__ );
    safe_fwrite( v->br.w, sizeof(real), cellsInVolume, snapshot, __FILE__, __LINE__ );

    safe_fwrite( v->bl.u, sizeof(real), cellsInVolume, snapshot, __FILE__, __LINE__ );
    safe_fwrite( v->bl.v, sizeof(real), cellsInVolume, snapshot, __FILE__, __LINE__ );
    safe_fwrite( v->bl.w, sizeof(real), cellsInVolume, snapshot, __FILE__, __LINE__ );
    */

#if defined(LOG_IO_STATS)
    /* stop inner timer */
    double tend_inner = dtime();
#endif
    /* close file and stop outer timer */
    safe_fclose(fname, snapshot, __FILE__, __LINE__ );
#if defined(LOG_IO_STATS)
    double tend_outer = dtime();

    double iospeed_inner = (( (double) cellsInVolume * sizeof(real) * 12.f) / (1000.f * 1000.f)) / (tend_inner - tstart_inner);
    double iospeed_outer = (( (double) cellsInVolume * sizeof(real) * 12.f) / (1000.f * 1000.f)) / (tend_outer - tstart_outer);

    print_stats("Write snapshot (%lf GB)", TOGB(cellsInVolume * sizeof(real) * 12));
    print_stats("\tInner time %lf seconds (%lf MB/s)", (tend_inner - tstart_inner), iospeed_inner);
    print_stats("\tOuter time %lf seconds (%lf MB/s)", (tend_outer - tstart_outer), iospeed_outer);
    print_stats("\tDifference %lf seconds", tend_outer - tend_inner);
#endif /* end pragma LOG_IO_STATS */
#endif /* end pragma DO_NOT_PERFORM_IO */

    POP_RANGE
};

/*
 * Reads the complete velocity field from disk.
 */
void read_snapshot(char *folder,
                   int suffix,
                   v_t *v,
                   const integer dimmz,
                   const integer dimmx,
                   const integer dimmy,
		   real **array_mallocs,
		   int i_mem)
{
    PUSH_RANGE

#if defined(DO_NOT_PERFORM_IO)
    print_info("We are not reading the snapshot here cause IO is not enabled!");
#else
    /* local variables */
    char fname[300];

    int rank = 0;
#if defined(USE_MPI)
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
#endif

    /* open file and read snapshot */
    sprintf(fname,"%s/snapshot.%03d.%05d", folder, rank, suffix);

#if defined(LOG_IO_STATS)
    double tstart_outer = dtime();
#endif
    FILE *snapshot = safe_fopen(fname,"rb", __FILE__, __LINE__ );
#if defined(LOG_IO_STATS)
    double tstart_inner = dtime();
#endif

    const integer cellsInVolume  = dimmz * dimmx * dimmy;

    /*CAFE*/
    //printf("Just before array_mallocs-- in read_snapshot Bwd phase \n");
    int wf = 0;
    //printf("Just before memcpy in read_snapshot Bwd phase \n");
    //printf(" i_mem = %d\n", i_mem);
    //printf(" array_mallocs = %X\n", array_mallocs);
    memcpy(v->tr.u, *(array_mallocs+i_mem)+wf, cellsInVolume); wf++;
    memcpy(v->tr.v, *(array_mallocs+i_mem)+wf, cellsInVolume); wf++;
    memcpy(v->tr.w, *(array_mallocs+i_mem)+wf, cellsInVolume); wf++;
    memcpy(v->tl.u, *(array_mallocs+i_mem)+wf, cellsInVolume); wf++;
    memcpy(v->tl.v, *(array_mallocs+i_mem)+wf, cellsInVolume); wf++;
    memcpy(v->tl.w, *(array_mallocs+i_mem)+wf, cellsInVolume); wf++;
    memcpy(v->br.u, *(array_mallocs+i_mem)+wf, cellsInVolume); wf++;
    memcpy(v->br.v, *(array_mallocs+i_mem)+wf, cellsInVolume); wf++;
    memcpy(v->br.w, *(array_mallocs+i_mem)+wf, cellsInVolume); wf++;
    memcpy(v->bl.u, *(array_mallocs+i_mem)+wf, cellsInVolume); wf++;
    memcpy(v->bl.v, *(array_mallocs+i_mem)+wf, cellsInVolume); wf++;
    memcpy(v->bl.w, *(array_mallocs+i_mem)+wf, cellsInVolume);
    //printf("Just after memcpy in read_snapshot Bwd phase \n");
    //CAFE 
    /*
    safe_fread( v->tr.u, sizeof(real), cellsInVolume, snapshot, __FILE__, __LINE__ );
    safe_fread( v->tr.v, sizeof(real), cellsInVolume, snapshot, __FILE__, __LINE__ );
    safe_fread( v->tr.w, sizeof(real), cellsInVolume, snapshot, __FILE__, __LINE__ );

    safe_fread( v->tl.u, sizeof(real), cellsInVolume, snapshot, __FILE__, __LINE__ );
    safe_fread( v->tl.v, sizeof(real), cellsInVolume, snapshot, __FILE__, __LINE__ );
    safe_fread( v->tl.w, sizeof(real), cellsInVolume, snapshot, __FILE__, __LINE__ );

    safe_fread( v->br.u, sizeof(real), cellsInVolume, snapshot, __FILE__, __LINE__ );
    safe_fread( v->br.v, sizeof(real), cellsInVolume, snapshot, __FILE__, __LINE__ );
    safe_fread( v->br.w, sizeof(real), cellsInVolume, snapshot, __FILE__, __LINE__ );

    safe_fread( v->bl.u, sizeof(real), cellsInVolume, snapshot, __FILE__, __LINE__ );
    safe_fread( v->bl.v, sizeof(real), cellsInVolume, snapshot, __FILE__, __LINE__ );
    safe_fread( v->bl.w, sizeof(real), cellsInVolume, snapshot, __FILE__, __LINE__ );
    */

#if defined(LOG_IO_STATS)
    /* stop inner timer */
    double tend_inner = dtime() - tstart_inner;
#endif
    /* close file and stop outer timer */
    safe_fclose(fname, snapshot, __FILE__, __LINE__ );
#if defined(LOG_IO_STATS)
    double tend_outer = dtime() - tstart_outer;

    double iospeed_inner = ((cellsInVolume * sizeof(real) * 12.f) / (1000.f * 1000.f)) / tend_inner;
    double iospeed_outer = ((cellsInVolume * sizeof(real) * 12.f) / (1000.f * 1000.f)) / tend_outer;

    print_stats("Read snapshot (%lf GB)", TOGB(cellsInVolume * sizeof(real) * 12));
    print_stats("\tInner time %lf seconds (%lf MiB/s)", tend_inner, iospeed_inner);
    print_stats("\tOuter time %lf seconds (%lf MiB/s)", tend_outer, iospeed_outer);
    print_stats("\tDifference %lf seconds", tend_outer - tend_inner);
#endif

#if defined(_OPENACC)
    #pragma acc update device(v->tr.u[0:cellsInVolume], v->tr.v[0:cellsInVolume], v->tr.w[0:cellsInVolume]) \
                       device(v->tl.u[0:cellsInVolume], v->tl.v[0:cellsInVolume], v->tl.w[0:cellsInVolume]) \
                       device(v->br.u[0:cellsInVolume], v->br.v[0:cellsInVolume], v->br.w[0:cellsInVolume]) \
                       device(v->bl.u[0:cellsInVolume], v->bl.v[0:cellsInVolume], v->bl.w[0:cellsInVolume]) \
                       async(H2D)
#endif /* end pragma _OPENACC */
#endif /* end pragma DO_NOT_PERFORM_IO */

    POP_RANGE
};

void propagate_shot(time_d        direction,
                    v_t           v,
                    s_t           s,
                    coeff_t       coeffs,
                    real          *rho,
                    int           timesteps,
                    int           ntbwd,
                    real          dt,
                    real          dzi,
                    real          dxi,
                    real          dyi,
                    integer       nz0,
                    integer       nzf,
                    integer       nx0,
                    integer       nxf,
                    integer       ny0,
                    integer       nyf,
                    integer       stacki,
                    char          *folder,
                    real          *UNUSED(dataflush),
                    integer       dimmz,
                    integer       dimmx,
                    integer       dimmy,
		    real **array_mallocs)
{
    PUSH_RANGE

    double tglobal_start, tglobal_total = 0.0;
    double tstress_start, tstress_total = 0.0;
    double tvel_start, tvel_total = 0.0;

    // CAFE
    static int i_mem = 0;
    // CAFE

    for(int t=0; t < timesteps; t++)
    {
        PUSH_RANGE

        if( t % 10 == 0 ) print_info("Computing %d-th timestep", t);

        /* perform IO */
        if ( t%stacki == 0 && direction == BACKWARD)
	{	
                // CAFE
		i_mem--;
                // CAFE
		read_snapshot(folder, ntbwd-t, &v, dimmz, dimmx, dimmy, array_mallocs, i_mem);
	}

        tglobal_start = dtime();

        /* wait read_snapshot H2D copies */
#if defined(_OPENACC)
        #pragma acc wait(H2D) if ( (t%stacki == 0 && direction == BACKWARD) || t==0 )
#endif

        /* ------------------------------------------------------------------------------ */
        /*                      VELOCITY COMPUTATION                                      */
        /* ------------------------------------------------------------------------------ */

        /* Phase 1. Computation of the left-most planes of the domain */
        velocity_propagator(v, s, coeffs, rho, dt, dzi, dxi, dyi,
                            nz0 +   HALO,
                            nzf -   HALO,
                            nx0 +   HALO,
                            nxf -   HALO,
                            ny0 +   HALO,
                            ny0 + 2*HALO,
                            dimmz, dimmx,
                            ONE_L);

        /* Phase 1. Computation of the right-most planes of the domain */
        velocity_propagator(v, s, coeffs, rho, dt, dzi, dxi, dyi,
                            nz0 +   HALO,
                            nzf -   HALO,
                            nx0 +   HALO,
                            nxf -   HALO,
                            nyf - 2*HALO,
                            nyf -   HALO,
                            dimmz, dimmx,
                            ONE_R);

#if defined(USE_MPI)
#if defined(HAVE_EXTRAE)
        Extrae_event (1234, 1);
#endif
        /* Boundary exchange for velocity values */
#if defined(HAVE_EXTRAE)
        exchange_velocity_boundaries( v, dimmz * dimmx, nyf, ny0);
        Extrae_event (1234, 0);
#endif
#endif

        /* Phase 2. Computation of the central planes. */
        tvel_start = dtime();

        velocity_propagator(v, s, coeffs, rho, dt, dzi, dxi, dyi,
                            nz0 +   HALO,
                            nzf -   HALO,
                            nx0 +   HALO,
                            nxf -   HALO,
                            ny0 + 2*HALO,
                            nyf - 2*HALO,
                            dimmz, dimmx,
                            TWO);

#if defined(_OPENACC)
        #pragma acc wait(ONE_L, ONE_R, TWO)
#endif
        tvel_total += (dtime() - tvel_start);

        /* ------------------------------------------------------------------------------ */
        /*                        STRESS COMPUTATION                                      */
        /* ------------------------------------------------------------------------------ */

        /* Phase 1. Computation of the left-most planes of the domain */
        stress_propagator(s, v, coeffs, rho, dt, dzi, dxi, dyi,
                          nz0 +   HALO,
                          nzf -   HALO,
                          nx0 +   HALO,
                          nxf -   HALO,
                          ny0 +   HALO,
                          ny0 + 2*HALO,
                          dimmz, dimmx,
                          ONE_L);

        /* Phase 1. Computation of the right-most planes of the domain */
        stress_propagator(s, v, coeffs, rho, dt, dzi, dxi, dyi,
                          nz0 +   HALO,
                          nzf -   HALO,
                          nx0 +   HALO,
                          nxf -   HALO,
                          nyf - 2*HALO,
                          nyf -   HALO,
                          dimmz, dimmx,
                          ONE_R);

#if defined(USE_MPI)
#if defined(HAVE_EXTRAE)
        Extrae_event (1234, 1);
#endif
        /* Boundary exchange for stress values */
        exchange_stress_boundaries( s, dimmz * dimmx, nyf, ny0);
#if defined(HAVE_EXTRAE)
        Extrae_event (1234, 0);
#endif
#endif

        /* Phase 2 computation. Central planes of the domain */
        tstress_start = dtime();

        stress_propagator(s, v, coeffs, rho, dt, dzi, dxi, dyi,
                          nz0 +   HALO,
                          nzf -   HALO,
                          nx0 +   HALO,
                          nxf -   HALO,
                          ny0 + 2*HALO,
                          nyf - 2*HALO,
                          dimmz, dimmx,
                          TWO);

#if defined(HAVE_EXTRAE)
        Extrae_event (1234, 1);
#endif
#if defined(_OPENACC)
        #pragma acc wait(ONE_L, ONE_R, TWO, H2D, D2H)
#endif
        tstress_total += (dtime() - tstress_start);

        tglobal_total += (dtime() - tglobal_start);

        /* perform IO */
        if ( t%stacki == 0 && direction == FORWARD) 
	{
		write_snapshot(folder, ntbwd-t, &v, dimmz, dimmx, dimmy, array_mallocs, i_mem);
                // CAFE
		i_mem++;
                // CAFE
	}

#if defined(USE_MPI)
        MPI_Barrier( MPI_COMM_WORLD );
#endif
#if defined(HAVE_EXTRAE)
        Extrae_event (1234, 0);
#endif
        POP_RANGE
    }

    /* compute some statistics */
    double megacells = ((nzf - nz0) * (nxf - nx0) * (nyf - ny0)) / 1e6;
    tglobal_total /= (double) timesteps;
    tstress_total /= (double) timesteps;
    tvel_total    /= (double) timesteps;

    print_stats("Maingrid GLOBAL   computation took %lf seconds - %lf Mcells/s", tglobal_total, (2*megacells) / tglobal_total);
    print_stats("Maingrid STRESS   computation took %lf seconds - %lf Mcells/s", tstress_total,  megacells / tstress_total);
    print_stats("Maingrid VELOCITY computation took %lf seconds - %lf Mcells/s", tvel_total, megacells / tvel_total);

    POP_RANGE
};

#if defined(USE_MPI)
/*
NAME:exchange_boundaries
PURPOSE: data exchanges between the boundary layers of the analyzed volume

v                   (in) struct containing velocity arrays (4 points / cell x 3 components / point = 12 arrays)
plane_size          (in) Number of elements per plane to exchange
nyf                 (in) final plane to be exchanged
ny0                 (in) intial plane to be exchanged

RETURN none
*/
void exchange_velocity_boundaries ( v_t v,
                                    const integer plane_size,
                                    const integer nyf,
                                    const integer ny0 )
{
    PUSH_RANGE

    int     rank;          // mpi local rank
    int     nranks;        // num mpi ranks

    /* Initialize local variables */
    MPI_Comm_rank ( MPI_COMM_WORLD, &rank );
    MPI_Comm_size ( MPI_COMM_WORLD, &nranks );

    const integer num_planes = HALO;
    const integer nelems     = num_planes * plane_size;

    const integer left_recv  = ny0;
    const integer left_send  = ny0+HALO;

    const integer right_recv = nyf-HALO;
    const integer right_send = nyf-2*HALO;

    MPI_Status  statuses[48];
    MPI_Request requests[48];
    integer      wait_total = 0;
    MPI_Request*  wait_reqs = &requests[24];
    MPI_Status* wait_status = &statuses[24];

    real* ptr00 = &v.tl.u[left_send]; real* ptr01 = &v.tl.u[left_recv];
    real* ptr02 = &v.tl.v[left_send]; real* ptr03 = &v.tl.v[left_recv];
    real* ptr04 = &v.tl.w[left_send]; real* ptr05 = &v.tl.w[left_recv];
    real* ptr06 = &v.tr.u[left_send]; real* ptr07 = &v.tr.u[left_recv];
    real* ptr08 = &v.tr.v[left_send]; real* ptr09 = &v.tr.v[left_recv];
    real* ptr10 = &v.tr.w[left_send]; real* ptr11 = &v.tr.w[left_recv];
    real* ptr12 = &v.bl.u[left_send]; real* ptr13 = &v.bl.u[left_recv];
    real* ptr14 = &v.bl.v[left_send]; real* ptr15 = &v.bl.v[left_recv];
    real* ptr16 = &v.bl.w[left_send]; real* ptr17 = &v.bl.w[left_recv];
    real* ptr18 = &v.br.u[left_send]; real* ptr19 = &v.br.u[left_recv];
    real* ptr20 = &v.br.v[left_send]; real* ptr21 = &v.br.v[left_recv];
    real* ptr22 = &v.br.w[left_send]; real* ptr23 = &v.br.w[left_recv];

    real* ptr24 = &v.tl.u[right_send]; real* ptr25 = &v.tl.u[right_recv];
    real* ptr26 = &v.tl.v[right_send]; real* ptr27 = &v.tl.v[right_recv];
    real* ptr28 = &v.tl.w[right_send]; real* ptr29 = &v.tl.w[right_recv];
    real* ptr30 = &v.tr.u[right_send]; real* ptr31 = &v.tr.u[right_recv];
    real* ptr32 = &v.tr.v[right_send]; real* ptr33 = &v.tr.v[right_recv];
    real* ptr34 = &v.tr.w[right_send]; real* ptr35 = &v.tr.w[right_recv];
    real* ptr36 = &v.bl.u[right_send]; real* ptr37 = &v.bl.u[right_recv];
    real* ptr38 = &v.bl.v[right_send]; real* ptr39 = &v.bl.v[right_recv];
    real* ptr40 = &v.bl.w[right_send]; real* ptr41 = &v.bl.w[right_recv];
    real* ptr42 = &v.br.u[right_send]; real* ptr43 = &v.br.u[right_recv];
    real* ptr44 = &v.br.v[right_send]; real* ptr45 = &v.br.v[right_recv];
    real* ptr46 = &v.br.w[right_send]; real* ptr47 = &v.br.w[right_recv];

    #if defined(_OPENACC)
    #pragma acc host_data use_device(ptr00,ptr01,ptr02,ptr03,ptr04,ptr05,ptr06,ptr07,ptr08,ptr09,\
                                     ptr10,ptr11,ptr12,ptr13,ptr14,ptr15,ptr16,ptr17,ptr18,ptr19,\
                                     ptr20,ptr21,ptr22,ptr23,ptr24,ptr25,ptr26,ptr27,ptr28,ptr29,\
                                     ptr30,ptr31,ptr32,ptr33,ptr34,ptr35,ptr36,ptr37,ptr38,ptr39,\
                                     ptr40,ptr41,ptr42,ptr43,ptr44,ptr45,ptr46,ptr47)
    #pragma acc update self(ptr00[0:nelems],ptr01[0:nelems],ptr02[0:nelems],ptr03[0:nelems],ptr04[0:nelems],ptr05[0:nelems],ptr06[0:nelems],ptr07[0:nelems],ptr08[0:nelems],ptr09[0:nelems],\
                            ptr10[0:nelems],ptr11[0:nelems],ptr12[0:nelems],ptr13[0:nelems],ptr14[0:nelems],ptr15[0:nelems],ptr16[0:nelems],ptr17[0:nelems],ptr18[0:nelems],ptr19[0:nelems],\
                            ptr20[0:nelems],ptr21[0:nelems],ptr22[0:nelems],ptr23[0:nelems],ptr24[0:nelems],ptr25[0:nelems],ptr26[0:nelems],ptr27[0:nelems],ptr28[0:nelems],ptr29[0:nelems],\
                            ptr30[0:nelems],ptr31[0:nelems],ptr32[0:nelems],ptr33[0:nelems],ptr34[0:nelems],ptr35[0:nelems],ptr36[0:nelems],ptr37[0:nelems],ptr38[0:nelems],ptr39[0:nelems],\
                            ptr40[0:nelems],ptr41[0:nelems],ptr42[0:nelems],ptr43[0:nelems],ptr44[0:nelems],ptr45[0:nelems],ptr46[0:nelems],ptr47[0:nelems])
    #endif
    {
        if ( rank != 0 )
        {
            // [RANK-1] <---> [RANK] communication
            EXCHANGE( &v.tl.u[left_send], &v.tl.u[left_recv], rank-1, rank, nelems, &requests[ 0] );
            EXCHANGE( &v.tl.v[left_send], &v.tl.v[left_recv], rank-1, rank, nelems, &requests[ 2] );
            EXCHANGE( &v.tl.w[left_send], &v.tl.w[left_recv], rank-1, rank, nelems, &requests[ 4] );

            EXCHANGE( &v.tr.u[left_send], &v.tr.u[left_recv], rank-1, rank, nelems, &requests[ 6] );
            EXCHANGE( &v.tr.v[left_send], &v.tr.v[left_recv], rank-1, rank, nelems, &requests[ 8] );
            EXCHANGE( &v.tr.w[left_send], &v.tr.w[left_recv], rank-1, rank, nelems, &requests[10] );

            EXCHANGE( &v.bl.u[left_send], &v.bl.u[left_recv], rank-1, rank, nelems, &requests[12] );
            EXCHANGE( &v.bl.v[left_send], &v.bl.v[left_recv], rank-1, rank, nelems, &requests[14] );
            EXCHANGE( &v.bl.w[left_send], &v.bl.w[left_recv], rank-1, rank, nelems, &requests[16] );

            EXCHANGE( &v.br.u[left_send], &v.br.u[left_recv], rank-1, rank, nelems, &requests[18] );
            EXCHANGE( &v.br.v[left_send], &v.br.v[left_recv], rank-1, rank, nelems, &requests[20] );
            EXCHANGE( &v.br.w[left_send], &v.br.w[left_recv], rank-1, rank, nelems, &requests[22] );

            wait_total += 24;
            wait_reqs   = requests;
            wait_status = statuses;
        }

        if ( rank != nranks -1 )  //task to exchange stress boundaries
        {
            //                [RANK] <---> [RANK+1] communication
            EXCHANGE( &v.tl.u[right_send], &v.tl.u[right_recv], rank+1, rank, nelems, &requests[24] );
            EXCHANGE( &v.tl.v[right_send], &v.tl.v[right_recv], rank+1, rank, nelems, &requests[26] );
            EXCHANGE( &v.tl.w[right_send], &v.tl.w[right_recv], rank+1, rank, nelems, &requests[28] );

            EXCHANGE( &v.tr.u[right_send], &v.tr.u[right_recv], rank+1, rank, nelems, &requests[30] );
            EXCHANGE( &v.tr.v[right_send], &v.tr.v[right_recv], rank+1, rank, nelems, &requests[32] );
            EXCHANGE( &v.tr.w[right_send], &v.tr.w[right_recv], rank+1, rank, nelems, &requests[34] );

            EXCHANGE( &v.bl.u[right_send], &v.bl.u[right_recv], rank+1, rank, nelems, &requests[36] );
            EXCHANGE( &v.bl.v[right_send], &v.bl.v[right_recv], rank+1, rank, nelems, &requests[38] );
            EXCHANGE( &v.bl.w[right_send], &v.bl.w[right_recv], rank+1, rank, nelems, &requests[40] );

            EXCHANGE( &v.br.u[right_send], &v.br.u[right_recv], rank+1, rank, nelems, &requests[42] );
            EXCHANGE( &v.br.v[right_send], &v.br.v[right_recv], rank+1, rank, nelems, &requests[44] );
            EXCHANGE( &v.br.w[right_send], &v.br.w[right_recv], rank+1, rank, nelems, &requests[46] );

            wait_total += 24;
        }

        MPI_Waitall(wait_total, wait_reqs, wait_status );
    }

    POP_RANGE
};

/*
NAME:exchange_stress_boundaries
PURPOSE: data exchanges between the boundary layers of the analyzed volume

s                   (in) struct containing stress arrays (4 points / cell x 6 components / point = 24 arrays)
plane_size          (in) Number of elements per plane to exchange
rank                (in) rank id (CPU id)
nranks              (in) number of CPUs
nyf                 (in) final plane to be exchanged
ny0                 (in) intial plane to be exchanged

RETURN none
*/
void exchange_stress_boundaries ( s_t s,
                                  const integer plane_size,
                                  const integer nyf,
                                  const integer ny0 )
{
    PUSH_RANGE

    int     rank;          // mpi local rank
    int     nranks;        // num mpi ranks

    /* Initialize local variables */
    MPI_Comm_rank ( MPI_COMM_WORLD, &rank );
    MPI_Comm_size ( MPI_COMM_WORLD, &nranks );

    const integer num_planes = HALO;
    const integer nelems     = num_planes * plane_size;

    const integer left_recv  = ny0;
    const integer left_send  = ny0+HALO;

    const integer right_recv = nyf-HALO;
    const integer right_send = nyf-2*HALO;

    MPI_Status  statuses[96];
    MPI_Request requests[96];
    integer      wait_total = 0;
    MPI_Request*  wait_reqs = &requests[48];
    MPI_Status* wait_status = &statuses[48];

    real* ptr00 = &s.tl.zz[left_send]; real* ptr01 = &s.tl.zz[left_recv];
    real* ptr02 = &s.tl.xz[left_send]; real* ptr03 = &s.tl.xz[left_recv];
    real* ptr04 = &s.tl.yz[left_send]; real* ptr05 = &s.tl.yz[left_recv];
    real* ptr06 = &s.tl.xx[left_send]; real* ptr07 = &s.tl.xx[left_recv];
    real* ptr08 = &s.tl.xy[left_send]; real* ptr09 = &s.tl.xy[left_recv];
    real* ptr10 = &s.tl.yy[left_send]; real* ptr11 = &s.tl.yy[left_recv];

    real* ptr12 = &s.tr.zz[left_send]; real* ptr13 = &s.tr.zz[left_recv];
    real* ptr14 = &s.tr.xz[left_send]; real* ptr15 = &s.tr.xz[left_recv];
    real* ptr16 = &s.tr.yz[left_send]; real* ptr17 = &s.tr.yz[left_recv];
    real* ptr18 = &s.tr.xx[left_send]; real* ptr19 = &s.tr.xx[left_recv];
    real* ptr20 = &s.tr.xy[left_send]; real* ptr21 = &s.tr.xy[left_recv];
    real* ptr22 = &s.tr.yy[left_send]; real* ptr23 = &s.tr.yy[left_recv];

    real* ptr24 = &s.bl.zz[left_send]; real* ptr25 = &s.bl.zz[left_recv];
    real* ptr26 = &s.bl.xz[left_send]; real* ptr27 = &s.bl.xz[left_recv];
    real* ptr28 = &s.bl.yz[left_send]; real* ptr29 = &s.bl.yz[left_recv];
    real* ptr30 = &s.bl.xx[left_send]; real* ptr31 = &s.bl.xx[left_recv];
    real* ptr32 = &s.bl.xy[left_send]; real* ptr33 = &s.bl.xy[left_recv];
    real* ptr34 = &s.bl.yy[left_send]; real* ptr35 = &s.bl.yy[left_recv];

    real* ptr36 = &s.br.zz[left_send]; real* ptr37 = &s.br.zz[left_recv];
    real* ptr38 = &s.br.xz[left_send]; real* ptr39 = &s.br.xz[left_recv];
    real* ptr40 = &s.br.yz[left_send]; real* ptr41 = &s.br.yz[left_recv];
    real* ptr42 = &s.br.xx[left_send]; real* ptr43 = &s.br.xx[left_recv];
    real* ptr44 = &s.br.xy[left_send]; real* ptr45 = &s.br.xy[left_recv];
    real* ptr46 = &s.br.yy[left_send]; real* ptr47 = &s.br.yy[left_recv];

    #if defined(_OPENACC)
    #pragma acc host_data use_device(ptr00,ptr01,ptr02,ptr03,ptr04,ptr05,ptr06,ptr07,ptr08,ptr09,\
                                     ptr10,ptr11,ptr12,ptr13,ptr14,ptr15,ptr16,ptr17,ptr18,ptr19,\
                                     ptr20,ptr21,ptr22,ptr23,ptr24,ptr25,ptr26,ptr27,ptr28,ptr29,\
                                     ptr30,ptr31,ptr32,ptr33,ptr34,ptr35,ptr36,ptr37,ptr38,ptr39,\
                                     ptr40,ptr41,ptr42,ptr43,ptr44,ptr45,ptr46,ptr47)
    #pragma acc update self(ptr00[0:nelems],ptr01[0:nelems],ptr02[0:nelems],ptr03[0:nelems],ptr04[0:nelems],ptr05[0:nelems],ptr06[0:nelems],ptr07[0:nelems],ptr08[0:nelems],ptr09[0:nelems],\
                            ptr10[0:nelems],ptr11[0:nelems],ptr12[0:nelems],ptr13[0:nelems],ptr14[0:nelems],ptr15[0:nelems],ptr16[0:nelems],ptr17[0:nelems],ptr18[0:nelems],ptr19[0:nelems],\
                            ptr20[0:nelems],ptr21[0:nelems],ptr22[0:nelems],ptr23[0:nelems],ptr24[0:nelems],ptr25[0:nelems],ptr26[0:nelems],ptr27[0:nelems],ptr28[0:nelems],ptr29[0:nelems],\
                            ptr30[0:nelems],ptr31[0:nelems],ptr32[0:nelems],ptr33[0:nelems],ptr34[0:nelems],ptr35[0:nelems],ptr36[0:nelems],ptr37[0:nelems],ptr38[0:nelems],ptr39[0:nelems],\
                            ptr40[0:nelems],ptr41[0:nelems],ptr42[0:nelems],ptr43[0:nelems],ptr44[0:nelems],ptr45[0:nelems],ptr46[0:nelems],ptr47[0:nelems])
    #endif
    {
        if ( rank != 0 )
        {
            // [RANK-1] <---> [RANK] communication
            EXCHANGE( &s.tl.zz[left_send], &s.tl.zz[left_recv], rank-1, rank, nelems, &requests[ 0] );
            EXCHANGE( &s.tl.xz[left_send], &s.tl.xz[left_recv], rank-1, rank, nelems, &requests[ 2] );
            EXCHANGE( &s.tl.yz[left_send], &s.tl.yz[left_recv], rank-1, rank, nelems, &requests[ 4] );
            EXCHANGE( &s.tl.xx[left_send], &s.tl.xx[left_recv], rank-1, rank, nelems, &requests[ 6] );
            EXCHANGE( &s.tl.xy[left_send], &s.tl.xy[left_recv], rank-1, rank, nelems, &requests[ 8] );
            EXCHANGE( &s.tl.yy[left_send], &s.tl.yy[left_recv], rank-1, rank, nelems, &requests[10] );

            EXCHANGE( &s.tr.zz[left_send], &s.tr.zz[left_recv], rank-1, rank, nelems, &requests[12] );
            EXCHANGE( &s.tr.xz[left_send], &s.tr.xz[left_recv], rank-1, rank, nelems, &requests[14] );
            EXCHANGE( &s.tr.yz[left_send], &s.tr.yz[left_recv], rank-1, rank, nelems, &requests[16] );
            EXCHANGE( &s.tr.xx[left_send], &s.tr.xx[left_recv], rank-1, rank, nelems, &requests[18] );
            EXCHANGE( &s.tr.xy[left_send], &s.tr.xy[left_recv], rank-1, rank, nelems, &requests[20] );
            EXCHANGE( &s.tr.yy[left_send], &s.tr.yy[left_recv], rank-1, rank, nelems, &requests[22] );

            EXCHANGE( &s.bl.zz[left_send], &s.bl.zz[left_recv], rank-1, rank, nelems, &requests[24] );
            EXCHANGE( &s.bl.xz[left_send], &s.bl.xz[left_recv], rank-1, rank, nelems, &requests[26] );
            EXCHANGE( &s.bl.yz[left_send], &s.bl.yz[left_recv], rank-1, rank, nelems, &requests[28] );
            EXCHANGE( &s.bl.xx[left_send], &s.bl.xx[left_recv], rank-1, rank, nelems, &requests[30] );
            EXCHANGE( &s.bl.xy[left_send], &s.bl.xy[left_recv], rank-1, rank, nelems, &requests[32] );
            EXCHANGE( &s.bl.yy[left_send], &s.bl.yy[left_recv], rank-1, rank, nelems, &requests[34] );

            EXCHANGE( &s.br.zz[left_send], &s.br.zz[left_recv], rank-1, rank, nelems, &requests[36] );
            EXCHANGE( &s.br.xz[left_send], &s.br.xz[left_recv], rank-1, rank, nelems, &requests[38] );
            EXCHANGE( &s.br.yz[left_send], &s.br.yz[left_recv], rank-1, rank, nelems, &requests[40] );
            EXCHANGE( &s.br.xx[left_send], &s.br.xx[left_recv], rank-1, rank, nelems, &requests[42] );
            EXCHANGE( &s.br.xy[left_send], &s.br.xy[left_recv], rank-1, rank, nelems, &requests[44] );
            EXCHANGE( &s.br.yy[left_send], &s.br.yy[left_recv], rank-1, rank, nelems, &requests[46] );

            wait_total += 48;
            wait_reqs   = requests;
            wait_status = statuses;

        }

        if ( rank != nranks-1 )
        {
            //                [RANK] <---> [RANK+1] communication
            EXCHANGE( &s.tl.zz[right_send], &s.tl.zz[right_recv], rank+1, rank, nelems, &requests[48] );
            EXCHANGE( &s.tl.xz[right_send], &s.tl.xz[right_recv], rank+1, rank, nelems, &requests[50] );
            EXCHANGE( &s.tl.yz[right_send], &s.tl.yz[right_recv], rank+1, rank, nelems, &requests[52] );
            EXCHANGE( &s.tl.xx[right_send], &s.tl.xx[right_recv], rank+1, rank, nelems, &requests[54] );
            EXCHANGE( &s.tl.xy[right_send], &s.tl.xy[right_recv], rank+1, rank, nelems, &requests[56] );
            EXCHANGE( &s.tl.yy[right_send], &s.tl.yy[right_recv], rank+1, rank, nelems, &requests[58] );
                                                                                                     
            EXCHANGE( &s.tr.zz[right_send], &s.tr.zz[right_recv], rank+1, rank, nelems, &requests[60] );
            EXCHANGE( &s.tr.xz[right_send], &s.tr.xz[right_recv], rank+1, rank, nelems, &requests[62] );
            EXCHANGE( &s.tr.yz[right_send], &s.tr.yz[right_recv], rank+1, rank, nelems, &requests[64] );
            EXCHANGE( &s.tr.xx[right_send], &s.tr.xx[right_recv], rank+1, rank, nelems, &requests[66] );
            EXCHANGE( &s.tr.xy[right_send], &s.tr.xy[right_recv], rank+1, rank, nelems, &requests[68] );
            EXCHANGE( &s.tr.yy[right_send], &s.tr.yy[right_recv], rank+1, rank, nelems, &requests[70] );
                                                                                                     
            EXCHANGE( &s.bl.zz[right_send], &s.bl.zz[right_recv], rank+1, rank, nelems, &requests[72] );
            EXCHANGE( &s.bl.xz[right_send], &s.bl.xz[right_recv], rank+1, rank, nelems, &requests[74] );
            EXCHANGE( &s.bl.yz[right_send], &s.bl.yz[right_recv], rank+1, rank, nelems, &requests[76] );
            EXCHANGE( &s.bl.xx[right_send], &s.bl.xx[right_recv], rank+1, rank, nelems, &requests[78] );
            EXCHANGE( &s.bl.xy[right_send], &s.bl.xy[right_recv], rank+1, rank, nelems, &requests[80] );
            EXCHANGE( &s.bl.yy[right_send], &s.bl.yy[right_recv], rank+1, rank, nelems, &requests[82] );
                                                                                                     
            EXCHANGE( &s.br.zz[right_send], &s.br.zz[right_recv], rank+1, rank, nelems, &requests[84] );
            EXCHANGE( &s.br.xz[right_send], &s.br.xz[right_recv], rank+1, rank, nelems, &requests[86] );
            EXCHANGE( &s.br.yz[right_send], &s.br.yz[right_recv], rank+1, rank, nelems, &requests[88] );
            EXCHANGE( &s.br.xx[right_send], &s.br.xx[right_recv], rank+1, rank, nelems, &requests[90] );
            EXCHANGE( &s.br.xy[right_send], &s.br.xy[right_recv], rank+1, rank, nelems, &requests[92] );
            EXCHANGE( &s.br.yy[right_send], &s.br.yy[right_recv], rank+1, rank, nelems, &requests[94] );

            wait_total += 48;
        }

        MPI_Waitall(wait_total, wait_reqs, wait_status );
    }

    POP_RANGE
};
#endif /* end of pragma USE_MPI */

