#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define HB_KERNEL_FLOAT_LEN 47
#if defined(TEST_ALIGN_MEMORY_8B)
  #define MEMORY_ALIGNMENT 8
#elif defined(TEST_ALIGN_MEMORY_16B)
  #define MEMORY_ALIGNMENT 16
#else
  #define MEMORY_ALIGNMENT 1
#endif

float HB_KERNEL_FLOAT[HB_KERNEL_FLOAT_LEN] =
    {
        -0.000998606272947510,
        0.000000000000000000,
        0.001695637278417295,
        0.000000000000000000,
        -0.003054430179754289,
        0.000000000000000000,
        0.005055504379767936,
        0.000000000000000000,
        -0.007901319195893647,
        0.000000000000000000,
        0.011873357051047719,
        0.000000000000000000,
        -0.017411159379930066,
        0.000000000000000000,
        0.025304817427568772,
        0.000000000000000000,
        -0.037225225204559217,
        0.000000000000000000,
        0.057533286997004301,
        0.000000000000000000,
        -0.102327462004259350,
        0.000000000000000000,
        0.317034472508947400,
        0.500000000000000000,
        0.317034472508947400,
        0.000000000000000000,
        -0.102327462004259350,
        0.000000000000000000,
        0.057533286997004301,
        0.000000000000000000,
        -0.037225225204559217,
        0.000000000000000000,
        0.025304817427568772,
        0.000000000000000000,
        -0.017411159379930066,
        0.000000000000000000,
        0.011873357051047719,
        0.000000000000000000,
        -0.007901319195893647,
        0.000000000000000000,
        0.005055504379767936,
        0.000000000000000000,
        -0.003054430179754289,
        0.000000000000000000,
        0.001695637278417295,
        0.000000000000000000,
        -0.000998606272947510
    };

#define FIRST_50_EXPECTED_LEN 50

const float FIRST_50_EXPECTED[] = {
    0.179536, 0.187342, 0.195148, 0.202954, 0.210760, 0.218566, 0.226372, 0.234178, 0.241984, 0.249789, 0.257595, 0.265401, 0.273207, 0.281013, 0.288819, 0.296625, 0.304431, 0.312237, 0.320043, 0.327849, 0.335655, 0.343460, 0.351266, 0.359072, 0.366878, 0.374684, 0.382490, 0.390296, 0.398102,
    0.405908, 0.413714, 0.421520, 0.429326, 0.437132, 0.444937, 0.452743, 0.460549, 0.468355, 0.476161, 0.483967, 0.491773, 0.499579, 0.507385, 0.515191, 0.522996, 0.530803, 0.538609, 0.546414, 0.554220, 0.562026
};

#if defined(TEST_GENERIC)

static inline void dot_prod(float *result,
              const float *input,
              const float *taps,
              unsigned int num_points) {

  float dotProduct = 0;
#if defined(TEST_NONALIGN_MEMORY)
  const float *aPtr = input;
  const float *bPtr = taps;
#else
  const float *aPtr = (float *) __builtin_assume_aligned(input, MEMORY_ALIGNMENT);
  const float *bPtr = (float *) __builtin_assume_aligned(taps, MEMORY_ALIGNMENT);
#endif
  unsigned int number = 0;

  for (number = 0; number < num_points; number++) {
    dotProduct += ((*aPtr++) * (*bPtr++));
  }

  *result = dotProduct;
}

#elif defined(TEST_GENSYMM)

static inline void dot_prod(float *result,
              const float *input,
              const float *taps,
              unsigned int num_points) {

  float dotProduct = 0;
#if defined(TEST_NONALIGN_MEMORY)
  const float *aPtr = input;
  const float *bPtr = taps;
#else
  const float *aPtr = (float *) __builtin_assume_aligned(input, MEMORY_ALIGNMENT);
  const float *bPtr = (float *) __builtin_assume_aligned(taps, MEMORY_ALIGNMENT);
#endif
  unsigned int number = 0;

  for (number = 0; number < num_points / 2; number++) {
    dotProduct += bPtr[number] * (aPtr[number] + aPtr[num_points - number - 1] );
  }
  dotProduct += bPtr[number] * aPtr[number];
  *result = dotProduct;
}

#elif defined(TEST_NEON1QSYMM)

#include <arm_neon.h>

static inline void dot_prod(float *result,
              const float *input,
              const float *taps,
              unsigned int num_points) {

  float dotProduct = 0;
#if defined(TEST_NONALIGN_MEMORY)
  const float *aPtr = input;
  const float *bPtr = taps;
  const float *cPtr = input + num_points;
#else
  const float *aPtr = (float *) __builtin_assume_aligned(input, MEMORY_ALIGNMENT);
  const float *bPtr = (float *) __builtin_assume_aligned(taps, MEMORY_ALIGNMENT);
  const float *cPtr = (float *) __builtin_assume_aligned(input + num_points, MEMORY_ALIGNMENT);
#endif
  unsigned int number = 0;

  unsigned int quarter_points = num_points / 2 / 4;
  float32x4_t a_val, b_val, r_val, accumulator0;
  accumulator0 = vdupq_n_f32(0);
  for (number = 0; number < quarter_points; number++) {
    cPtr -= 4;

    a_val = vld1q_f32(aPtr);
    b_val = vld1q_f32(bPtr);
    r_val = vld1q_f32(cPtr);

    float32x2_t lo = vrev64_f32(vget_high_f32(r_val));
    float32x2_t hi = vrev64_f32(vget_low_f32(r_val));
    r_val = vcombine_f32(lo, hi);

    accumulator0 = vmlaq_f32(accumulator0, vaddq_f32(a_val, r_val), b_val);

    aPtr += 4;
    bPtr += 4;
  }
    float accumulator[4];
    vst1q_f32(accumulator, accumulator0);
    dotProduct = accumulator[0] + accumulator[1] + accumulator[2] + accumulator[3];
  for( number = quarter_points * 4; number <num_points / 2; number++  ) {
    cPtr--;
    dotProduct += (*bPtr++) * ((*aPtr++) + (*cPtr) );
  }
  dotProduct += (*bPtr) * (*aPtr);
  *result = dotProduct;
}

#elif defined(TEST_NEON4Q)

#include <arm_neon.h>

static inline void dot_prod(float* result,
                                                     const float* input,
                                                     const float* taps,
                                                     unsigned int num_points)
{

    unsigned int quarter_points = num_points / 16;
    float dotProduct = 0;
#if defined(TEST_NONALIGN_MEMORY)
  const float *aPtr = input;
  const float *bPtr = taps;
#else
  const float *aPtr = (float *) __builtin_assume_aligned(input, MEMORY_ALIGNMENT);
  const float *bPtr = (float *) __builtin_assume_aligned(taps, MEMORY_ALIGNMENT);
#endif
    unsigned int number = 0;

    float32x4x4_t a_val, b_val, accumulator0;
    accumulator0.val[0] = vdupq_n_f32(0);
    accumulator0.val[1] = vdupq_n_f32(0);
    accumulator0.val[2] = vdupq_n_f32(0);
    accumulator0.val[3] = vdupq_n_f32(0);
    // factor of 4 loop unroll with independent accumulators
    // uses 12 out of 16 neon q registers
    for (number = 0; number < quarter_points; ++number) {
        a_val = vld4q_f32(aPtr);
        b_val = vld4q_f32(bPtr);
#if defined(TEST_PREFETCH)
        __builtin_prefetch(aPtr+16);
        __builtin_prefetch(bPtr+16);
#endif
        accumulator0.val[0] = vmlaq_f32(accumulator0.val[0], a_zval.val[0], b_val.val[0]);
        accumulator0.val[1] = vmlaq_f32(accumulator0.val[1], a_val.val[1], b_val.val[1]);
        accumulator0.val[2] = vmlaq_f32(accumulator0.val[2], a_val.val[2], b_val.val[2]);
        accumulator0.val[3] = vmlaq_f32(accumulator0.val[3], a_val.val[3], b_val.val[3]);
        aPtr += 16;
        bPtr += 16;
    }
    accumulator0.val[0] = vaddq_f32(accumulator0.val[0], accumulator0.val[1]);
    accumulator0.val[2] = vaddq_f32(accumulator0.val[2], accumulator0.val[3]);
    accumulator0.val[0] = vaddq_f32(accumulator0.val[2], accumulator0.val[0]);
    float accumulator[4];
    vst1q_f32(accumulator, accumulator0.val[0]);
    dotProduct = accumulator[0] + accumulator[1] + accumulator[2] + accumulator[3];

    for (number = quarter_points * 16; number < num_points; number++) {
        dotProduct += ((*aPtr++) * (*bPtr++));
    }

    *result = dotProduct;
}

#elif defined(TEST_NEON1Q)

#include <arm_neon.h>

static inline void dot_prod(float* result,
                                                 const float* input,
                                                 const float* taps,
                                                 unsigned int num_points)
{

    unsigned int quarter_points = num_points / 4;
    float dotProduct = 0;
#if defined(TEST_NONALIGN_MEMORY)
  const float *aPtr = input;
  const float *bPtr = taps;
#else
  const float *aPtr = (float *) __builtin_assume_aligned(input, MEMORY_ALIGNMENT);
  const float *bPtr = (float *) __builtin_assume_aligned(taps, MEMORY_ALIGNMENT);
#endif
    unsigned int number = 0;

    float32x4_t a_val, b_val, accumulator_val;
    accumulator_val = vdupq_n_f32(0);
    for (number = 0; number < quarter_points; ++number) {
        a_val = vld1q_f32(aPtr);
        b_val = vld1q_f32(bPtr);
#if defined(TEST_PREFETCH)
        __builtin_prefetch(aPtr+4);
        __builtin_prefetch(bPtr+4);
#endif
        accumulator_val =
            vmlaq_f32(accumulator_val, a_val, b_val);
        aPtr += 4;
        bPtr += 4;
    }
    float accumulator[4];
    vst1q_f32(accumulator, accumulator_val);
    dotProduct = accumulator[0] + accumulator[1] + accumulator[2] + accumulator[3];

    for (number = quarter_points * 4; number < num_points; number++) {
        dotProduct += ((*aPtr++) * (*bPtr++));
    }

    *result = dotProduct;
}
#elif defined(TEST_NEON2QI)

#include <arm_neon.h>

static inline void dot_prod(float *result,
                            const float *input,
                            const float *taps,
                            unsigned int num_points) {

  unsigned int quarter_points = num_points / 16;
  float dotProduct = 0;
#if defined(TEST_NONALIGN_MEMORY)
  const float *aPtr = input;
  const float *bPtr = taps;
#else
  const float *aPtr = (float *) __builtin_assume_aligned(input, MEMORY_ALIGNMENT);
  const float *bPtr = (float *) __builtin_assume_aligned(taps, MEMORY_ALIGNMENT);
#endif
  unsigned int number = 0;

  float32x4x2_t a_val, b_val, c_val, d_val;
  float32x4x4_t accumulator0;
  accumulator0.val[0] = vdupq_n_f32(0);
  accumulator0.val[1] = vdupq_n_f32(0);
  accumulator0.val[2] = vdupq_n_f32(0);
  accumulator0.val[3] = vdupq_n_f32(0);

  a_val = vld2q_f32(aPtr);
  b_val = vld2q_f32(bPtr);

  aPtr += 8;
  bPtr += 8;

  // factor of 2 loop unroll with independent accumulators
  for (number = 0; number < quarter_points; ++number) {
    c_val = vld2q_f32(aPtr);
    d_val = vld2q_f32(bPtr);

    accumulator0.val[0] =
        vmlaq_f32(accumulator0.val[0], a_val.val[0], b_val.val[0]);
    accumulator0.val[1] =
        vmlaq_f32(accumulator0.val[1], a_val.val[1], b_val.val[1]);

    aPtr += 8;
    bPtr += 8;

    if (number == (quarter_points - 1)) {
      break;
    }

    a_val = vld2q_f32(aPtr);
    b_val = vld2q_f32(bPtr);

    accumulator0.val[2] =
        vmlaq_f32(accumulator0.val[2], c_val.val[0], d_val.val[0]);
    accumulator0.val[3] =
        vmlaq_f32(accumulator0.val[3], c_val.val[1], d_val.val[1]);

    aPtr += 8;
    bPtr += 8;
  }
  accumulator0.val[2] =
      vmlaq_f32(accumulator0.val[2], c_val.val[0], d_val.val[0]);
  accumulator0.val[3] =
      vmlaq_f32(accumulator0.val[3], c_val.val[1], d_val.val[1]);

  accumulator0.val[0] = vaddq_f32(accumulator0.val[0], accumulator0.val[1]);
  accumulator0.val[2] = vaddq_f32(accumulator0.val[2], accumulator0.val[3]);
  accumulator0.val[0] = vaddq_f32(accumulator0.val[2], accumulator0.val[0]);
  float accumulator[4];
  vst1q_f32(accumulator, accumulator0.val[0]);
  dotProduct = accumulator[0] + accumulator[1] + accumulator[2] + accumulator[3];

  for (number = quarter_points * 16; number < num_points; number++) {
    dotProduct += ((*aPtr++) * (*bPtr++));
  }

  *result = dotProduct;
}

#elif defined(TEST_NEON1QI)

#include <arm_neon.h>

static inline void dot_prod(float* result,
                                                 const float* input,
                                                 const float* taps,
                                                 unsigned int num_points)
{

    unsigned int quarter_points = num_points / 8;
    float dotProduct = 0;
#if defined(TEST_NONALIGN_MEMORY)
  const float *aPtr = input;
  const float *bPtr = taps;
#else
  const float *aPtr = (float *) __builtin_assume_aligned(input, MEMORY_ALIGNMENT);
  const float *bPtr = (float *) __builtin_assume_aligned(taps, MEMORY_ALIGNMENT);
#endif
    unsigned int number = 0;

    float32x4_t a_val, b_val, c_val, d_val, accumulator_val, accumulator_val1;
    accumulator_val = vdupq_n_f32(0);
    accumulator_val1 = vdupq_n_f32(0);

    a_val = vld1q_f32(aPtr);
    b_val = vld1q_f32(bPtr);
    aPtr += 4;
    bPtr += 4;
    for (number = 0; number < quarter_points; ++number) {
        c_val = vld1q_f32(aPtr);
        d_val = vld1q_f32(bPtr);

        accumulator_val =
            vmlaq_f32(accumulator_val, a_val, b_val);

        aPtr += 4;
        bPtr += 4;

        if (number == (quarter_points - 1)) {
          break;
        }

        a_val = vld1q_f32(aPtr);
        b_val = vld1q_f32(bPtr);

        accumulator_val1 =
            vmlaq_f32(accumulator_val1, c_val, d_val);

        aPtr += 4;
        bPtr += 4;
    }
    accumulator_val1 =
        vmlaq_f32(accumulator_val1, c_val, d_val);
    accumulator_val = vaddq_f32(accumulator_val1, accumulator_val);
    float accumulator[4];
    vst1q_f32(accumulator, accumulator_val);
    dotProduct = accumulator[0] + accumulator[1] + accumulator[2] + accumulator[3];

    for (number = quarter_points * 8; number < num_points; number++) {
        dotProduct += ((*aPtr++) * (*bPtr++));
    }

    *result = dotProduct;
}

#elif defined(TEST_NEON2Q)

#include <arm_neon.h>

static inline void dot_prod(float* result,
                                                 const float* input,
                                                 const float* taps,
                                                 unsigned int num_points)
{

    unsigned int quarter_points = num_points / 8;
    float dotProduct = 0;
#if defined(TEST_NONALIGN_MEMORY)
  const float *aPtr = input;
  const float *bPtr = taps;
#else
  const float *aPtr = (float *) __builtin_assume_aligned(input, MEMORY_ALIGNMENT);
  const float *bPtr = (float *) __builtin_assume_aligned(taps, MEMORY_ALIGNMENT);
#endif
    unsigned int number = 0;

    float32x4x2_t a_val, b_val, accumulator_val;
    accumulator_val.val[0] = vdupq_n_f32(0);
    accumulator_val.val[1] = vdupq_n_f32(0);
    // factor of 2 loop unroll with independent accumulators
    for (number = 0; number < quarter_points; ++number) {
        a_val = vld2q_f32(aPtr);
        b_val = vld2q_f32(bPtr);
#if defined(TEST_PREFETCH)
        __builtin_prefetch(aPtr+8);
        __builtin_prefetch(bPtr+8);
#endif
        accumulator_val.val[0] =
            vmlaq_f32(accumulator_val.val[0], a_val.val[0], b_val.val[0]);
        accumulator_val.val[1] =
            vmlaq_f32(accumulator_val.val[1], a_val.val[1], b_val.val[1]);
        aPtr += 8;
        bPtr += 8;
    }
    accumulator_val.val[0] = vaddq_f32(accumulator_val.val[0], accumulator_val.val[1]);
    float accumulator[4];
    vst1q_f32(accumulator, accumulator_val.val[0]);
    dotProduct = accumulator[0] + accumulator[1] + accumulator[2] + accumulator[3];

    for (number = quarter_points * 8; number < num_points; number++) {
        dotProduct += ((*aPtr++) * (*bPtr++));
    }

    *result = dotProduct;
}

#elif defined(TEST_GENCACHE)

void dot_prod(float *result,
                      const float *input,
                      const float *taps,
                      unsigned int num_points,
                      size_t output_len) {

  const float *aPtr = input;
  const float *bPtr = taps;

  for( int i =0;i<output_len;i++ ) {
    result[i] = bPtr[0]  * (aPtr[0]  + aPtr[47 - 1])
			+ bPtr[1]  * (aPtr[1]  + aPtr[47 - 2])
			+ bPtr[2]  * (aPtr[2]  + aPtr[47 - 3])
			+ bPtr[3]  * (aPtr[3]  + aPtr[47 - 4])

        + bPtr[4]  * (aPtr[4]  + aPtr[47 - 5])
			+ bPtr[5]  * (aPtr[5]  + aPtr[47 - 6])
			+ bPtr[6]  * (aPtr[6]  + aPtr[47 - 7])
			+ bPtr[7]  * (aPtr[7]  + aPtr[47 - 8])

			+ bPtr[8]  * (aPtr[8]  + aPtr[47 - 9])
			+ bPtr[9]  * (aPtr[9]  + aPtr[47 - 10])
			+ bPtr[10] * (aPtr[10] + aPtr[47 - 11])
			+ bPtr[11] * (aPtr[11] + aPtr[47 - 12])

			+ bPtr[12]  * (aPtr[12]  + aPtr[47 - 13])
			+ bPtr[13]  * (aPtr[13]  + aPtr[47 - 14])
			+ bPtr[14]  * (aPtr[14]  + aPtr[47 - 15])
			+ bPtr[15]  * (aPtr[15]  + aPtr[47 - 16])

			+ bPtr[16]  * (aPtr[16]  + aPtr[47 - 17])
			+ bPtr[17]  * (aPtr[17]  + aPtr[47 - 18])
			+ bPtr[18]  * (aPtr[18]  + aPtr[47 - 19])
			+ bPtr[19]  * (aPtr[19]  + aPtr[47 - 20])

			+ bPtr[20] * (aPtr[20] + aPtr[47 - 21])
			+ bPtr[21] * (aPtr[21] + aPtr[47 - 22])
			+ bPtr[22] * (aPtr[22] + aPtr[47 - 23])
			+ bPtr[23] * (aPtr[23] + 0);

    aPtr++;
  }
}

#elif defined(TEST_VOLK)

#include <volk/volk.h>

static inline void dot_prod(float *result,
                      const float *input,
                      const float *taps,
                      unsigned int num_points) {

  volk_32f_x2_dot_prod_32f(result, input, taps, num_points);
}

#else

void dot_prod(float *result,
                      const float *input,
                      const float *taps,
                      unsigned int num_points) {

}

#endif

int main(int argc, char **argv) {
  size_t input_size = 262144;
  size_t aligned_input_size = input_size;
  unsigned int taps_size = HB_KERNEL_FLOAT_LEN;
  unsigned int aligned_taps_size = taps_size;
  float *taps = HB_KERNEL_FLOAT;

  float **aligned_taps = NULL;
  int number_of_aligned_taps = MEMORY_ALIGNMENT / sizeof(float);
#if !defined(TEST_NONALIGN_MEMORY)
  // Make a set of taps at all possible alignments
  float **result = malloc(number_of_aligned_taps * sizeof(float *));
  if (result == NULL) {
    return EXIT_FAILURE;
  }
  unsigned int aligned_to_memory_taps_size = taps_size + number_of_aligned_taps - 1;
  for (int i = 0; i < number_of_aligned_taps; i++) {
    posix_memalign((void **) &(result[i]), MEMORY_ALIGNMENT, sizeof(float) * aligned_to_memory_taps_size);
    // some taps will be longer than original, but
    // since they contain zeros, multiplication on an input will produce 0
    // there is a tradeoff: multiply unaligned input or
    // multiply aligned input but with additional zeros
    memset(result[i], 0, sizeof(float) * aligned_to_memory_taps_size);
    memcpy(result[i] + i, taps, sizeof(float) * taps_size);
  }
  // re-define taps sizes so that they can aligned later on
  taps_size = aligned_to_memory_taps_size;
  aligned_taps_size = taps_size;
  aligned_taps = result;
#endif

#if !defined(TEST_NONALIGN_MEMORY)
  int align_size;
#if (defined(TEST_NEON1Q))
  align_size = 4;
#elif (defined(TEST_NEON2Q) || defined(TEST_NEON1QI))
  align_size = 8;
#elif (defined(TEST_NEON4Q) || defined(TEST_NEON2QI))
  align_size = 16;
#else
  align_size = 1;
#endif
  if (align_size != 1) {
    if (input_size % align_size != 0) {
      aligned_input_size = ((input_size / align_size) + 1) * align_size;
    }
    if (taps_size % align_size != 0) {
      aligned_taps_size = ((taps_size / align_size) + 1) * align_size;
#if defined(TEST_NOALIGN_MEMORY)
      taps = malloc(sizeof(float) * aligned_taps_size);
      memset(taps, 0, sizeof(float) * aligned_taps_size);
      memcpy(taps, HB_KERNEL_FLOAT, sizeof(float) * taps_size);
#else
      for (int i = 0; i < number_of_aligned_taps; i++) {
        float *aligned_by_memory_and_size = NULL;
        posix_memalign((void **) &(aligned_by_memory_and_size), MEMORY_ALIGNMENT, sizeof(float) * aligned_taps_size);
        memset(aligned_by_memory_and_size, 0, sizeof(float) * aligned_taps_size);
        memcpy(aligned_by_memory_and_size, result[i], sizeof(float) * taps_size);
        free(result[i]);
        result[i] = aligned_by_memory_and_size;
      }
#endif
    }
  }
#endif

  float *input = NULL;
#if defined(TEST_NONALIGN_MEMORY)
  input = malloc(sizeof(float) * aligned_input_size);
#else
  int memory_code = posix_memalign((void **) &input, MEMORY_ALIGNMENT, sizeof(float) * aligned_input_size);
  if (memory_code != 0) {
    return EXIT_FAILURE;
  }
#endif
  if (input == NULL) {
    return EXIT_FAILURE;
  }


  for (size_t i = 0; i < input_size; i++) {
    // don't care about the loss of data
    input[i] = ((float) (i)) / 128.0f;
  }

  float *output = NULL;
  size_t output_len = input_size - aligned_taps_size - 1;
#if defined(TEST_NONALIGN_MEMORY)
  output = malloc(sizeof(float) * output_len);
#else
  memory_code = posix_memalign((void **) &output, MEMORY_ALIGNMENT, sizeof(float) * output_len);
  if (memory_code != 0) {
    return EXIT_FAILURE;
  }
#endif
  if (output == NULL) {
    return EXIT_FAILURE;
  }
  //printf("input: %p\noutput: %p\ntaps: %p\ntaps len: %d\n", input, output, taps, aligned_taps_size);
  int total_executions = 50;
  clock_t begin = clock();
  for (int i = 0; i < total_executions; i++) {
#if defined(TEST_GENCACHE)
    dot_prod(output, input, taps, aligned_taps_size, output_len);
#else
    for (int j = 0; j < output_len; j++) {
#if defined(TEST_NONALIGN_MEMORY)
      dot_prod(&output[j], input + j, taps, aligned_taps_size);
#else
      const float *buf = (const float *) (input + j);
      const float *aligned_buffer = (const float *) ((size_t) buf & ~(MEMORY_ALIGNMENT - 1));
      unsigned align_index = buf - aligned_buffer;
      dot_prod(&output[j], aligned_buffer, aligned_taps[align_index], aligned_taps_size);
#endif
    }
#endif
  }

  clock_t end = clock();
  double time_spent = (double) (end - begin) / CLOCKS_PER_SEC;
  printf("%f", time_spent / total_executions);
  // validation is here to make sure assembly implementation is correct + -O2 optimization won't throw away calculations completely
  for (int i = 0; i < FIRST_50_EXPECTED_LEN; i++) {
    if (((int) (output[i] * 1000)) != ((int) (FIRST_50_EXPECTED[i] * 1000))) {
      printf("invalid output at index %d. expected: %f got %f\n", i, FIRST_50_EXPECTED[i], output[i]);
      return EXIT_FAILURE;
    }
  }
  return EXIT_SUCCESS;
}