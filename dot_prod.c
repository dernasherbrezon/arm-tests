#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define HB_KERNEL_FLOAT_LEN 47

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

void dot_prod(float *result,
              const float *input,
              const float *taps,
              unsigned int num_points) {

  float dotProduct = 0;
#if defined(TEST_ALIGN_MEMORY)
  const float* aPtr = (float *)__builtin_assume_aligned(input, 128);
  const float* bPtr = (float *)__builtin_assume_aligned(taps, 128);
#else
  const float *aPtr = input;
  const float *bPtr = taps;
#endif
  unsigned int number = 0;

  for (number = 0; number < num_points; number++) {
    dotProduct += ((*aPtr++) * (*bPtr++));
  }

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
#if defined(TEST_ALIGN_MEMORY)
  const float* aPtr = (float *)__builtin_assume_aligned(input, 128);
  const float* bPtr = (float *)__builtin_assume_aligned(taps, 128);
#else
  const float *aPtr = input;
  const float *bPtr = taps;
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
        accumulator0.val[0] = vmlaq_f32(accumulator0.val[0], a_val.val[0], b_val.val[0]);
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
#if defined(TEST_ALIGN_MEMORY)
  const float* aPtr = (float *)__builtin_assume_aligned(input, 128);
  const float* bPtr = (float *)__builtin_assume_aligned(taps, 128);
#else
  const float *aPtr = input;
  const float *bPtr = taps;
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
#if defined(TEST_ALIGN_MEMORY)
  const float* aPtr = (float *)__builtin_assume_aligned(input, 128);
  const float* bPtr = (float *)__builtin_assume_aligned(taps, 128);
#else
  const float *aPtr = input;
  const float *bPtr = taps;
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

    if (number == (quarter_points - 1)) {
      break;
    }

    aPtr += 8;
    bPtr += 8;

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
#if defined(TEST_ALIGN_MEMORY)
  const float* aPtr = (float *)__builtin_assume_aligned(input, 128);
  const float* bPtr = (float *)__builtin_assume_aligned(taps, 128);
#else
  const float *aPtr = input;
  const float *bPtr = taps;
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

        if (number == (quarter_points - 1)) {
          break;
        }

        aPtr += 4;
        bPtr += 4;

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
#if defined(TEST_ALIGN_MEMORY)
  const float* aPtr = (float *)__builtin_assume_aligned(input, 128);
  const float* bPtr = (float *)__builtin_assume_aligned(taps, 128);
#else
  const float *aPtr = input;
  const float *bPtr = taps;
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

#else

void dot_prod(float *result,
                      const float *input,
                      const float *taps,
                      unsigned int num_points) {

}

#endif

int main(int argc, char **argv) {
  size_t max_input = 262144;
  size_t memory_size = max_input;
  unsigned int num_points = HB_KERNEL_FLOAT_LEN;
  float *taps = HB_KERNEL_FLOAT;
#if (defined(TEST_NEON1Q) || defined(TEST_NEON1QI) || defined(TEST_GENERIC)) && defined(TEST_ALIGN_SIZE)
  if( max_input % 4 != 0 ) {
    memory_size = ((max_input / 4) + 1) * 4;
  }
  if( num_points % 4 != 0 ) {
    num_points = ((num_points / 4) + 1) * 4;
    taps = malloc(sizeof(float) * num_points);
    memset(taps, 0, sizeof(float) * num_points);
    memcpy(taps, HB_KERNEL_FLOAT, sizeof(float) * HB_KERNEL_FLOAT_LEN);
  }
#elif (defined(TEST_NEON2Q) || defined(TEST_NEON2QI)) && defined(TEST_ALIGN_SIZE)
  if( max_input % 8 != 0 ) {
    memory_size = ((max_input / 8) + 1) * 8;
  }
  if( num_points % 8 != 0 ) {
    num_points = ((num_points / 8) + 1) * 8;
    taps = malloc(sizeof(float) * num_points);
    memset(taps, 0, sizeof(float) * num_points);
    memcpy(taps, HB_KERNEL_FLOAT, sizeof(float) * HB_KERNEL_FLOAT_LEN);
  }
#elif defined(TEST_NEON4Q) && defined(TEST_ALIGN_SIZE)
  if( max_input % 16 != 0 ) {
    memory_size = ((max_input / 16) + 1) * 16;
  }
  if( num_points % 16 != 0 ) {
    num_points = ((num_points / 16) + 1) * 16;
    taps = malloc(sizeof(float) * num_points);
    memset(taps, 0, sizeof(float) * num_points);
    memcpy(taps, HB_KERNEL_FLOAT, sizeof(float) * HB_KERNEL_FLOAT_LEN);
  }
#endif
  float *input = NULL;
#if defined(TEST_ALIGN_MEMORY)
  int memory_code = posix_memalign((void **) &input, 128, sizeof(float) * memory_size);
  if (memory_code != 0) {
    return EXIT_FAILURE;
  }
#else
  input = malloc(sizeof(float) * max_input);
#endif
  if (input == NULL) {
    return EXIT_FAILURE;
  }
  printf("input: %p\n", input);
  for (size_t i = 0; i < max_input; i++) {
    // don't care about the loss of data
    input[i] = ((float) (i)) / 128.0f;
  }

  float *output = NULL;
  size_t output_len = max_input - num_points - 1;
#if defined(TEST_ALIGN_MEMORY)
  memory_code = posix_memalign((void **) &output, 128, sizeof(float) * output_len);
  if (memory_code != 0) {
    return EXIT_FAILURE;
  }
#else
  output = malloc(sizeof(float) * output_len);
#endif
  if (output == NULL) {
    return EXIT_FAILURE;
  }
  //printf("actual memory allocated: %zu number of points: %d input %p output %p\n", memory_size, num_points, input, output);
  int total_executions = 50;
  clock_t begin = clock();
  for (int i = 0; i < total_executions; i++) {
    for (int j = 0; j < output_len; j++) {
      dot_prod(&output[j], input + j, taps, num_points);
    }
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