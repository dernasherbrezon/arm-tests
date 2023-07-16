#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>

#define TAPS_SIZE 13
float TAPS[TAPS_SIZE] =
    {
        -0.000998606272947510,
        0.001695637278417295,
        -0.003054430179754289,
        0.005055504379767936,
        -0.007901319195893647,
        0.011873357051047719,
        -0.017411159379930066,
        0.025304817427568772,
        -0.037225225204559217,
        0.057533286997004301,
        -0.102327462004259350,
        0.317034472508947400,
        0.500000000000000000
    };

#define FIRST_50_EXPECTED_LEN 50

const float FIRST_50_EXPECTED[] = {
    0.179536, 0.187342, 0.195148, 0.202954, 0.210760, 0.218566, 0.226372, 0.234178, 0.241984, 0.249789, 0.257595, 0.265401, 0.273207, 0.281013, 0.288819, 0.296625, 0.304431, 0.312237, 0.320043, 0.327849, 0.335655, 0.343460, 0.351266, 0.359072, 0.366878, 0.374684, 0.382490, 0.390296, 0.398102,
    0.405908, 0.413714, 0.421520, 0.429326, 0.437132, 0.444937, 0.452743, 0.460549, 0.468355, 0.476161, 0.483967, 0.491773, 0.499579, 0.507385, 0.515191, 0.522996, 0.530803, 0.538609, 0.546414, 0.554220, 0.562026
};

#if defined(TEST_GENERIC)

void dot_prod(float *result,
                            const float *input,
                            const float *taps) {

  const float *aPtr = input;
  const float *bPtr = taps;

  *result = bPtr[0] * (aPtr[0] + aPtr[46])
            + bPtr[1] * (aPtr[2] + aPtr[44])
            + bPtr[2] * (aPtr[4] + aPtr[42])
            + bPtr[3] * (aPtr[6] + aPtr[40])
            + bPtr[4] * (aPtr[8] + aPtr[38])
            + bPtr[5] * (aPtr[10] + aPtr[36])
            + bPtr[6] * (aPtr[12] + aPtr[34])
            + bPtr[7] * (aPtr[14] + aPtr[32])
            + bPtr[8] * (aPtr[16] + aPtr[30])
            + bPtr[9] * (aPtr[18] + aPtr[28])
            + bPtr[10] * (aPtr[20] + aPtr[26])
            + bPtr[11] * (aPtr[22] + aPtr[24])
            + bPtr[12] * aPtr[23];
}

#elif defined(TEST_UNROLL4)

static inline void dot_prod(float *result,
                            const float *input,
                            const float *taps,
                            int output_len) {

  const float *aPtr = input;
  const float *bPtr = taps;
  for (int i = 0; i < output_len / 4; i++) {

    result[4 * i] = bPtr[0] * (aPtr[0] + aPtr[46])
            + bPtr[1] * (aPtr[2] + aPtr[44])
            + bPtr[2] * (aPtr[4] + aPtr[42])
            + bPtr[3] * (aPtr[6] + aPtr[40])
            + bPtr[4] * (aPtr[8] + aPtr[38])
            + bPtr[5] * (aPtr[10] + aPtr[36])
            + bPtr[6] * (aPtr[12] + aPtr[34])
            + bPtr[7] * (aPtr[14] + aPtr[32])
            + bPtr[8] * (aPtr[16] + aPtr[30])
            + bPtr[9] * (aPtr[18] + aPtr[28])
            + bPtr[10] * (aPtr[20] + aPtr[26])
            + bPtr[11] * (aPtr[22] + aPtr[24])
            + bPtr[12] * aPtr[23];
    result[4 * i + 1] = bPtr[0] * (aPtr[1+0] + aPtr[1+46])
               + bPtr[1] * (aPtr[1+2] + aPtr[1+44])
               + bPtr[2] * (aPtr[1+4] + aPtr[1+42])
               + bPtr[3] * (aPtr[1+6] + aPtr[1+40])
               + bPtr[4] * (aPtr[1+8] + aPtr[1+38])
               + bPtr[5] * (aPtr[1+10] + aPtr[1+36])
               + bPtr[6] * (aPtr[1+12] + aPtr[1+34])
               + bPtr[7] * (aPtr[1+14] + aPtr[1+32])
               + bPtr[8] * (aPtr[1+16] + aPtr[1+30])
               + bPtr[9] * (aPtr[1+18] + aPtr[1+28])
               + bPtr[10] * (aPtr[1+20] + aPtr[1+26])
               + bPtr[11] * (aPtr[1+22] + aPtr[1+24])
               + bPtr[12] * aPtr[1+23];

    result[4 * i + 2] = bPtr[0] * (aPtr[2+0] + aPtr[2+46])
               + bPtr[1] * (aPtr[2+2] + aPtr[2+44])
               + bPtr[2] * (aPtr[2+4] + aPtr[2+42])
               + bPtr[3] * (aPtr[2+6] + aPtr[2+40])
               + bPtr[4] * (aPtr[2+8] + aPtr[2+38])
               + bPtr[5] * (aPtr[2+10] + aPtr[2+36])
               + bPtr[6] * (aPtr[2+12] + aPtr[2+34])
               + bPtr[7] * (aPtr[2+14] + aPtr[2+32])
               + bPtr[8] * (aPtr[2+16] + aPtr[2+30])
               + bPtr[9] * (aPtr[2+18] + aPtr[2+28])
               + bPtr[10] * (aPtr[2+20] + aPtr[2+26])
               + bPtr[11] * (aPtr[2+22] + aPtr[2+24])
               + bPtr[12] * aPtr[2+23];


    result[4 * i + 3]  = bPtr[0] * (aPtr[3+0] + aPtr[3+46])
               + bPtr[1] * (aPtr[3+2] + aPtr[3+44])
               + bPtr[2] * (aPtr[3+4] + aPtr[3+42])
               + bPtr[3] * (aPtr[3+6] + aPtr[3+40])
               + bPtr[4] * (aPtr[3+8] + aPtr[3+38])
               + bPtr[5] * (aPtr[3+10] + aPtr[3+36])
               + bPtr[6] * (aPtr[3+12] + aPtr[3+34])
               + bPtr[7] * (aPtr[3+14] + aPtr[3+32])
               + bPtr[8] * (aPtr[3+16] + aPtr[3+30])
               + bPtr[9] * (aPtr[3+18] + aPtr[3+28])
               + bPtr[10] * (aPtr[3+20] + aPtr[3+26])
               + bPtr[11] * (aPtr[3+22] + aPtr[3+24])
               + bPtr[12] * aPtr[3+23];
    aPtr += 4;
  }
}

#else
static inline void dot_prod(float *result,
                            const float *input,
                            const float *taps,
                            int output_len) {


}
#endif

int main(int argc, char **argv) {
  size_t input_size = 200000;
  float *input = malloc(sizeof(float) * input_size);
  if (input == NULL) {
    return EXIT_FAILURE;
  }
  for (size_t i = 0; i < input_size; i++) {
    // don't care about the loss of data
    input[i] = ((float) (i)) / 128.0f;
  }
  size_t output_len = ((input_size - TAPS_SIZE - 1) / 4) * 4;
  float *output = malloc(sizeof(float) * output_len);
  if (output == NULL) {
    return EXIT_FAILURE;
  }
  int total_executions = 50;
  clock_t begin = clock();
  for (int i = 0; i < total_executions; i++) {
#if defined(TEST_UNROLL4)
    dot_prod(output, input, TAPS, output_len);
#else
    for (int j = 0; j < output_len; j++) {
      dot_prod(&output[j], input + j, TAPS);
    }
#endif
  }
  clock_t end = clock();
  double time_spent = (double) (end - begin) / CLOCKS_PER_SEC;
  printf("%f\n", time_spent / total_executions);
  // validation is here to make sure assembly implementation is correct + -O2 optimization won't throw away calculations completely
  for (int i = 0; i < FIRST_50_EXPECTED_LEN; i++) {
    if (((int) (output[i] * 1000)) != ((int) (FIRST_50_EXPECTED[i] * 1000))) {
      printf("invalid output at index %d. expected: %f got %f\n", i, FIRST_50_EXPECTED[i], output[i]);
      return EXIT_FAILURE;
    }
  }
  return EXIT_SUCCESS;
}