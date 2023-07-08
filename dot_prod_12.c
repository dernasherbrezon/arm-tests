#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>

#define TAPS_SIZE 12
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
        0.317034472508947400
    };

#define FIRST_50_EXPECTED_LEN 50

const float FIRST_50_EXPECTED[] = {
    0.044846, 0.048746, 0.052646, 0.056545, 0.060445, 0.064345, 0.068244, 0.072144, 0.076044, 0.079943, 0.083843, 0.087743, 0.091642, 0.095542, 0.099442, 0.103341, 0.107241, 0.111141, 0.115040, 0.118940, 0.122840, 0.126739, 0.130639, 0.134539, 0.138438, 0.142338, 0.146238, 0.150137, 0.154037,
    0.157937, 0.161836, 0.165736, 0.169636, 0.173535, 0.177435, 0.181335, 0.185234, 0.189134, 0.193034, 0.196933, 0.200833, 0.204733, 0.208632, 0.212532, 0.216432, 0.220331, 0.224231, 0.228131, 0.232030, 0.235930
};

#if defined(TEST_GENERIC)

void dot_prod(float *result,
                            const float *input,
                            const float *taps) {

  const float *aPtr = input;
  const float *bPtr = taps;

  *result = bPtr[0] * (aPtr[0] + aPtr[24 - 1])
            + bPtr[1] * (aPtr[1] + aPtr[24 - 2])
            + bPtr[2] * (aPtr[2] + aPtr[24 - 3])
            + bPtr[3] * (aPtr[3] + aPtr[24 - 4])
            + bPtr[4] * (aPtr[4] + aPtr[24 - 5])
            + bPtr[5] * (aPtr[5] + aPtr[24 - 6])
            + bPtr[6] * (aPtr[6] + aPtr[24 - 7])
            + bPtr[7] * (aPtr[7] + aPtr[24 - 8])
            + bPtr[8] * (aPtr[8] + aPtr[24 - 9])
            + bPtr[9] * (aPtr[9] + aPtr[24 - 10])
            + bPtr[10] * (aPtr[10] + aPtr[24 - 11])
            + bPtr[11] * (aPtr[11] + aPtr[24 - 12]);
}

#elif defined(TEST_UNROLL4)

static inline void dot_prod(float *result,
                            const float *input,
                            const float *taps,
                            int output_len) {

  const float *aPtr = input;
  const float *bPtr = taps;
  for (int i = 0; i < output_len / 4; i++) {
    result[4 * i] = bPtr[0] * (aPtr[0] + aPtr[24 - 1])
                    + bPtr[1] * (aPtr[1] + aPtr[24 - 2])
                    + bPtr[2] * (aPtr[2] + aPtr[24 - 3])
                    + bPtr[3] * (aPtr[3] + aPtr[24 - 4])
                    + bPtr[4] * (aPtr[4] + aPtr[24 - 5])
                    + bPtr[5] * (aPtr[5] + aPtr[24 - 6])
                    + bPtr[6] * (aPtr[6] + aPtr[24 - 7])
                    + bPtr[7] * (aPtr[7] + aPtr[24 - 8])
                    + bPtr[8] * (aPtr[8] + aPtr[24 - 9])
                    + bPtr[9] * (aPtr[9] + aPtr[24 - 10])
                    + bPtr[10] * (aPtr[10] + aPtr[24 - 11])
                    + bPtr[11] * (aPtr[11] + aPtr[24 - 12]);

    result[4 * i + 1] = bPtr[0] * (aPtr[1] + aPtr[25 - 1])
                        + bPtr[1] * (aPtr[2] + aPtr[25 - 2])
                        + bPtr[2] * (aPtr[3] + aPtr[25 - 3])
                        + bPtr[3] * (aPtr[4] + aPtr[25 - 4])
                        + bPtr[4] * (aPtr[5] + aPtr[25 - 5])
                        + bPtr[5] * (aPtr[6] + aPtr[25 - 6])
                        + bPtr[6] * (aPtr[7] + aPtr[25 - 7])
                        + bPtr[7] * (aPtr[8] + aPtr[25 - 8])
                        + bPtr[8] * (aPtr[9] + aPtr[25 - 9])
                        + bPtr[9] * (aPtr[10] + aPtr[25 - 10])
                        + bPtr[10] * (aPtr[11] + aPtr[25 - 11])
                        + bPtr[11] * (aPtr[12] + aPtr[25 - 12]);

    result[4 * i + 2] = bPtr[0] * (aPtr[2] + aPtr[26 - 1])
                        + bPtr[1] * (aPtr[3] + aPtr[26 - 2])
                        + bPtr[2] * (aPtr[4] + aPtr[26 - 3])
                        + bPtr[3] * (aPtr[5] + aPtr[26 - 4])
                        + bPtr[4] * (aPtr[6] + aPtr[26 - 5])
                        + bPtr[5] * (aPtr[7] + aPtr[26 - 6])
                        + bPtr[6] * (aPtr[8] + aPtr[26 - 7])
                        + bPtr[7] * (aPtr[9] + aPtr[26 - 8])
                        + bPtr[8] * (aPtr[10] + aPtr[26 - 9])
                        + bPtr[9] * (aPtr[11] + aPtr[26 - 10])
                        + bPtr[10] * (aPtr[12] + aPtr[26 - 11])
                        + bPtr[11] * (aPtr[13] + aPtr[26 - 12]);

    result[4 * i + 3] = bPtr[0] * (aPtr[3] + aPtr[27 - 1])
                        + bPtr[1] * (aPtr[4] + aPtr[27 - 2])
                        + bPtr[2] * (aPtr[5] + aPtr[27 - 3])
                        + bPtr[3] * (aPtr[6] + aPtr[27 - 4])
                        + bPtr[4] * (aPtr[7] + aPtr[27 - 5])
                        + bPtr[5] * (aPtr[8] + aPtr[27 - 6])
                        + bPtr[6] * (aPtr[9] + aPtr[27 - 7])
                        + bPtr[7] * (aPtr[10] + aPtr[27 - 8])
                        + bPtr[8] * (aPtr[11] + aPtr[27 - 9])
                        + bPtr[9] * (aPtr[12] + aPtr[27 - 10])
                        + bPtr[10] * (aPtr[13] + aPtr[27 - 11])
                        + bPtr[11] * (aPtr[14] + aPtr[27 - 12]);
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
  size_t input_size = 262144;
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